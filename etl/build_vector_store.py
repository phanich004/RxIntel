"""
build_vector_store.py
======================
Stream the DrugBank XML into five ChromaDB collections — one per text
field type — for semantic retrieval.

Design notes
------------
* Same streaming `iterparse` + top-level-drug filter + group filter as
  `parse_drugbank.py`. We import those helpers rather than duplicating
  the iteration / invariant logic.
* One collection per text type (NOT one mega-collection):
    descriptions, indications, mechanisms, pharmacodynamics,
    contraindications
  Rationale: an "alternatives to warfarin" query should hit the
  `indications` collection, not mix with toxicity language.
* Source fields:
    descriptions      <- <description>
    indications       <- <indication>
    mechanisms        <- <mechanism-of-action>
    pharmacodynamics  <- <pharmacodynamics>
    contraindications <- <toxicity> + regex-extracted warning paragraphs
                         from <description>
* Chunking:
    - If field text <= 1000 tokens -> one chunk = whole field.
    - Else: sentence-packed 200-token windows, 50-token overlap.
  Token counts use the HF tokenizer of the embedding model (BERT
  WordPiece for all-MiniLM-L6-v2) so "1000 tokens" here is what the
  model actually sees.
* Embedding: sentence-transformers/all-MiniLM-L6-v2 via
  ChromaDB's `SentenceTransformerEmbeddingFunction`.
* Metadata per chunk: drug_id, drug_name, collection, chunk_index,
  groups. Chroma requires scalar metadata, so `groups` is a
  comma-separated string.

Usage
-----
    # Fresh build
    python etl/build_vector_store.py --xml data/full_database.xml

    # Dry-run: chunk count per collection, no Chroma writes
    python etl/build_vector_store.py --xml data/full_database.xml --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Disable HF tokenizer thread pool. Without this, the first embedding call
# inside chromadb.add() deadlocks against chromadb's own fork/multiprocessing
# state on macOS. Must be set before transformers / sentence_transformers load.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from tqdm import tqdm

# Re-use the Step-2 parser's namespace + streaming loop + group filter.
sys.path.insert(0, str(Path(__file__).parent))
from parse_drugbank import (  # type: ignore[import-not-found]
    ALLOWED_GROUPS_DEFAULT,
    T,
    drug_groups,
    iter_drugs,
    primary_id,
)

# -------------------------------------------------------------------- #
# Config
# -------------------------------------------------------------------- #
COLLECTIONS: List[str] = [
    "descriptions",
    "indications",
    "mechanisms",
    "pharmacodynamics",
    "contraindications",
]

# Direct XML-tag-per-collection mapping. `contraindications` is synthesized
# downstream from <toxicity> + warning paragraphs in <description>.
FIELD_TAGS: Dict[str, str] = {
    "descriptions": "description",
    "indications": "indication",
    "mechanisms": "mechanism-of-action",
    "pharmacodynamics": "pharmacodynamics",
}

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# MiniLM-L6-v2 max context is 256 tokens (trained). Cap at 240 for
# [CLS]/[SEP] headroom — no chunk we hand to the embedder exceeds the
# model's real context window.
CHUNK_TOKEN_THRESHOLD = 240
WINDOW_TOKENS = 200
OVERLAP_TOKENS = 50
BATCH_SIZE = 512  # chunks per Chroma write

# Warning/caution patterns used to pull contraindication-relevant
# paragraphs out of the free-text <description>. Paragraph-level so the
# embedded text carries enough context for retrieval.
WARNING_PATTERN = re.compile(
    r"(?i)\b("
    r"contraindicated|contraindication|"
    r"should not be (?:used|administered|given|taken)|"
    r"must not be (?:used|administered|given|taken)|"
    r"avoid (?:the )?(?:use|administration|coadministration)|"
    r"not recommended (?:in|for)|"
    r"warning|caution(?:ed)?|"
    r"black[- ]?box|boxed warning|"
    r"hypersensitivity"
    r")\b"
)

# Sentence splitter: capture `.!?` followed by whitespace + capital letter.
# Good enough for DrugBank prose; avoids the Punkt/spaCy model download.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


# -------------------------------------------------------------------- #
# Tokenizer (lazy)
# -------------------------------------------------------------------- #
_tokenizer = None


def _tok():
    """Return a cached HF tokenizer matching the embedding model."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count WordPiece tokens (no special tokens) for the embed model."""
    if not text:
        return 0
    return len(_tok().encode(text, add_special_tokens=False))


# -------------------------------------------------------------------- #
# Chunking
# -------------------------------------------------------------------- #
def split_sentences(text: str) -> List[str]:
    """Regex sentence split — adequate for drug-monograph prose."""
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _hard_split_sentence(sentence: str, max_tokens: int) -> List[str]:
    """
    Token-slice a single sentence whose length exceeds ``max_tokens``.

    Guarantees every returned piece is <= max_tokens per the embed-model
    tokenizer, so no piece can overrun MiniLM's 256-token context.
    """
    tok = _tok()
    ids = tok.encode(sentence, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return [sentence]
    pieces: List[str] = []
    for start in range(0, len(ids), max_tokens):
        piece_ids = ids[start : start + max_tokens]
        pieces.append(tok.decode(piece_ids, skip_special_tokens=True))
    return pieces


def chunk_text(text: str) -> List[str]:
    """
    Return 1+ chunks for a text field.

    Strategy:
      - If the whole field is <= CHUNK_TOKEN_THRESHOLD tokens, return
        a single chunk containing the full text.
      - Otherwise, sentence-pack into WINDOW_TOKENS windows with
        OVERLAP_TOKENS of tail carry-over between consecutive windows.
      - Any single sentence longer than WINDOW_TOKENS is hard-split at
        the tokenizer level first so no chunk can exceed MiniLM's
        256-token real context window.
    """
    text = (text or "").strip()
    if not text:
        return []

    if count_tokens(text) <= CHUNK_TOKEN_THRESHOLD:
        return [text]

    sentences = split_sentences(text)
    if not sentences:
        return [text]

    expanded: List[str] = []
    for s in sentences:
        if count_tokens(s) > WINDOW_TOKENS:
            expanded.extend(_hard_split_sentence(s, WINDOW_TOKENS))
        else:
            expanded.append(s)
    sentences = expanded

    chunks: List[str] = []
    cur: List[str] = []
    cur_tok = 0
    i = 0
    while i < len(sentences):
        s = sentences[i]
        s_tok = count_tokens(s)

        if cur_tok + s_tok <= WINDOW_TOKENS or not cur:
            cur.append(s)
            cur_tok += s_tok
            i += 1
            continue

        chunks.append(" ".join(cur))
        # Overlap tail must leave room for the next sentence, else the
        # same tail keeps re-forming and we spin forever.
        tail_budget = min(OVERLAP_TOKENS, WINDOW_TOKENS - s_tok)
        tail: List[str] = []
        tail_tok = 0
        if tail_budget > 0:
            for s_prev in reversed(cur):
                t = count_tokens(s_prev)
                if tail_tok + t > tail_budget:
                    break
                tail.insert(0, s_prev)
                tail_tok += t
        cur = tail
        cur_tok = tail_tok

    if cur:
        chunks.append(" ".join(cur))
    return chunks


# -------------------------------------------------------------------- #
# Per-drug field extraction
# -------------------------------------------------------------------- #
def _field_text(drug_elem, tag: str) -> str:
    el = drug_elem.find(T(tag))
    if el is None or not el.text:
        return ""
    return el.text.strip()


def extract_fields(drug_elem, description: str) -> Dict[str, str]:
    """
    Return `{collection_name: text}` for each collection that has content
    for this drug. `description` is passed in to avoid a second XML lookup.
    """
    out: Dict[str, str] = {}

    for coll, tag in FIELD_TAGS.items():
        txt = _field_text(drug_elem, tag)
        if txt:
            out[coll] = txt

    # Contraindications: <toxicity> plus paragraphs in <description> that
    # match the warning pattern. Paragraph-level granularity preserves
    # clinical context in the embedded chunk.
    contra_parts: List[str] = []
    tox = _field_text(drug_elem, "toxicity")
    if tox:
        contra_parts.append(tox)
    if description:
        for para in re.split(r"\n\s*\n", description):
            para = para.strip()
            if para and WARNING_PATTERN.search(para):
                contra_parts.append(para)
    if contra_parts:
        out["contraindications"] = "\n\n".join(contra_parts)

    return out


# -------------------------------------------------------------------- #
# Main
# -------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> None:
    persist_dir = Path(args.persist_dir).resolve()
    allowed_groups: Optional[set] = (
        None if args.include_all else ALLOWED_GROUPS_DEFAULT
    )

    # Lazy Chroma setup so --dry-run doesn't require the DB.
    collections = None
    model = None
    if not args.dry_run:
        import chromadb
        from chromadb.utils import embedding_functions
        from sentence_transformers import SentenceTransformer

        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir))

        # Load the SentenceTransformer ourselves so we pre-compute embeddings
        # and hand them to chromadb via `embeddings=`. Letting chromadb invoke
        # SentenceTransformerEmbeddingFunction from inside `add()` deadlocks
        # on macOS against chromadb's multiprocessing state.
        print(f"→ Loading embedder {EMBED_MODEL}", file=sys.stderr, flush=True)
        model = SentenceTransformer(EMBED_MODEL)

        # Keep the embedding_function attached for query-time use (so
        # `collection.query(query_texts=[...])` works after the build).
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )

        # Fresh rebuild: drop + recreate each collection so chunk counts
        # are deterministic across reruns.
        for coll in COLLECTIONS:
            try:
                client.delete_collection(coll)
            except Exception:  # noqa: BLE001 — collection may not exist
                pass
        collections = {
            coll: client.create_collection(coll, embedding_function=embed_fn)
            for coll in COLLECTIONS
        }

    # Per-collection write buffers.
    buf: Dict[str, Dict[str, List]] = {
        coll: {"ids": [], "docs": [], "metas": []} for coll in COLLECTIONS
    }

    def flush(coll: str) -> None:
        b = buf[coll]
        if not b["ids"]:
            return
        if collections is not None and model is not None:
            embeds = model.encode(
                b["docs"],
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ).tolist()
            collections[coll].add(
                ids=b["ids"],
                documents=b["docs"],
                metadatas=b["metas"],
                embeddings=embeds,
            )
        b["ids"].clear()
        b["docs"].clear()
        b["metas"].clear()

    per_coll_chunks: Dict[str, int] = {c: 0 for c in COLLECTIONS}
    per_coll_drugs: Dict[str, int] = {c: 0 for c in COLLECTIONS}
    n_drugs_seen = n_drugs_kept = 0

    print(f"→ Streaming drugs from {args.xml}", file=sys.stderr)
    for drug_elem in tqdm(iter_drugs(args.xml), unit="drug"):
        n_drugs_seen += 1
        dbid = primary_id(drug_elem)
        if not dbid:
            continue
        name_elem = drug_elem.find(T("name"))
        name = (
            name_elem.text.strip()
            if name_elem is not None and name_elem.text
            else ""
        )
        if not name:
            continue

        groups = drug_groups(drug_elem)
        if allowed_groups is not None and not (groups & allowed_groups):
            continue
        n_drugs_kept += 1

        description = _field_text(drug_elem, "description")
        fields = extract_fields(drug_elem, description)
        groups_str = ",".join(sorted(groups))

        for coll, text in fields.items():
            chunks = chunk_text(text)
            if not chunks:
                continue
            per_coll_drugs[coll] += 1
            per_coll_chunks[coll] += len(chunks)
            for idx, chunk in enumerate(chunks):
                buf[coll]["ids"].append(f"{dbid}:{idx}")
                buf[coll]["docs"].append(chunk)
                buf[coll]["metas"].append(
                    {
                        "drug_id": dbid,
                        "drug_name": name,
                        "collection": coll,
                        "chunk_index": idx,
                        "groups": groups_str,
                    }
                )
            if len(buf[coll]["ids"]) >= BATCH_SIZE:
                flush(coll)

    for coll in COLLECTIONS:
        flush(coll)

    print("\n" + "=" * 60, file=sys.stderr)
    print("VECTOR BUILD SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Drugs seen in XML:     {n_drugs_seen:,}", file=sys.stderr)
    print(f"Drugs kept:            {n_drugs_kept:,}", file=sys.stderr)
    print(f"{'collection':22s}{'drugs':>10s}{'chunks':>12s}", file=sys.stderr)
    for coll in COLLECTIONS:
        print(
            f"  {coll:20s}{per_coll_drugs[coll]:>10,d}{per_coll_chunks[coll]:>12,d}",
            file=sys.stderr,
        )
    if args.dry_run:
        print("\n(dry-run — no Chroma writes)", file=sys.stderr)
    else:
        print(f"\nPersisted to {persist_dir}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml", required=True, help="Path to unzipped DrugBank XML")
    p.add_argument(
        "--persist-dir",
        default="./chroma_db",
        help="Chroma PersistentClient path (default: ./chroma_db)",
    )
    p.add_argument(
        "--include-all",
        action="store_true",
        help="Keep all drug groups (default: approved+investigational only)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk; print summary; do not write to Chroma",
    )
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
