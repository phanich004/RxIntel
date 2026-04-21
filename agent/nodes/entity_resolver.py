"""Three-tier drug-name resolver (invariant #6).

Tier 1 — rapidfuzz over the gazetteer built in Step 4 (~140k normalized
names). Token-set-ratio with score_cutoff=85, exact hits win outright.

Tier 2 — Med7 clinical NER. Runs only when tier 1 yields nothing. If the
``en_core_med7_lg`` pipeline is not installed we log a warning once and
behave as if tier 2 returned empty, so the resolver still falls through
to tier 3 instead of crashing the graph.

Tier 3 — semantic nearest-neighbor over the ChromaDB ``descriptions``
collection built in Step 3. The spec calls this "FAISS over description
embeddings"; ChromaDB's HNSW index is the same thing topologically and
reuses embeddings we already paid for, so we query that directly rather
than rebuilding a FAISS index. Top-3 drug ids are returned.

The module exposes a single ``entity_resolver`` LangGraph node function
and a reusable ``resolve()`` helper.
"""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from agent.schemas import AgentState
from etl.build_gazetteer import normalize

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
GAZETTEER_EXACT_PATH = REPO_ROOT / "gazetteer.pkl"
GAZETTEER_FUZZY_PATH = REPO_ROOT / "gazetteer_fuzzy.pkl"
CHROMA_PATH = REPO_ROOT / "chroma_db"

TIER1_CUTOFF = 85
TIER2_CUTOFF = 75
MAX_SPAN_WORDS = 4
MIN_SPAN_CHARS = 3
TIER3_TOP_K = 3

# Pure-English glue we never want to hand to rapidfuzz as a single-token
# candidate. Short drug names ("HCL", "INH") are preserved by the
# MIN_SPAN_CHARS gate + uppercase forms being normalized separately.
_GLUE_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "and", "or", "but", "if", "then", "so",
        "with", "without", "from", "to", "of", "in", "on", "at", "by",
        "for", "as", "into", "about", "over", "under",
        "do", "does", "did", "can", "could", "will", "would", "should",
        "how", "what", "why", "when", "who", "which",
        "my", "your", "our", "his", "her", "its", "their",
        "i", "we", "you", "he", "she", "it", "they",
        "safe", "safely", "ok", "okay", "fine",
        "take", "taking", "use", "using", "mix", "mixing", "combine",
        "interaction", "interactions", "interact",
        "patient", "patients",
    }
)

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]*")


def _candidate_spans(query: str, max_n: int = MAX_SPAN_WORDS) -> list[str]:
    """Generate 1..n-gram spans from the query, skipping pure-glue spans."""
    tokens = _TOKEN_RE.findall(query)
    tokens_lc = [t.lower() for t in tokens]
    spans: list[str] = []
    seen: set[str] = set()
    for i in range(len(tokens_lc)):
        for n in range(1, max_n + 1):
            j = i + n
            if j > len(tokens_lc):
                break
            window = tokens_lc[i:j]
            if all(t in _GLUE_STOPWORDS for t in window):
                continue
            span = " ".join(window)
            if len(span) < MIN_SPAN_CHARS:
                continue
            if span in seen:
                continue
            seen.add(span)
            spans.append(span)
    return spans


def _dedupe_best(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the best hit per drug_id; sort by score desc.

    On score ties, prefer the shorter matched name — a single-token exact
    hit ("ibuprofen") beats a multi-word brand that scored 100 only
    because the query tokens were a subset ("comfort pac with ibuprofen").
    """
    best: dict[str, dict[str, Any]] = {}
    for h in hits:
        dbid = h["drug_id"]
        prev = best.get(dbid)
        if prev is None:
            best[dbid] = h
            continue
        if h["match_score"] > prev["match_score"]:
            best[dbid] = h
            continue
        if h["match_score"] == prev["match_score"] and len(h["name"]) < len(prev["name"]):
            best[dbid] = h
    return sorted(best.values(), key=lambda r: -r["match_score"])


@dataclass
class EntityResolver:
    """Owns gazetteer + lazy Med7/chromadb handles for the three tiers."""

    gaz_exact_path: Path = GAZETTEER_EXACT_PATH
    gaz_fuzzy_path: Path = GAZETTEER_FUZZY_PATH
    chroma_path: Path = CHROMA_PATH

    _exact: dict[str, str] = field(init=False)
    _fuzzy_choices: list[str] = field(init=False)
    _id_by_name: dict[str, str] = field(init=False)
    _med7_state: str = field(default="unloaded", init=False)
    _med7: Any = field(default=None, init=False)
    _chroma_col: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        with self.gaz_exact_path.open("rb") as f:
            self._exact = pickle.load(f)
        with self.gaz_fuzzy_path.open("rb") as f:
            pairs: list[tuple[str, str]] = pickle.load(f)
        self._fuzzy_choices = [name for name, _ in pairs]
        self._id_by_name = dict(pairs)

    def resolve(self, query: str) -> list[dict[str, Any]]:
        """Three-tier fallback. Returns [] only if all tiers miss."""
        tier1 = self._tier1(query)
        if tier1:
            return tier1
        tier2 = self._tier2(query)
        if tier2:
            return tier2
        return self._tier3(query)

    def _tier1(self, query: str) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for span in _candidate_spans(query):
            norm = normalize(span)
            if not norm or len(norm) < MIN_SPAN_CHARS:
                continue

            if norm in self._exact:
                hits.append(
                    {
                        "drug_id": self._exact[norm],
                        "name": norm,
                        "match_score": 100.0,
                        "tier": 1,
                    }
                )
                continue

            best = process.extractOne(
                norm,
                self._fuzzy_choices,
                scorer=fuzz.token_set_ratio,
                score_cutoff=TIER1_CUTOFF,
            )
            if best is None:
                continue
            matched_name, score, _ = best
            hits.append(
                {
                    "drug_id": self._id_by_name[matched_name],
                    "name": matched_name,
                    "match_score": float(score),
                    "tier": 1,
                }
            )
        return _dedupe_best(hits)

    def _load_med7(self) -> Any:
        if self._med7_state == "loaded":
            return self._med7
        if self._med7_state == "unavailable":
            return None
        try:
            import spacy

            self._med7 = spacy.load("en_core_med7_lg")
            self._med7_state = "loaded"
            return self._med7
        except Exception as exc:  # pragma: no cover - env-dependent
            logger.warning(
                "Med7 unavailable (%s); tier-2 resolution disabled. "
                "Install via `pip install https://huggingface.co/"
                "kormilitzin/en_core_med7_lg/resolve/main/"
                "en_core_med7_lg-any-py3-none-any.whl`",
                exc,
            )
            self._med7_state = "unavailable"
            return None

    def _tier2(self, query: str) -> list[dict[str, Any]]:
        nlp = self._load_med7()
        if nlp is None:
            return []
        doc = nlp(query)
        hits: list[dict[str, Any]] = []
        for ent in doc.ents:
            if ent.label_ != "DRUG":
                continue
            norm = normalize(ent.text)
            if not norm:
                continue
            if norm in self._exact:
                hits.append(
                    {
                        "drug_id": self._exact[norm],
                        "name": norm,
                        "match_score": 100.0,
                        "tier": 2,
                    }
                )
                continue
            best = process.extractOne(
                norm,
                self._fuzzy_choices,
                scorer=fuzz.token_set_ratio,
                score_cutoff=TIER2_CUTOFF,
            )
            if best is None:
                continue
            matched_name, score, _ = best
            hits.append(
                {
                    "drug_id": self._id_by_name[matched_name],
                    "name": matched_name,
                    "match_score": float(score),
                    "tier": 2,
                }
            )
        return _dedupe_best(hits)

    def _load_chroma(self) -> Any:
        if self._chroma_col is not None:
            return self._chroma_col
        import chromadb

        client = chromadb.PersistentClient(path=str(self.chroma_path))
        self._chroma_col = client.get_collection("descriptions")
        return self._chroma_col

    def _tier3(self, query: str) -> list[dict[str, Any]]:
        try:
            col = self._load_chroma()
        except Exception as exc:
            logger.warning("Tier-3 unavailable: %s", exc)
            return []
        res = col.query(query_texts=[query], n_results=TIER3_TOP_K)
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        hits: list[dict[str, Any]] = []
        seen: set[str] = set()
        for meta, dist in zip(metas, dists, strict=False):
            dbid = meta.get("drug_id")
            if not dbid or dbid in seen:
                continue
            seen.add(dbid)
            hits.append(
                {
                    "drug_id": dbid,
                    "name": meta.get("drug_name", ""),
                    # Convert cosine distance to a [0,100] similarity so the
                    # score lives on the same scale as tiers 1/2.
                    "match_score": max(0.0, 100.0 * (1.0 - float(dist))),
                    "tier": 3,
                }
            )
        return hits


_resolver_singleton: EntityResolver | None = None


def _get_resolver() -> EntityResolver:
    global _resolver_singleton
    if _resolver_singleton is None:
        _resolver_singleton = EntityResolver()
    return _resolver_singleton


def entity_resolver(state: AgentState) -> AgentState:
    """LangGraph node: populate ``resolved_drugs`` from ``query``."""
    query = state.get("query", "")
    resolved = _get_resolver().resolve(query) if query else []
    return {"resolved_drugs": resolved}
