"""ChromaDB vector retriever with per-mode collection dispatch.

Modes and the collections each one reads from:

* ``alternatives`` — ``indications`` + ``contraindications``. Query text
  is the router-extracted ``semantic_constraint`` if present, else the
  raw ``query``. Top-5 per collection, then merge+dedupe by drug_id.
* ``describe`` — ``descriptions`` + ``mechanisms`` + ``pharmacodynamics``.
  If exactly one drug is resolved, a metadata filter pins results to
  that drug — we want chunks ABOUT the named drug, not semantic
  lookalikes.
* ``hybrid`` — ``indications`` + ``contraindications``, restricted via a
  ``drug_id $in [...]`` metadata filter to the drugs the graph
  retriever already qualified (enzyme filter). Top-10 per collection
  since the candidate pool is pre-narrowed.
* ``ddi_check`` / ``polypharmacy`` — no-op; graph owns these modes.

We compute query embeddings ourselves against a pre-warmed
SentenceTransformer and hand them to ``collection.query()`` via
``query_embeddings=``. Letting Chroma invoke the embedding function
inside ``query()`` deadlocks on macOS (same failure mode the ETL build
in Step 3 hit); ``query_embeddings=`` side-steps the whole class.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence

from dotenv import load_dotenv

from agent.schemas import AgentState

load_dotenv()
logger = logging.getLogger(__name__)

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_TOP_K_PER_COLLECTION = 5
_TOP_K_HYBRID = 10

ALTERNATIVES_COLLECTIONS: tuple[str, ...] = ("indications", "contraindications")
DESCRIBE_COLLECTIONS: tuple[str, ...] = (
    "descriptions",
    "mechanisms",
    "pharmacodynamics",
)
HYBRID_COLLECTIONS: tuple[str, ...] = ("indications", "contraindications")

_model: Any = None
_client: Any = None
_collections: dict[str, Any] = {}


def _get_model() -> Any:
    """Lazy singleton SentenceTransformer. Cached model loads from disk."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_EMBED_MODEL)
    return _model


def _get_client() -> Any:
    """Lazy singleton ChromaDB PersistentClient.

    CHROMA_PERSIST_DIR has a sensible non-secret default so we keep the
    plain ``os.environ.get`` fallback here — unlike Neo4j credentials,
    which must fail loudly via ``_require_env``.
    """
    global _client
    if _client is None:
        import chromadb

        persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
        _client = chromadb.PersistentClient(path=persist_dir)
    return _client


def _get_collection(name: str) -> Any:
    if name not in _collections:
        _collections[name] = _get_client().get_collection(name)
    return _collections[name]


def _embed(text: str) -> list[float]:
    vec = _get_model().encode(
        [text], convert_to_numpy=True, show_progress_bar=False
    )
    return list(vec[0].tolist())


def _query_one(
    collection: str,
    query_text: str,
    top_k: int,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Single-collection Chroma query, shaped to the retriever row spec."""
    col = _get_collection(collection)
    kwargs: dict[str, Any] = {
        "query_embeddings": [_embed(query_text)],
        "n_results": top_k,
    }
    if where:
        kwargs["where"] = where
    res = col.query(**kwargs)
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    rows: list[dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists, strict=False):
        drug_id = meta.get("drug_id") if meta else None
        if not drug_id:
            continue
        rows.append(
            {
                "drug_id": drug_id,
                "drug_name": meta.get("drug_name", ""),
                "collection": collection,
                "text": doc,
                "cosine": 1.0 - float(dist),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "query_type": "vector",
            }
        )
    return rows


def _dedupe_by_drug(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse rows to one-per-drug keeping the highest cosine.

    Each surviving row also gets a ``source_collections`` list naming
    the collections that contributed — downstream consumers can tell
    "two collections both surfaced this drug" from "only one did".
    """
    best: dict[str, dict[str, Any]] = {}
    sources: dict[str, list[str]] = {}
    for r in rows:
        did = r["drug_id"]
        sources.setdefault(did, []).append(r["collection"])
        prev = best.get(did)
        if prev is None or r["cosine"] > prev["cosine"]:
            best[did] = r
    out: list[dict[str, Any]] = []
    for did, row in best.items():
        seen: set[str] = set()
        uniq: list[str] = []
        for c in sources[did]:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        out.append({**row, "source_collections": uniq})
    out.sort(key=lambda r: -r["cosine"])
    return out


def _hybrid_where(
    graph_results: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    """Build the Chroma metadata filter pinning results to graph hits."""
    drug_ids = sorted({r["drug_id"] for r in graph_results if r.get("drug_id")})
    if not drug_ids:
        return None
    return {"drug_id": {"$in": drug_ids}}


def _describe_where(
    resolved_drugs: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    """Single-drug filter so describe pulls chunks ABOUT that drug."""
    if len(resolved_drugs) == 1 and resolved_drugs[0].get("drug_id"):
        return {"drug_id": {"$eq": resolved_drugs[0]["drug_id"]}}
    return None


def vector_retriever(state: AgentState) -> AgentState:
    """LangGraph node: per-mode Chroma dispatch, writes ``vector_results``."""
    mode = state.get("mode")
    if mode in (None, "ddi_check", "polypharmacy"):
        return {"vector_results": []}

    raw_query = state.get("query", "") or ""
    constraint = state.get("semantic_constraint") or ""

    if mode == "alternatives":
        qtext = constraint or raw_query
        if not qtext:
            return {"vector_results": []}
        rows: list[dict[str, Any]] = []
        for coll in ALTERNATIVES_COLLECTIONS:
            rows.extend(_query_one(coll, qtext, _TOP_K_PER_COLLECTION))
        return {"vector_results": _dedupe_by_drug(rows)}

    if mode == "describe":
        if not raw_query:
            return {"vector_results": []}
        resolved = state.get("resolved_drugs", []) or []
        where = _describe_where(resolved)
        rows = []
        for coll in DESCRIBE_COLLECTIONS:
            rows.extend(
                _query_one(coll, raw_query, _TOP_K_PER_COLLECTION, where=where)
            )
        return {"vector_results": _dedupe_by_drug(rows)}

    if mode == "hybrid":
        qtext = constraint or raw_query
        graph_results = state.get("graph_results", []) or []
        where = _hybrid_where(graph_results)
        if not qtext or where is None:
            return {"vector_results": []}
        rows = []
        for coll in HYBRID_COLLECTIONS:
            rows.extend(_query_one(coll, qtext, _TOP_K_HYBRID, where=where))
        return {"vector_results": _dedupe_by_drug(rows)}

    return {"vector_results": []}
