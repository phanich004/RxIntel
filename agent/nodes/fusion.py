"""Reciprocal Rank Fusion — merge graph + vector retriever output.

RRF (Cormack et al. 2009) turns two ranked lists into one without
needing calibrated scores: each drug's fused score is the sum of
``1 / (k + rank + 1)`` over every list it appears in, with rank
0-indexed from best. Drugs that appear in both graph and vector
results naturally outrank drugs that only appear in one. We use
``k = 60`` as documented in CLAUDE.md §6.

Graph rows for ``direct`` and ``multi_hop`` are pair-shaped — an
``INTERACTS_WITH`` edge between Warfarin and Aspirin describes BOTH
drugs, not one. ``_expand_graph_rows`` emits one fusion entry per
endpoint so the pair row contributes to the fused rank of Warfarin AND
Aspirin, with the full pair row preserved in each endpoint's lineage.
``enzyme_filter`` rows already carry a single ``drug_id`` and pass
through unchanged.

Node dispatch:
* ``ddi_check`` / ``polypharmacy`` — graph only (single-source wrap)
* ``alternatives`` / ``describe`` — vector only (single-source wrap)
* ``hybrid`` — true two-list RRF merge
* anything else — empty fused_results
"""

from __future__ import annotations

from typing import Any, Iterable

from agent.schemas import AgentState

RRF_K = 60

# query_type values whose rows are pair-shaped and must be expanded to
# two fusion entries (one per endpoint).
_PAIRWISE_TYPES: frozenset[str] = frozenset({"direct", "multi_hop"})


def _expand_graph_rows(
    graph_results: Iterable[dict[str, Any]],
) -> list[tuple[str, str, dict[str, Any]]]:
    """Emit ``(drug_id, drug_name, original_row)`` tuples, one per endpoint.

    Pairwise rows (direct / multi_hop) fan out to two tuples; enzyme
    filter rows emit a single tuple. The original row is carried on
    every tuple so each endpoint's lineage references the same edge.
    """
    out: list[tuple[str, str, dict[str, Any]]] = []
    for row in graph_results:
        qtype = row.get("query_type", "")
        if qtype in _PAIRWISE_TYPES:
            a_id = row.get("drug_a_id")
            b_id = row.get("drug_b_id")
            if a_id:
                out.append((a_id, row.get("drug_a_name", ""), row))
            if b_id:
                out.append((b_id, row.get("drug_b_name", ""), row))
        else:
            did = row.get("drug_id")
            if did:
                out.append((did, row.get("drug_name", ""), row))
    return out


def reciprocal_rank_fusion(
    graph_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Merge two ranked lists by RRF, dedupe on drug_id, preserve lineage.

    Each input row must have a ``drug_id`` key (or, for pairwise graph
    rows, ``drug_a_id`` + ``drug_b_id``). Output rows carry:

    * ``drug_id``
    * ``drug_name`` — first-seen wins if different sources disagree
    * ``rrf_score`` — sum of ``1 / (k + rank + 1)`` across all hits
    * ``graph``  — list of every graph row naming this drug (may be [])
    * ``vector`` — list of every vector row naming this drug (may be [])
    """
    scores: dict[str, float] = {}
    names: dict[str, str] = {}
    graph_lineage: dict[str, list[dict[str, Any]]] = {}
    vector_lineage: dict[str, list[dict[str, Any]]] = {}

    for rank, (did, name, row) in enumerate(_expand_graph_rows(graph_results)):
        scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
        if name:
            names.setdefault(did, name)
        graph_lineage.setdefault(did, []).append(row)

    for rank, row in enumerate(vector_results):
        vdid_raw = row.get("drug_id")
        if not vdid_raw:
            continue
        vdid = str(vdid_raw)
        scores[vdid] = scores.get(vdid, 0.0) + 1.0 / (k + rank + 1)
        vname = str(row.get("drug_name") or "")
        if vname:
            names.setdefault(vdid, vname)
        vector_lineage.setdefault(vdid, []).append(row)

    fused: list[dict[str, Any]] = [
        {
            "drug_id": did,
            "drug_name": names.get(did, ""),
            "rrf_score": score,
            "graph": graph_lineage.get(did, []),
            "vector": vector_lineage.get(did, []),
        }
        for did, score in scores.items()
    ]
    fused.sort(key=lambda r: float(r["rrf_score"]), reverse=True)
    return fused


def _wrap_single_source(
    rows: list[dict[str, Any]],
    source: str,
) -> list[dict[str, Any]]:
    """Shape single-source retriever output into the fused record format.

    Delegates to ``reciprocal_rank_fusion`` with the other side empty so
    single-source modes emit the exact same record shape as hybrid —
    downstream code never has to branch on mode.
    """
    if source == "graph":
        return reciprocal_rank_fusion(rows, [])
    if source == "vector":
        return reciprocal_rank_fusion([], rows)
    return []


def fusion(state: AgentState) -> AgentState:
    """LangGraph node: route to graph-only / vector-only / RRF merge."""
    mode = state.get("mode")
    graph_results = state.get("graph_results") or []
    vector_results = state.get("vector_results") or []

    if mode in ("ddi_check", "polypharmacy"):
        fused = _wrap_single_source(graph_results, "graph")
    elif mode in ("alternatives", "describe"):
        fused = _wrap_single_source(vector_results, "vector")
    elif mode == "hybrid":
        fused = reciprocal_rank_fusion(graph_results, vector_results)
    else:
        fused = []

    return {"fused_results": fused}
