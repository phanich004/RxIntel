"""Retrieval dispatch gates — mocks every downstream retriever.

No live Neo4j, Chroma, or Groq calls. The parallelism test uses
``time.sleep(0.2)`` inside each mocked retriever and asserts the
wall-clock total stays under 0.35s, which would be impossible under
serial execution.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

from agent.nodes.retrieval_dispatch import retrieval_dispatch


def _graph_row(drug_id: str, qtype: str = "enzyme_filter") -> dict[str, Any]:
    return {
        "drug_id": drug_id,
        "drug_name": drug_id,
        "actions": ["inhibitor"],
        "query_type": qtype,
    }


def _vector_row(drug_id: str) -> dict[str, Any]:
    return {
        "drug_id": drug_id,
        "drug_name": drug_id,
        "collection": "indications",
        "text": "...",
        "cosine": 0.5,
        "chunk_index": 0,
        "query_type": "vector",
    }


def test_ddi_check_runs_graph_only() -> None:
    graph_rows = [_graph_row("DB00001"), _graph_row("DB00002")]
    state: dict[str, Any] = {
        "query": "is warfarin safe with aspirin",
        "mode": "ddi_check",
        "resolved_drugs": [{"drug_id": "DB00682"}, {"drug_id": "DB00945"}],
    }
    with (
        patch(
            "agent.nodes.retrieval_dispatch.graph_retriever",
            return_value={"graph_results": graph_rows},
        ) as g,
        patch(
            "agent.nodes.retrieval_dispatch.vector_retriever",
            return_value={"vector_results": []},
        ) as v,
    ):
        out = retrieval_dispatch(state)

    g.assert_called_once_with(state)
    v.assert_not_called()
    assert out["graph_results"] == graph_rows
    assert out.get("vector_results", []) == []
    assert out["fused_results"], "expected fusion to emit non-empty output"
    ids = {r["drug_id"] for r in out["fused_results"]}
    assert ids == {"DB00001", "DB00002"}


def test_polypharmacy_runs_graph_only() -> None:
    graph_rows = [_graph_row("DB00001")]
    state: dict[str, Any] = {"query": "", "mode": "polypharmacy"}
    with (
        patch(
            "agent.nodes.retrieval_dispatch.graph_retriever",
            return_value={"graph_results": graph_rows},
        ) as g,
        patch(
            "agent.nodes.retrieval_dispatch.vector_retriever",
            return_value={"vector_results": []},
        ) as v,
    ):
        out = retrieval_dispatch(state)

    g.assert_called_once()
    v.assert_not_called()
    assert out["graph_results"] == graph_rows
    assert out["fused_results"]


def test_alternatives_runs_vector_only() -> None:
    vector_rows = [_vector_row("DB00003"), _vector_row("DB00004")]
    state: dict[str, Any] = {"query": "alt to warfarin", "mode": "alternatives"}
    with (
        patch(
            "agent.nodes.retrieval_dispatch.graph_retriever",
            return_value={"graph_results": []},
        ) as g,
        patch(
            "agent.nodes.retrieval_dispatch.vector_retriever",
            return_value={"vector_results": vector_rows},
        ) as v,
    ):
        out = retrieval_dispatch(state)

    g.assert_not_called()
    v.assert_called_once_with(state)
    assert out.get("graph_results", []) == []
    assert out["vector_results"] == vector_rows
    ids = {r["drug_id"] for r in out["fused_results"]}
    assert ids == {"DB00003", "DB00004"}


def test_describe_runs_vector_only() -> None:
    vector_rows = [_vector_row("DB13928")]
    state: dict[str, Any] = {"query": "mechanism of semaglutide", "mode": "describe"}
    with (
        patch(
            "agent.nodes.retrieval_dispatch.graph_retriever",
            return_value={"graph_results": []},
        ) as g,
        patch(
            "agent.nodes.retrieval_dispatch.vector_retriever",
            return_value={"vector_results": vector_rows},
        ) as v,
    ):
        out = retrieval_dispatch(state)

    g.assert_not_called()
    v.assert_called_once()
    assert out["vector_results"] == vector_rows
    assert len(out["fused_results"]) == 1
    assert out["fused_results"][0]["drug_id"] == "DB13928"


def test_hybrid_runs_both_retrievers_and_merges() -> None:
    graph_rows = [_graph_row("DB00001"), _graph_row("DB00002")]
    vector_rows = [_vector_row("DB00001"), _vector_row("DB00005")]
    state: dict[str, Any] = {
        "query": "CYP3A4 inhibitors interacting with statins",
        "mode": "hybrid",
        "semantic_constraint": "statin",
    }
    with (
        patch(
            "agent.nodes.retrieval_dispatch.graph_retriever",
            return_value={"graph_results": graph_rows},
        ) as g,
        patch(
            "agent.nodes.retrieval_dispatch.vector_retriever",
            return_value={"vector_results": vector_rows},
        ) as v,
    ):
        out = retrieval_dispatch(state)

    g.assert_called_once()
    v.assert_called_once()
    assert out["graph_results"] == graph_rows
    assert out["vector_results"] == vector_rows
    ids = [r["drug_id"] for r in out["fused_results"]]
    assert set(ids) == {"DB00001", "DB00002", "DB00005"}
    # DB00001 appears in BOTH lists → must outrank either solo drug.
    assert ids[0] == "DB00001", (
        f"RRF boost failed: expected DB00001 first, got {ids}"
    )


def test_unknown_mode_returns_empty_triple() -> None:
    state: dict[str, Any] = {"query": "anything", "mode": "bogus"}
    with (
        patch("agent.nodes.retrieval_dispatch.graph_retriever") as g,
        patch("agent.nodes.retrieval_dispatch.vector_retriever") as v,
    ):
        out = retrieval_dispatch(state)

    g.assert_not_called()
    v.assert_not_called()
    assert out == {"graph_results": [], "vector_results": [], "fused_results": []}


def test_hybrid_runs_retrievers_in_parallel() -> None:
    """Each mocked retriever sleeps 0.2s. Serial execution would take
    ~0.4s; the ThreadPoolExecutor should finish well under 0.35s."""

    def slow_graph(state: Any) -> dict[str, Any]:
        time.sleep(0.2)
        return {"graph_results": [_graph_row("DB00001")]}

    def slow_vector(state: Any) -> dict[str, Any]:
        time.sleep(0.2)
        return {"vector_results": [_vector_row("DB00002")]}

    state: dict[str, Any] = {"query": "x", "mode": "hybrid"}
    with (
        patch(
            "agent.nodes.retrieval_dispatch.graph_retriever",
            side_effect=slow_graph,
        ),
        patch(
            "agent.nodes.retrieval_dispatch.vector_retriever",
            side_effect=slow_vector,
        ),
    ):
        t0 = time.perf_counter()
        out = retrieval_dispatch(state)
        elapsed = time.perf_counter() - t0

    assert elapsed < 0.35, (
        f"hybrid dispatch took {elapsed:.3f}s — serial execution suspected"
    )
    assert {r["drug_id"] for r in out["fused_results"]} == {"DB00001", "DB00002"}
