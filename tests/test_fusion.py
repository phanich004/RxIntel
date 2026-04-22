"""Fusion / RRF gates.

Pure-Python tests — no Neo4j, no Chroma, no Groq. The whole suite
should complete in well under 100ms.

Covers:
  1. empty + empty
  2. graph-only (no vector)
  3. vector-only (no graph)
  4. overlap boost — in-both outranks in-one
  5. pairwise expansion (direct row → two endpoints)
  6. same-drug, multiple enzymes → summed RRF
  7. same-drug, graph + multiple vector chunks → combined lineage
  8. node wrapper mode dispatch
  9. ``k`` parameter respected
"""

from __future__ import annotations

import math
from typing import Any

from agent.nodes.fusion import (
    RRF_K,
    _expand_graph_rows,
    fusion,
    reciprocal_rank_fusion,
)


def _enz(drug_id: str, name: str = "") -> dict[str, Any]:
    return {
        "drug_id": drug_id,
        "drug_name": name or drug_id,
        "actions": ["inhibitor"],
        "query_type": "enzyme_filter",
    }


def _vec(drug_id: str, name: str = "", chunk_index: int = 0) -> dict[str, Any]:
    return {
        "drug_id": drug_id,
        "drug_name": name or drug_id,
        "collection": "indications",
        "text": "...",
        "cosine": 0.5,
        "chunk_index": chunk_index,
        "query_type": "vector",
    }


def _direct(
    a_id: str, a_name: str, b_id: str, b_name: str, severity: str = "Major"
) -> dict[str, Any]:
    return {
        "drug_a_id": a_id,
        "drug_a_name": a_name,
        "drug_b_id": b_id,
        "drug_b_name": b_name,
        "severity": severity,
        "description": "...",
        "query_type": "direct",
    }


def test_empty_inputs_return_empty_list() -> None:
    assert reciprocal_rank_fusion([], []) == []


def test_graph_only_preserves_order_empty_vector_lineage() -> None:
    graph = [_enz("DB00001"), _enz("DB00002"), _enz("DB00003")]
    out = reciprocal_rank_fusion(graph, [])
    assert [r["drug_id"] for r in out] == ["DB00001", "DB00002", "DB00003"]
    for r in out:
        assert len(r["graph"]) == 1
        assert r["vector"] == []
    # Scores monotonically decreasing at ranks 0,1,2
    assert out[0]["rrf_score"] > out[1]["rrf_score"] > out[2]["rrf_score"]


def test_vector_only_preserves_order_empty_graph_lineage() -> None:
    vector = [_vec("DB00001"), _vec("DB00002"), _vec("DB00003")]
    out = reciprocal_rank_fusion([], vector)
    assert [r["drug_id"] for r in out] == ["DB00001", "DB00002", "DB00003"]
    for r in out:
        assert r["graph"] == []
        assert len(r["vector"]) == 1


def test_in_both_lists_outranks_in_one_list() -> None:
    """DB00001 hits rank 0 in graph AND rank 0 in vector. DB00002 hits
    only rank 1 in graph. The fused score for DB00001 is 2/(k+1) while
    DB00002 is 1/(k+2), so DB00001 must rank first."""
    graph = [_enz("DB00001"), _enz("DB00002")]
    vector = [_vec("DB00001")]
    out = reciprocal_rank_fusion(graph, vector)
    ids = [r["drug_id"] for r in out]
    assert ids == ["DB00001", "DB00002"]

    by_id = {r["drug_id"]: r for r in out}
    expected_01 = 1 / (RRF_K + 1) + 1 / (RRF_K + 1)
    expected_02 = 1 / (RRF_K + 2)
    assert math.isclose(by_id["DB00001"]["rrf_score"], expected_01)
    assert math.isclose(by_id["DB00002"]["rrf_score"], expected_02)


def test_pairwise_direct_row_expands_to_both_endpoints() -> None:
    pair = _direct("DB00682", "Warfarin", "DB00945", "Aspirin")
    expanded = _expand_graph_rows([pair])
    assert len(expanded) == 2
    ids = [did for did, _name, _row in expanded]
    assert set(ids) == {"DB00682", "DB00945"}
    # Both endpoints reference the same row object.
    assert expanded[0][2] is pair
    assert expanded[1][2] is pair

    out = reciprocal_rank_fusion([pair], [])
    assert {r["drug_id"] for r in out} == {"DB00682", "DB00945"}
    for r in out:
        assert r["graph"] == [pair]
        assert r["vector"] == []


def test_enzyme_filter_duplicate_drug_sums_ranks_and_keeps_lineage() -> None:
    """Same drug appears in ENZYME_FILTER at ranks 0, 1, 2 (e.g. it
    inhibits three different enzymes). The fused row must carry all
    three rows in its graph lineage and an rrf_score summing the three
    rank contributions."""
    r0 = {**_enz("DB00001"), "enzyme": "CYP3A4"}
    r1 = {**_enz("DB00001"), "enzyme": "CYP2D6"}
    r2 = {**_enz("DB00001"), "enzyme": "CYP1A2"}
    out = reciprocal_rank_fusion([r0, r1, r2], [])
    assert len(out) == 1
    row = out[0]
    assert row["drug_id"] == "DB00001"
    assert len(row["graph"]) == 3
    assert [g["enzyme"] for g in row["graph"]] == ["CYP3A4", "CYP2D6", "CYP1A2"]
    expected = 1 / (RRF_K + 1) + 1 / (RRF_K + 2) + 1 / (RRF_K + 3)
    assert math.isclose(row["rrf_score"], expected)


def test_combined_graph_and_multi_vector_lineage_preserved() -> None:
    """DB00001 at graph rank 0 AND vector ranks 0 and 2. Lineage must
    carry 1 graph row + 2 vector rows; rrf_score sums all three."""
    graph = [_enz("DB00001")]
    vector = [
        _vec("DB00001", chunk_index=0),
        _vec("DB99999"),  # occupies vector rank 1
        _vec("DB00001", chunk_index=5),
    ]
    out = reciprocal_rank_fusion(graph, vector)
    by_id = {r["drug_id"]: r for r in out}
    row = by_id["DB00001"]
    assert len(row["graph"]) == 1
    assert len(row["vector"]) == 2
    chunk_indices = sorted(v["chunk_index"] for v in row["vector"])
    assert chunk_indices == [0, 5]
    expected = 1 / (RRF_K + 1) + 1 / (RRF_K + 1) + 1 / (RRF_K + 3)
    assert math.isclose(row["rrf_score"], expected)


def test_node_wrapper_mode_dispatch() -> None:
    graph = [_enz("DB00001"), _enz("DB00002")]
    vector = [_vec("DB00003")]

    # ddi_check → graph-only wrap
    out = fusion(
        {"query": "", "mode": "ddi_check", "graph_results": graph, "vector_results": vector}
    )
    assert {r["drug_id"] for r in out["fused_results"]} == {"DB00001", "DB00002"}
    assert all(r["vector"] == [] for r in out["fused_results"])

    # polypharmacy → graph-only wrap
    out = fusion(
        {"query": "", "mode": "polypharmacy", "graph_results": graph, "vector_results": vector}
    )
    assert {r["drug_id"] for r in out["fused_results"]} == {"DB00001", "DB00002"}

    # alternatives → vector-only wrap
    out = fusion(
        {"query": "", "mode": "alternatives", "graph_results": graph, "vector_results": vector}
    )
    assert {r["drug_id"] for r in out["fused_results"]} == {"DB00003"}
    assert all(r["graph"] == [] for r in out["fused_results"])

    # describe → vector-only wrap
    out = fusion(
        {"query": "", "mode": "describe", "graph_results": graph, "vector_results": vector}
    )
    assert {r["drug_id"] for r in out["fused_results"]} == {"DB00003"}

    # hybrid → true RRF merge
    out = fusion(
        {"query": "", "mode": "hybrid", "graph_results": graph, "vector_results": vector}
    )
    assert {r["drug_id"] for r in out["fused_results"]} == {
        "DB00001",
        "DB00002",
        "DB00003",
    }

    # unknown / unset mode → empty
    out = fusion(
        {"query": "", "graph_results": graph, "vector_results": vector}  # type: ignore[typeddict-item]
    )
    assert out["fused_results"] == []


def test_k_parameter_controls_magnitude_not_ordering() -> None:
    graph = [_enz("DB00001"), _enz("DB00002"), _enz("DB00003")]
    vector = [_vec("DB00002")]  # boost DB00002

    out_default = reciprocal_rank_fusion(graph, vector)
    out_small_k = reciprocal_rank_fusion(graph, vector, k=10)

    # Order must be identical across k values — RRF is monotone in k.
    assert [r["drug_id"] for r in out_default] == [
        r["drug_id"] for r in out_small_k
    ]
    # Smaller k → larger reciprocal → bigger fused scores.
    for r_d, r_s in zip(out_default, out_small_k, strict=True):
        assert r_s["rrf_score"] > r_d["rrf_score"]
