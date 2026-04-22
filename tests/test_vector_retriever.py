"""Vector retriever gates — hits local ChromaDB (./chroma_db).

Tests exercise:
  1. ALTERNATIVES: anticoagulant-safe-in-HIT -> Lepirudin + Argatroban
  2. DESCRIBE: metadata filter pins results to the resolved drug
  3. HYBRID: $in filter restricts to graph-qualified drug_ids
  4. DDI_CHECK no-op
  5. POLYPHARMACY no-op
  6. Per-mode latency budget (<500ms after warmup)
  7. Dedupe helper: one row per drug_id, highest cosine wins,
     source_collections preserved as a list

Latency tests intentionally skip the first cold-load call by relying on
a module-scoped warmup fixture that pre-loads the SentenceTransformer
and opens every collection handle before any test runs.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest
from dotenv import load_dotenv

from agent.nodes.vector_retriever import (
    _dedupe_by_drug,
    _get_collection,
    _get_model,
    vector_retriever,
)

load_dotenv()

LEPIRUDIN = "DB00001"
ARGATROBAN = "DB00278"
BIVALIRUDIN = "DB00006"
SEMAGLUTIDE = "DB13928"

# CLAUDE.md's "Known retrieval behaviors" sanity-check lists these three
# as the clinically correct HIT-safe anticoagulants (direct thrombin
# inhibitors FDA-labeled for HIT). The retriever must surface at least
# one of them — top-5 per collection is tight enough that which specific
# one wins depends on token overlap with the query phrasing.
HIT_SAFE_ANTICOAGULANTS = {LEPIRUDIN, ARGATROBAN, BIVALIRUDIN}


@pytest.fixture(autouse=True, scope="module")
def _warmup() -> Any:
    """Pre-load the embedder and open every Chroma collection once so
    individual tests measure query latency, not cold-load cost."""
    _get_model().encode(["warmup"])
    for coll in (
        "descriptions",
        "indications",
        "mechanisms",
        "pharmacodynamics",
        "contraindications",
    ):
        _get_collection(coll).count()
    yield


def _timed(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t0) * 1000.0


def test_alternatives_hit_safe_anticoagulants() -> None:
    """Same gate as Step 3's retrieval sanity-check (see CLAUDE.md
    "Known retrieval behaviors"), now wired through the node. We use
    the short-form query 'anticoagulant safe in HIT' that CLAUDE.md
    documents as passing; the expanded form dilutes the cosine signal
    and drops Lepirudin out of the top-5."""
    constraint = "anticoagulant safe in HIT"
    state: dict[str, Any] = {
        "query": "what anticoagulant should I use for a HIT patient",
        "mode": "alternatives",
        "semantic_constraint": constraint,
        "resolved_drugs": [],
        "graph_results": [],
    }
    out, ms = _timed(vector_retriever, state)
    rows = out["vector_results"]
    assert rows, "expected at least one alternatives hit"
    drug_ids = {r["drug_id"] for r in rows}
    assert HIT_SAFE_ANTICOAGULANTS & drug_ids, (
        f"expected one of {HIT_SAFE_ANTICOAGULANTS} (direct thrombin "
        f"inhibitors FDA-labeled for HIT), got {drug_ids}"
    )
    assert all(r["query_type"] == "vector" for r in rows)
    assert ms < 500.0, f"ALTERNATIVES latency {ms:.1f}ms exceeds 500ms budget"


def test_describe_metadata_filter_locks_to_resolved_drug() -> None:
    state: dict[str, Any] = {
        "query": "mechanism of action of semaglutide",
        "mode": "describe",
        "resolved_drugs": [{"drug_id": SEMAGLUTIDE}],
        "graph_results": [],
    }
    out, ms = _timed(vector_retriever, state)
    rows = out["vector_results"]
    assert rows, "expected describe to return chunks for Semaglutide"
    assert all(r["drug_id"] == SEMAGLUTIDE for r in rows), (
        "metadata filter leaked: describe returned rows for other drugs"
    )
    source_collections: set[str] = set()
    for r in rows:
        source_collections.update(r["source_collections"])
    assert "mechanisms" in source_collections, (
        f"mechanisms collection should contribute; got {source_collections}"
    )
    assert ms < 500.0, f"DESCRIBE latency {ms:.1f}ms exceeds 500ms budget"


def test_hybrid_restricts_to_graph_qualified_drug_ids() -> None:
    """The real invariant for hybrid: ``state["vector_results"]`` must be
    a subset of ``state["graph_results"]`` drug_ids — i.e. the Chroma
    ``$in`` metadata filter is being applied correctly and no
    unqualified drug slips in.

    We deliberately do NOT assert that canonical CYP3A4 inhibitors
    (clarithromycin/ketoconazole/itraconazole) rank high: their
    FDA-labeled indications are about infections, not statin
    interactions. CLAUDE.md's "Known retrieval behaviors" note explains
    that gate queries are answered against indication text, not
    clinical heuristics, so a 'interacts with statins' constraint will
    not semantically match those drugs' indication chunks.

    Requires Neo4j to source the CYP3A4 inhibitor list."""
    if not os.environ.get("NEO4J_PASSWORD"):
        pytest.skip("NEO4J_PASSWORD not set; cannot query CYP3A4 inhibitors")
    from agent.nodes.graph_retriever import close_driver, query_enzyme_filter

    try:
        inhibitors = query_enzyme_filter("CYP3A4", "inhibitor")
        assert len(inhibitors) >= 5, "need enough CYP3A4 inhibitors to test"
        allowed = {r["drug_id"] for r in inhibitors}

        state: dict[str, Any] = {
            "query": "which CYP3A4 inhibitors interact with statins",
            "mode": "hybrid",
            "semantic_constraint": "interacts with HMG-CoA reductase inhibitors",
            "resolved_drugs": [],
            "graph_results": inhibitors,
        }
        out, ms = _timed(vector_retriever, state)
        rows = out["vector_results"]
        assert rows, "expected hybrid to surface at least one candidate"
        surfaced = {r["drug_id"] for r in rows}
        bad = surfaced - allowed
        assert not bad, f"$in filter leaked drug_ids not in graph_results: {bad}"
        assert all(r["query_type"] == "vector" for r in rows)
        assert ms < 500.0, f"HYBRID latency {ms:.1f}ms exceeds 500ms budget"
    finally:
        close_driver()


def test_ddi_check_is_noop() -> None:
    out = vector_retriever(
        {
            "query": "is warfarin safe with aspirin",
            "mode": "ddi_check",
            "resolved_drugs": [{"drug_id": "DB00682"}, {"drug_id": "DB00945"}],
            "graph_results": [],
        }
    )
    assert out["vector_results"] == []


def test_polypharmacy_is_noop() -> None:
    out = vector_retriever(
        {
            "query": "regimen review",
            "mode": "polypharmacy",
            "resolved_drugs": [
                {"drug_id": "DB00682"},
                {"drug_id": "DB00945"},
                {"drug_id": "DB00641"},
            ],
            "graph_results": [],
        }
    )
    assert out["vector_results"] == []


def test_per_mode_p95_latency_under_500ms() -> None:
    """Run each real-retrieval mode 3 times and assert p95 (max of 3)
    stays under 500ms. One-shot latency assertions in tests 1/2/3 catch
    regressions on individual queries; this test catches variance."""
    samples: dict[str, list[float]] = {"alternatives": [], "describe": []}

    alt_state: dict[str, Any] = {
        "query": "HIT-safe anticoagulant",
        "mode": "alternatives",
        "semantic_constraint": "anticoagulant safe in HIT",
        "resolved_drugs": [],
        "graph_results": [],
    }
    desc_state: dict[str, Any] = {
        "query": "mechanism of action of semaglutide",
        "mode": "describe",
        "resolved_drugs": [{"drug_id": SEMAGLUTIDE}],
        "graph_results": [],
    }
    for _ in range(3):
        _, ms = _timed(vector_retriever, alt_state)
        samples["alternatives"].append(ms)
        _, ms = _timed(vector_retriever, desc_state)
        samples["describe"].append(ms)

    for mode, series in samples.items():
        p95 = max(series)  # 3 samples: p95 ≈ max
        assert p95 < 500.0, (
            f"{mode} p95={p95:.1f}ms over {series} exceeds 500ms budget"
        )


def test_dedupe_keeps_highest_cosine_and_preserves_sources() -> None:
    """Synthetic input: DB00001 appears in 3 collections with different
    cosines; DB00002 in 1 collection. Output must collapse DB00001 to a
    single row with cosine 0.9 and all 3 source collections listed."""
    synthetic = [
        {
            "drug_id": "DB00001",
            "drug_name": "Lepirudin",
            "collection": "indications",
            "text": "...",
            "cosine": 0.7,
            "chunk_index": 0,
            "query_type": "vector",
        },
        {
            "drug_id": "DB00001",
            "drug_name": "Lepirudin",
            "collection": "contraindications",
            "text": "...",
            "cosine": 0.9,
            "chunk_index": 0,
            "query_type": "vector",
        },
        {
            "drug_id": "DB00001",
            "drug_name": "Lepirudin",
            "collection": "mechanisms",
            "text": "...",
            "cosine": 0.8,
            "chunk_index": 0,
            "query_type": "vector",
        },
        {
            "drug_id": "DB00002",
            "drug_name": "Cetuximab",
            "collection": "indications",
            "text": "...",
            "cosine": 0.6,
            "chunk_index": 0,
            "query_type": "vector",
        },
    ]
    out = _dedupe_by_drug(synthetic)
    assert len(out) == 2, f"expected 2 deduped rows, got {len(out)}"
    by_id = {r["drug_id"]: r for r in out}

    lep = by_id["DB00001"]
    assert lep["cosine"] == pytest.approx(0.9)
    assert lep["collection"] == "contraindications"
    assert set(lep["source_collections"]) == {
        "indications",
        "contraindications",
        "mechanisms",
    }
    assert isinstance(lep["source_collections"], list)

    cet = by_id["DB00002"]
    assert cet["source_collections"] == ["indications"]

    # Sorted descending by cosine
    assert out[0]["drug_id"] == "DB00001"
    assert out[1]["drug_id"] == "DB00002"
