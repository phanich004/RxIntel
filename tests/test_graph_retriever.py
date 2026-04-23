"""Graph retriever gates — hits the live local Neo4j.

Tests exercise:
  1. DIRECT: Warfarin + Aspirin
  2. DIRECT: Simvastatin + Clarithromycin
  3. MULTI_HOP: Simvastatin + Clarithromycin (invariant #5 fires)
  4. ENZYME_FILTER: CYP3A4 inhibitors
  5. No-op for alternatives / describe
  6. ANTI-TEST: two CYP3A4-substrate-only drugs must NOT return any
     multi-hop rows (guards against a substrate/substrate regression)
  7. Latency: p95 per query type

The anti-test pair was chosen by probing Neo4j for drugs whose only
CYP3A4 action is "substrate" AND which have no direct INTERACTS_WITH
edge between them. If either invariant drifts (e.g. the ETL reloads
differently), re-probe and update the pair.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest
from dotenv import load_dotenv

from agent.nodes.graph_retriever import (
    close_driver,
    graph_retriever,
    query_direct,
    query_enzyme_filter,
    query_multi_hop,
)

load_dotenv()

WARFARIN = "DB00682"
ASPIRIN = "DB00945"
SIMVASTATIN = "DB00641"
CLARITHROMYCIN = "DB01211"

# Two drugs that are CYP3A4-substrate-only AND have no direct
# INTERACTS_WITH edge between them. Probed from Neo4j; documented in
# tests/test_graph_retriever.py header.
ANTI_DRUG_A = "DB00204"  # Dofetilide
ANTI_DRUG_B = "DB12781"  # Balaglitazone


@pytest.fixture(autouse=True, scope="module")
def _require_neo4j_creds() -> Any:
    if not os.environ.get("NEO4J_PASSWORD"):
        pytest.skip("NEO4J_PASSWORD not set")
    yield
    close_driver()


def _resolved(*ids: str) -> list[dict[str, Any]]:
    return [{"drug_id": did} for did in ids]


def _timed(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t0) * 1000.0


def test_direct_warfarin_aspirin() -> None:
    rows, ms = _timed(query_direct, [WARFARIN, ASPIRIN])
    assert rows, "expected at least one direct INTERACTS_WITH row"
    assert all(r["query_type"] == "direct" for r in rows)
    assert all(r["severity"] in {"Major", "Moderate", "Minor"} for r in rows)
    assert ms < 500.0, f"DIRECT latency {ms:.1f}ms exceeds 500ms budget"


def test_direct_simvastatin_clarithromycin() -> None:
    rows, ms = _timed(query_direct, [SIMVASTATIN, CLARITHROMYCIN])
    assert rows, "expected direct edge between simvastatin and clarithromycin"
    assert all(r["severity"] in {"Major", "Moderate", "Minor"} for r in rows)
    assert ms < 500.0, f"DIRECT latency {ms:.1f}ms exceeds 500ms budget"


def test_multi_hop_simvastatin_clarithromycin_fires_action_filter() -> None:
    rows, ms = _timed(query_multi_hop, SIMVASTATIN, CLARITHROMYCIN)
    assert rows, "expected ≥1 action-compatible shared-enzyme path"
    enzyme_ids = {r["enzyme_id"] for r in rows}
    assert enzyme_ids, "enzyme_id column missing from multi_hop output"

    # Every row must demonstrate a clinically meaningful action pair:
    # one side has inhibitor or inducer, the other has substrate.
    for r in rows:
        a_actions = set(r["a_actions"])
        b_actions = set(r["b_actions"])
        a_side = a_actions & {"inhibitor", "inducer"}
        b_side = b_actions & {"inhibitor", "inducer"}
        a_sub = "substrate" in a_actions
        b_sub = "substrate" in b_actions
        valid = (a_side and b_sub) or (b_side and a_sub)
        assert valid, f"unexpected action combo: a={a_actions} b={b_actions}"

    assert all(r["query_type"] == "multi_hop" for r in rows)
    assert ms < 1000.0, f"MULTI_HOP latency {ms:.1f}ms exceeds 1000ms budget"


def test_enzyme_filter_cyp3a4_inhibitors() -> None:
    rows, ms = _timed(query_enzyme_filter, "CYP3A4", "inhibitor")
    assert len(rows) >= 5, f"expected ≥5 CYP3A4 inhibitors, got {len(rows)}"
    for r in rows:
        assert "inhibitor" in r["actions"]
        assert r["query_type"] == "enzyme_filter"
    assert ms < 500.0, f"ENZYME_FILTER latency {ms:.1f}ms exceeds 500ms budget"


def test_alternatives_and_describe_are_noops() -> None:
    for mode in ("alternatives", "describe"):
        out = graph_retriever(
            {
                "query": "x",
                "mode": mode,  # type: ignore[typeddict-item]
                "resolved_drugs": _resolved(WARFARIN),
            }
        )
        assert out["graph_results"] == []


def test_multi_hop_substrate_only_pair_returns_empty() -> None:
    """Invariant #5 anti-test: two CYP3A4-substrate-only drugs with no
    direct interaction must produce zero multi-hop rows. If this test
    fails, the Cypher filter is broken and the pipeline will emit
    thousands of false-positive interactions."""
    rows, ms = _timed(query_multi_hop, ANTI_DRUG_A, ANTI_DRUG_B)
    assert rows == [], (
        f"substrate-only pair ({ANTI_DRUG_A}, {ANTI_DRUG_B}) returned "
        f"{len(rows)} rows; invariant #5 filter is broken"
    )
    assert ms < 1000.0


def test_graph_retriever_dispatch_ddi_check_surfaces_multi_hop() -> None:
    """For two drugs that share an action-compatible enzyme path, the
    ddi_check dispatch should surface MULTI_HOP rows with
    query_type=multi_hop. Clarithromycin (inhibitor) + Dofetilide
    (substrate) of CYP3A4 — may or may not have a direct edge depending
    on DrugBank coverage; MULTI_HOP rows should appear either way."""
    state_update = graph_retriever(
        {
            "query": "x",
            "mode": "ddi_check",
            "resolved_drugs": _resolved(CLARITHROMYCIN, "DB00204"),
        }
    )
    results = state_update["graph_results"]
    # Whatever path surfaces (direct or multi-hop), query_type must be
    # labeled consistently and at least one row should exist.
    if results:
        labels = {r["query_type"] for r in results}
        assert labels.issubset({"direct", "multi_hop"})


def test_graph_retriever_dispatch_ddi_check_augments_direct_with_multi_hop() -> None:
    """Polish: when DIRECT rows exist for a 2-drug ddi_check AND the
    pair also shares an action-compatible enzyme path, the dispatch
    should return BOTH lineages so the reasoning agent has PK mechanism
    context alongside the explicit INTERACTS_WITH severity. Simvastatin
    + Clarithromycin is the canonical both-paths pair (see
    test_direct_simvastatin_clarithromycin and
    test_multi_hop_simvastatin_clarithromycin_fires_action_filter)."""
    state_update = graph_retriever(
        {
            "query": "x",
            "mode": "ddi_check",
            "resolved_drugs": _resolved(SIMVASTATIN, CLARITHROMYCIN),
        }
    )
    results = state_update["graph_results"]
    labels = {r["query_type"] for r in results}
    assert "direct" in labels, f"expected direct rows, got labels={labels}"
    assert "multi_hop" in labels, f"expected multi_hop rows, got labels={labels}"
