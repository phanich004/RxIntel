"""Gate assertions for the three-tier entity resolver.

These tests hit the real gazetteer built in Step 4 and (for tier 3) the
real ChromaDB store built in Step 3. They are skipped if either artifact
is missing, so cloning + running pytest on a fresh checkout is harmless.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.nodes.entity_resolver import (
    GAZETTEER_EXACT_PATH,
    GAZETTEER_FUZZY_PATH,
    _candidate_spans,
    entity_resolver,
)


@pytest.fixture(autouse=True, scope="module")
def _require_gazetteer() -> None:
    for p in (GAZETTEER_EXACT_PATH, GAZETTEER_FUZZY_PATH):
        if not Path(p).exists():
            pytest.skip(f"{p} not found — run build_gazetteer.py first")


def _ids(state_update: dict[str, object]) -> set[str]:
    resolved = state_update["resolved_drugs"]
    assert isinstance(resolved, list)
    return {r["drug_id"] for r in resolved}


def test_tier1_warfarin_aspirin() -> None:
    out = entity_resolver({"query": "Is warfarin + aspirin safe?"})
    assert {"DB00682", "DB00945"}.issubset(_ids(out))
    for r in out["resolved_drugs"]:
        assert r["tier"] == 1
        assert r["match_score"] >= 85.0


def test_tier1_coumadin_synonym() -> None:
    out = entity_resolver({"query": "Coumadin with ibuprofen"})
    ids = _ids(out)
    assert "DB00682" in ids  # coumadin → warfarin
    assert "DB01050" in ids  # ibuprofen


def test_tier1_or_2_novoseven_brand() -> None:
    out = entity_resolver({"query": "novoseven interaction"})
    resolved = out["resolved_drugs"]
    assert len(resolved) >= 1
    novo = resolved[0]
    assert novo["tier"] in (1, 2)
    assert novo["drug_id"] == "DB00036"


def test_candidate_spans_skip_glue() -> None:
    spans = _candidate_spans("Is warfarin + aspirin safe?")
    assert "warfarin" in spans
    assert "aspirin" in spans
    assert "is" not in spans
    assert "safe" not in spans


def test_empty_query_returns_empty() -> None:
    out = entity_resolver({"query": ""})
    assert out["resolved_drugs"] == []
