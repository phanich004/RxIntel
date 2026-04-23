"""Gate: 10 canned queries, 2 per mode, must classify correctly @ T=0.

These tests hit the live Groq API. All are skipped by default so a plain
``pytest`` run does not burn quota. Remove the module-level skip (or
``--no-skip`` via a CLI flag if you wire one up) once you intend to pay
the 10 LLM calls.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from agent.nodes.entity_resolver import entity_resolver
from agent.nodes.query_router import classify

load_dotenv()

ROUTER_GATE: list[tuple[str, str]] = [
    ("Is warfarin safe with aspirin?", "ddi_check"),
    ("Can I take metformin and ibuprofen together?", "ddi_check"),
    ("What's an alternative to warfarin for HIT?", "alternatives"),
    (
        "I need a replacement for ibuprofen, easier on the kidneys",
        "alternatives",
    ),
    ("Which CYP2D6 inhibitors affect antidepressants?", "hybrid"),
    ("Which drugs inhibit CYP3A4 and interact with statins?", "hybrid"),
    ("What is semaglutide?", "describe"),
    ("How does atorvastatin work?", "describe"),
    (
        "Patient takes warfarin, metformin, omeprazole, clopidogrel — any concerns?",
        "polypharmacy",
    ),
    (
        "My patient is on 5 medications and I'm worried about interactions",
        "polypharmacy",
    ),
]


@pytest.fixture(autouse=True, scope="module")
def _require_groq_key() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")


@pytest.mark.skip(reason="live Groq call — run manually")
@pytest.mark.parametrize(("query", "expected_mode"), ROUTER_GATE)
def test_router_classification(query: str, expected_mode: str) -> None:
    resolved = entity_resolver({"query": query}).get("resolved_drugs", []) or []
    result = classify(query, resolved)
    assert result.mode == expected_mode, (
        f"query={query!r} expected={expected_mode} got={result.mode} "
        f"(confidence={result.confidence}, filter={result.graph_filter}, "
        f"constraint={result.semantic_constraint})"
    )
