"""Schema validation, defaults, and mode-alias coverage."""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from agent.schemas import (
    AgentState,
    CriticOutput,
    Mode,
    ReasoningOutput,
    RouterOutput,
)

ALL_MODES: tuple[str, ...] = (
    "ddi_check",
    "alternatives",
    "hybrid",
    "describe",
    "polypharmacy",
)


def test_mode_alias_exhaustive() -> None:
    """Single source of truth: Mode covers exactly the five pipeline modes
    and each one is assignable to AgentState['mode']."""
    assert set(get_args(Mode)) == set(ALL_MODES)

    for m in ALL_MODES:
        state: AgentState = {"query": "x", "mode": m}  # type: ignore[typeddict-item]
        assert state["mode"] == m


def test_router_output_validates_confidence_and_defaults() -> None:
    ok = RouterOutput(mode="ddi_check", confidence=0.9)
    assert ok.graph_filter is None
    assert ok.semantic_constraint is None

    with pytest.raises(ValidationError):
        RouterOutput(mode="ddi_check", confidence=1.5)

    with pytest.raises(ValidationError):
        RouterOutput(mode="not_a_mode", confidence=0.5)  # type: ignore[arg-type]


def test_reasoning_output_requires_mechanism_and_sources_defaults_severity() -> None:
    r = ReasoningOutput(
        mode="describe",
        mechanism="competitive inhibition of vitamin K epoxide reductase",
        recommendation="avoid combination",
        confidence=0.8,
        sources=["DB00682"],
    )
    assert r.severity == "n/a"
    assert r.insufficient_evidence is False
    assert r.interacting_pairs == []
    assert r.candidates == []
    assert r.summary == ""

    insufficient = ReasoningOutput(
        mode="ddi_check",
        severity="unknown",
        mechanism="",
        recommendation="escalate to human review",
        confidence=0.0,
        insufficient_evidence=True,
        sources=[],
    )
    assert insufficient.severity == "unknown"
    assert insufficient.insufficient_evidence is True

    with pytest.raises(ValidationError):
        ReasoningOutput(  # type: ignore[call-arg]
            mode="ddi_check",
            recommendation="x",
            confidence=0.5,
            sources=[],
        )


def test_critic_output_validates_ranges_and_defaults() -> None:
    c = CriticOutput(
        approved=True,
        factual_accuracy=0.9,
        safety_score=0.85,
        completeness_score=0.7,
        composite=0.85,
    )
    assert c.issues == []
    assert c.revision_prompt == ""

    with pytest.raises(ValidationError):
        CriticOutput(
            approved=False,
            factual_accuracy=0.9,
            safety_score=0.85,
            completeness_score=0.7,
            composite=1.01,
        )
