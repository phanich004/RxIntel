"""Critic agent tests — 8 offline + 3 live Groq.

Same two-tier structure as ``test_reasoning_agent.py``: offline tests
mock ``_invoke`` so there's no API call, and every live test carries
``@pytest.mark.skip`` by default so a plain ``pytest`` run does not
burn Groq quota.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest
from dotenv import load_dotenv

from agent.nodes.critic_agent import (
    APPROVAL_THRESHOLD,
    MAX_RETRIES,
    _compute_composite,
    _format_user_message,
    critic_agent,
    judge,
)
from agent.schemas import CriticOutput

load_dotenv()


# ---------------------------------------------------------------- #
# Canned reasoning outputs and states used across tests
# ---------------------------------------------------------------- #
def _ddi_state_with_output(
    output: dict[str, Any], retry_count: int = 0
) -> dict[str, Any]:
    return {
        "query": "Is warfarin safe with aspirin?",
        "mode": "ddi_check",
        "fused_results": [
            {
                "drug_id": "DB00682",
                "graph": [
                    {
                        "drug_a_id": "DB00682",
                        "drug_a_name": "Warfarin",
                        "drug_b_id": "DB00945",
                        "drug_b_name": "Aspirin",
                        "query_type": "direct",
                        "severity": "Major",
                        "description": (
                            "Aspirin may increase the anticoagulant "
                            "activity of Warfarin, increasing bleeding risk."
                        ),
                    }
                ],
                "vector": [],
            }
        ],
        "reasoning_output": output,
        "retry_count": retry_count,
    }


_STRONG_DDI_OUTPUT: dict[str, Any] = {
    "mode": "ddi_check",
    "severity": "Major",
    "interacting_pairs": [
        {
            "drug_a": "Warfarin (DB00682)",
            "drug_b": "Aspirin (DB00945)",
            "severity": "Major",
            "rationale": (
                "Aspirin may increase the anticoagulant activity of "
                "Warfarin, increasing bleeding risk."
            ),
        }
    ],
    "mechanism": (
        "The interaction between Warfarin and Aspirin occurs because "
        "Aspirin may increase the anticoagulant activity of Warfarin, "
        "which can lead to an increased bleeding risk. This is a "
        "pharmacodynamic interaction where the combined effect of both "
        "drugs enhances the anticoagulant effect of Warfarin."
    ),
    "recommendation": (
        "Monitor INR weekly when co-administering Warfarin and Aspirin."
    ),
    "confidence": 0.9,
    "insufficient_evidence": False,
    "sources": ["DB00682", "DB00945"],
}


_WEAK_DDI_OUTPUT: dict[str, Any] = {
    "mode": "ddi_check",
    "severity": "Major",
    "interacting_pairs": [
        {
            "drug_a": "Warfarin (DB00682)",
            "drug_b": "Aspirin (DB00945)",
            "severity": "Major",
            "rationale": "dangerous combination",
        }
    ],
    "mechanism": "",
    "recommendation": "avoid",
    "confidence": 1.0,
    "insufficient_evidence": False,
    "sources": ["DB00682", "DB00945"],
}


_INSUFFICIENT_OUTPUT: dict[str, Any] = {
    "mode": "ddi_check",
    "severity": "unknown",
    "interacting_pairs": [],
    "mechanism": (
        "No direct or enzyme-pathway evidence for this drug pair "
        "was retrieved."
    ),
    "recommendation": (
        "Escalate to human clinical review — automated evidence is "
        "insufficient."
    ),
    "confidence": 0.0,
    "insufficient_evidence": True,
    "sources": [],
}


# ---------------------------------------------------------------- #
# 1. Composite math
# ---------------------------------------------------------------- #
def test_compute_composite_weights() -> None:
    assert _compute_composite(0.9, 0.9, 0.5) == pytest.approx(0.84, abs=1e-6)
    assert _compute_composite(1.0, 1.0, 1.0) == pytest.approx(1.0, abs=1e-6)
    assert _compute_composite(0.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-6)
    # Approval threshold lives exactly at composite == 0.75.
    assert APPROVAL_THRESHOLD == 0.75


# ---------------------------------------------------------------- #
# 2. judge() parses mocked critic JSON and re-derives composite
# ---------------------------------------------------------------- #
def test_judge_parses_mocked_critic_json() -> None:
    canned = (
        '{"approved": false, "factual_accuracy": 0.9, '
        '"safety_score": 0.9, "completeness_score": 0.5, '
        '"composite": 0.1, '  # deliberately wrong — node must recompute
        '"issues": ["x"], "revision_prompt": "fix x"}'
    )
    state = _ddi_state_with_output(_STRONG_DDI_OUTPUT)
    with patch("agent.nodes.critic_agent._invoke", return_value=canned):
        out = judge(state)
    assert isinstance(out, CriticOutput)
    assert out.factual_accuracy == pytest.approx(0.9)
    assert out.safety_score == pytest.approx(0.9)
    assert out.completeness_score == pytest.approx(0.5)
    assert out.composite == pytest.approx(0.84, abs=1e-6)
    # Node overrides LLM-reported approved from recomputed composite.
    assert out.approved is True
    assert out.issues == ["x"]
    assert out.revision_prompt == "fix x"


# ---------------------------------------------------------------- #
# 3. critic_agent approves when composite >= threshold
# ---------------------------------------------------------------- #
def test_critic_agent_approves_when_composite_above_threshold() -> None:
    canned = (
        '{"approved": true, "factual_accuracy": 0.95, '
        '"safety_score": 0.9, "completeness_score": 0.9, '
        '"composite": 0.9275, "issues": [], "revision_prompt": ""}'
    )
    state = _ddi_state_with_output(_STRONG_DDI_OUTPUT, retry_count=0)
    with patch("agent.nodes.critic_agent._invoke", return_value=canned):
        update = critic_agent(state)
    assert update["critic_score"]["approved"] is True  # type: ignore[index]
    assert update["critic_feedback"] is None
    assert "escalated" not in update


# ---------------------------------------------------------------- #
# 4. critic_agent rejects on first retry and sets feedback
# ---------------------------------------------------------------- #
def test_critic_agent_rejects_first_time_sets_feedback() -> None:
    canned = (
        '{"approved": false, "factual_accuracy": 0.5, '
        '"safety_score": 0.6, "completeness_score": 0.4, '
        '"composite": 0.52, "issues": ["mechanism empty"], '
        '"revision_prompt": "Populate the mechanism field with the '
        'pharmacodynamic basis of the Warfarin+Aspirin interaction."}'
    )
    state = _ddi_state_with_output(_WEAK_DDI_OUTPUT, retry_count=0)
    with patch("agent.nodes.critic_agent._invoke", return_value=canned):
        update = critic_agent(state)
    assert update["critic_score"]["approved"] is False  # type: ignore[index]
    assert isinstance(update["critic_feedback"], str)
    assert "mechanism" in update["critic_feedback"].lower()
    assert "escalated" not in update


# ---------------------------------------------------------------- #
# 5. critic_agent escalates when MAX_RETRIES exhausted
# ---------------------------------------------------------------- #
def test_critic_agent_escalates_on_final_rejection() -> None:
    """With MAX_RETRIES=2, retry_count=2 means two revisions have already
    happened; next rejection must escalate, not trigger a third retry."""
    assert MAX_RETRIES == 2
    canned = (
        '{"approved": false, "factual_accuracy": 0.5, '
        '"safety_score": 0.6, "completeness_score": 0.4, '
        '"composite": 0.52, "issues": ["still weak"], '
        '"revision_prompt": "fix still-empty mechanism"}'
    )
    state = _ddi_state_with_output(_WEAK_DDI_OUTPUT, retry_count=MAX_RETRIES)
    with patch("agent.nodes.critic_agent._invoke", return_value=canned):
        update = critic_agent(state)
    assert update["critic_score"]["approved"] is False  # type: ignore[index]
    assert update["critic_feedback"] is None
    assert update["escalated"] is True


# ---------------------------------------------------------------- #
# 6. Insufficient-evidence outputs must not be penalized
# ---------------------------------------------------------------- #
def test_critic_does_not_block_insufficient_evidence_output() -> None:
    """When the reasoning agent correctly emits insufficient_evidence=True
    with canonical escape values, the critic prompt instructs a full-score
    pass. We mock that behavior and verify the node approves cleanly."""
    canned = (
        '{"approved": true, "factual_accuracy": 1.0, '
        '"safety_score": 1.0, "completeness_score": 1.0, '
        '"composite": 1.0, "issues": [], "revision_prompt": ""}'
    )
    state = {
        "query": "obscure drug X",
        "mode": "ddi_check",
        "fused_results": [],
        "reasoning_output": _INSUFFICIENT_OUTPUT,
        "retry_count": 0,
    }
    with patch("agent.nodes.critic_agent._invoke", return_value=canned):
        update = critic_agent(state)
    assert update["critic_score"]["approved"] is True  # type: ignore[index]
    assert update["critic_score"]["completeness_score"] == 1.0  # type: ignore[index]
    assert update["critic_feedback"] is None


# ---------------------------------------------------------------- #
# 7. _format_user_message contains evidence AND reasoning output
# ---------------------------------------------------------------- #
def test_format_user_message_contains_both_evidence_and_output() -> None:
    state = _ddi_state_with_output(_STRONG_DDI_OUTPUT)
    msg = _format_user_message(state)
    assert "EVIDENCE THE REASONING AGENT WAS GIVEN:" in msg
    # Evidence block appears verbatim — Warfarin+Aspirin pair row
    assert "Warfarin" in msg
    assert "Aspirin" in msg
    # Reasoning output appears in the review card
    assert "REASONING OUTPUT UNDER REVIEW" in msg
    assert "mode=ddi_check" in msg
    assert "severity" in msg
    assert "mechanism" in msg


# ---------------------------------------------------------------- #
# 8. critic_agent only writes its own keys (doesn't clobber state)
# ---------------------------------------------------------------- #
def test_critic_agent_update_keys_are_scoped() -> None:
    canned = (
        '{"approved": true, "factual_accuracy": 0.95, '
        '"safety_score": 0.9, "completeness_score": 0.9, '
        '"composite": 0.9275, "issues": [], "revision_prompt": ""}'
    )
    state = _ddi_state_with_output(_STRONG_DDI_OUTPUT)
    with patch("agent.nodes.critic_agent._invoke", return_value=canned):
        update = critic_agent(state)
    # Only critic-owned keys in the partial update
    assert set(update.keys()) <= {
        "critic_score",
        "critic_feedback",
        "escalated",
    }
    # Original state keys untouched in the update dict
    assert "fused_results" not in update
    assert "reasoning_output" not in update


# ---------------------------------------------------------------- #
# Live Groq gate + live tests (SKIPPED BY DEFAULT)
# ---------------------------------------------------------------- #
@pytest.fixture
def _require_groq_key() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")


@pytest.mark.skip(reason="live Groq call — run manually")
def test_live_critic_approves_strong_ddi_output(
    _require_groq_key: None,
) -> None:
    """The Step 13 live ddi_check output fed back through the critic —
    expect approval with composite >= 0.80 and no hard issues."""
    state = _ddi_state_with_output(_STRONG_DDI_OUTPUT)
    out = judge(state)
    import json as _json
    print("\n----- CriticOutput JSON -----")
    print(_json.dumps(out.model_dump(), indent=2, ensure_ascii=False))
    print("-----------------------------")
    assert out.approved is True
    assert out.composite >= 0.80
    assert out.factual_accuracy >= 0.8
    assert out.safety_score >= 0.8


@pytest.mark.skip(reason="live Groq call — run manually")
def test_live_critic_rejects_fabricated_weak_output(
    _require_groq_key: None,
) -> None:
    """Severity=Major claimed, confidence=1.0, mechanism="", fused_results=[]
    — nothing in the evidence supports the Major claim; critic should
    catch both the ungrounded severity and the empty mechanism."""
    state: dict[str, Any] = {
        "query": "Is warfarin safe with aspirin?",
        "mode": "ddi_check",
        "fused_results": [],
        "reasoning_output": _WEAK_DDI_OUTPUT,
        "retry_count": 0,
    }
    out = judge(state)
    import json as _json
    print("\n----- CriticOutput JSON -----")
    print(_json.dumps(out.model_dump(), indent=2, ensure_ascii=False))
    print("-----------------------------")
    assert out.approved is False
    assert out.composite < APPROVAL_THRESHOLD  # rejected (specific bound not asserted — see note)
    assert out.factual_accuracy <= 0.5  # ungrounded claims should score low
    assert out.safety_score <= 0.8      # unsupported recommendation isn't fully safe
    # Composite may score as low as 0.0 for fully ungrounded adversarial outputs.
    # We don't assert a lower bound — the critic appropriately distinguishes
    # "fully fabricated" (→ 0.0) from "partially grounded but weak" (→ 0.3-0.5).
    # Both are rejection-worthy; both are correct critic behavior.
    lowered = " ".join(out.issues).lower()
    assert "severity" in lowered or "evidence" in lowered
    assert "mechanism" in lowered or "empty" in lowered


@pytest.mark.skip(reason="live Groq call — run manually")
def test_live_critic_approves_insufficient_evidence_output(
    _require_groq_key: None,
) -> None:
    """Canonical insufficient-evidence output with empty fused_results —
    critic must treat this as a WIN per Calibration 1."""
    state: dict[str, Any] = {
        "query": "obscure drug pair",
        "mode": "ddi_check",
        "fused_results": [],
        "reasoning_output": _INSUFFICIENT_OUTPUT,
        "retry_count": 0,
    }
    out = judge(state)
    import json as _json
    print("\n----- CriticOutput JSON -----")
    print(_json.dumps(out.model_dump(), indent=2, ensure_ascii=False))
    print("-----------------------------")
    assert out.approved is True
    assert out.completeness_score >= 0.8
    assert out.factual_accuracy >= 0.8
