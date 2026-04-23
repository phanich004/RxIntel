"""LangGraph node: audit the reasoning agent's output against its evidence.

The critic is mode-agnostic — one prompt, one judgment procedure across
all five pipeline modes. It reads ``state["reasoning_output"]`` and
regenerates the same evidence block the reasoning agent saw by reusing
``agent.nodes.reasoning_agent._pack_evidence`` — this guarantees the
critic is grading against an apples-to-apples view, not a different
summarization of the same fused_results.

The LLM is asked to compute its own composite score, but this node
recomputes ``composite`` from the weighted sub-scores (0.50 / 0.35 /
0.15) and re-derives ``approved`` from the fixed 0.75 threshold. That
way the contract is enforced in code, not prompt discipline — an LLM
drift in score-math cannot silently bypass the threshold.

Retry / escalation is owned here, not by LangGraph wiring:

* approved → ``critic_feedback=None``
* rejected AND ``retry_count < MAX_RETRIES`` → ``critic_feedback`` carries
  the revision_prompt; reasoning_agent patches against it next turn.
* rejected AND ``retry_count >= MAX_RETRIES`` → ``escalated=True``,
  ``critic_feedback=None``; the Step 15 wiring routes to a human-review
  sink instead of looping.

``retry_count`` is incremented by the LangGraph wiring (Step 15), NOT
by this node — the critic reads the pre-increment value.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Final, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from agent import token_tracker
from agent.nodes.reasoning_agent import _extract_json, _pack_evidence
from agent.prompts.critic_prompts import CRITIC_SYSTEM_PROMPT
from agent.rate_limit import groq_retry
from agent.schemas import AgentState, CriticOutput

logger = logging.getLogger(__name__)

# Per CLAUDE.md §Critic Agent. Weights are load-bearing — changing them
# shifts which failures trip approval. Factual accuracy dominates because
# a well-formatted hallucination is worse than a rough-edged true answer.
FACTUAL_ACCURACY_WEIGHT: Final[float] = 0.50
SAFETY_WEIGHT: Final[float] = 0.35
COMPLETENESS_WEIGHT: Final[float] = 0.15
APPROVAL_THRESHOLD: Final[float] = 0.75

# Two revisions allowed before escalation (3 total reasoning attempts).
# Calibrated middle between the Step 14 spec's original 1 and CLAUDE.md's
# earlier 3. ``retry_count`` is pre-incremented when the critic reads it.
MAX_RETRIES: Final[int] = 2

DEFAULT_CRITIC_MODEL = "claude-sonnet-4-6"
CRITIC_MAX_TOKENS = 1024

_client: ChatAnthropic | None = None


def _get_client() -> ChatAnthropic:
    global _client
    if _client is None:
        _client = ChatAnthropic(  # type: ignore[call-arg]
            model=os.environ.get("CRITIC_MODEL", DEFAULT_CRITIC_MODEL),
            temperature=0,
            max_tokens=CRITIC_MAX_TOKENS,
            default_request_timeout=60.0,
        )
    return _client


def _compute_composite(
    factual_accuracy: float, safety_score: float, completeness_score: float
) -> float:
    return (
        FACTUAL_ACCURACY_WEIGHT * factual_accuracy
        + SAFETY_WEIGHT * safety_score
        + COMPLETENESS_WEIGHT * completeness_score
    )


def _format_reasoning_for_review(output: dict[str, Any]) -> str:
    """Pretty-print the reasoning JSON with field headers.

    A raw ``json.dumps`` works but scannable headers help the critic
    focus on the per-field grounding check rather than parsing braces.
    """
    mode = output.get("mode", "<unknown>")
    lines = [f"mode: {mode}"]
    for key in (
        "severity",
        "confidence",
        "insufficient_evidence",
        "summary",
        "mechanism",
        "recommendation",
        "interacting_pairs",
        "candidates",
        "sources",
    ):
        if key not in output:
            continue
        val = output[key]
        if isinstance(val, (list, dict)):
            rendered = json.dumps(val, ensure_ascii=False, indent=2)
        else:
            rendered = str(val)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines)


def _format_user_message(state: AgentState) -> str:
    """Build the user turn: evidence block + output-under-review."""
    evidence = _pack_evidence(state)
    reasoning_output = state.get("reasoning_output") or {}
    review_card = _format_reasoning_for_review(reasoning_output)
    mode = state.get("mode") or "<unknown>"
    return (
        "EVIDENCE THE REASONING AGENT WAS GIVEN:\n"
        f"{evidence}\n\n"
        f"REASONING OUTPUT UNDER REVIEW (mode={mode}):\n"
        f"{review_card}\n\n"
        "Score the reasoning output on factual_accuracy, safety, and "
        "completeness. Emit the JSON object per the schema in your "
        "system prompt."
    )


@groq_retry
def _invoke(system: str, user: str) -> str:
    client = _get_client()
    response = client.invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    token_tracker.record("critic", response)
    content = response.content
    if isinstance(content, list):
        content = "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
        )
    return _extract_json(str(content))


def judge(state: AgentState) -> CriticOutput:
    """Pure helper: AgentState -> validated CriticOutput.

    Recomputes ``composite`` and ``approved`` from the sub-scores so the
    contract holds even if the LLM drifts on score-math.
    """
    reasoning_output = state.get("reasoning_output")
    if not reasoning_output:
        # Defensive: no reasoning output means the graph is misordered;
        # surface this as an auto-reject rather than crashing.
        logger.warning(
            "critic_agent invoked with empty reasoning_output; auto-rejecting"
        )
        return CriticOutput(
            approved=False,
            factual_accuracy=0.0,
            safety_score=0.0,
            completeness_score=0.0,
            composite=0.0,
            issues=["No reasoning_output present in state to review."],
            revision_prompt=(
                "No prior reasoning output was supplied to the critic. "
                "Re-run the reasoning agent."
            ),
        )

    raw = _invoke(CRITIC_SYSTEM_PROMPT, _format_user_message(state))
    parsed = json.loads(raw)

    fa = float(parsed.get("factual_accuracy", 0.0))
    ss = float(parsed.get("safety_score", 0.0))
    cs = float(parsed.get("completeness_score", 0.0))
    composite = _compute_composite(fa, ss, cs)
    approved = composite >= APPROVAL_THRESHOLD

    return CriticOutput(
        approved=approved,
        factual_accuracy=fa,
        safety_score=ss,
        completeness_score=cs,
        composite=composite,
        issues=list(parsed.get("issues", []) or []),
        revision_prompt=str(parsed.get("revision_prompt", "") or ""),
    )


def critic_agent(state: AgentState) -> AgentState:
    """LangGraph node: judge + set retry/escalation signals.

    Writes (partial state update):
      - critic_score: CriticOutput as dict
      - critic_feedback: revision_prompt string if a retry is due,
        else None
      - escalated: True if we are past MAX_RETRIES and still rejected
    """
    out = judge(state)
    update: dict[str, Any] = {"critic_score": out.model_dump()}
    retry_count = state.get("retry_count", 0) or 0

    if out.approved:
        update["critic_feedback"] = None
    elif retry_count < MAX_RETRIES:
        update["critic_feedback"] = out.revision_prompt
    else:
        update["critic_feedback"] = None
        update["escalated"] = True

    return cast(AgentState, update)
