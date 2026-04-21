"""Pydantic models and the LangGraph AgentState TypedDict.

The single ``Mode`` alias below is the source of truth for the five
pipeline modes; every routing / reasoning / state field imports it so a
new mode only has to be added in one place.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

Mode = Literal["ddi_check", "alternatives", "hybrid", "describe", "polypharmacy"]


class _ExcludeNoneModel(BaseModel):
    """Base for pipeline output models: serialization drops None fields.

    ``ConfigDict(exclude_none=True)`` is NOT a real Pydantic V2 config
    key — it's silently ignored — so we override the serializers instead.
    Callers can still pass ``exclude_none=False`` explicitly if they need
    the null-bearing form.
    """

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(**kwargs)


class AgentState(TypedDict, total=False):
    """LangGraph channel state threaded through every node.

    ``total=False`` so early nodes can populate a subset without tripping
    the type checker; invariants about required-at-read-time live in the
    node code, not the schema.
    """

    query: str
    resolved_drugs: list[dict[str, Any]]
    mode: Mode | None
    graph_filter: dict[str, Any] | None
    semantic_constraint: str | None
    graph_results: list[dict[str, Any]]
    vector_results: list[dict[str, Any]]
    fused_results: list[dict[str, Any]]
    reasoning_output: dict[str, Any] | None
    critic_feedback: str | None
    critic_score: dict[str, Any] | None
    retry_count: int
    final_output: dict[str, Any] | None
    escalated: bool


class RouterOutput(_ExcludeNoneModel):
    """Structured output emitted by the query_router node."""

    mode: Mode
    confidence: float = Field(ge=0.0, le=1.0)
    graph_filter: dict[str, Any] | None = None
    semantic_constraint: str | None = None


class ReasoningOutput(_ExcludeNoneModel):
    """Structured output emitted by the reasoning_agent node.

    ``severity="unknown"`` plus ``insufficient_evidence=True`` is the
    sanctioned way to refuse a confident answer when retrieval returns
    nothing (invariant #7). Do not collapse this into a generic error.
    """

    mode: Mode
    severity: Literal["Major", "Moderate", "Minor", "unknown", "n/a"] = "n/a"
    interacting_pairs: list[dict[str, Any]] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""
    mechanism: str
    recommendation: str
    confidence: float = Field(ge=0.0, le=1.0)
    insufficient_evidence: bool = False
    sources: list[str]


class CriticOutput(_ExcludeNoneModel):
    """Structured output emitted by the critic_agent node.

    Composite uses the CLAUDE.md weights (0.50 / 0.35 / 0.15); the model
    does not recompute composite from the three sub-scores, the critic
    does. We only validate ranges here.
    """

    approved: bool
    factual_accuracy: float = Field(ge=0.0, le=1.0)
    safety_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    composite: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    revision_prompt: str = ""
