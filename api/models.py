"""Pydantic request / response models for the RxIntel FastAPI service.

The response envelope ``AskResponse`` uses ``extra="allow"`` because
``reasoning_output`` has five mode-specific shapes and pinning them to
a discriminated union would be a maintenance burden that doesn't pay
off until we're serving external clients. The UI layer treats unknown
fields as pass-through dicts; the typed fields below are the ones the
clinician screen reads directly.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    """Clinician query envelope.

    ``thread_id`` is an opaque string the caller can pass to correlate
    multi-turn conversations in logs/traces; it's not used by the graph
    today beyond being echoed into the MLflow params.
    """

    query: str = Field(..., min_length=1, max_length=2000)
    thread_id: str | None = Field(default=None, max_length=128)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


class ErrorResponse(BaseModel):
    error: str
    message: str


class AskResponse(BaseModel):
    """Full run_graph() return flattened for JSON serialization.

    Fields explicitly modeled here are the ones the Streamlit UI and
    external clients depend on. Additional keys on the underlying state
    (e.g. ``graph_results``, ``vector_results``, ``resolved_drugs``)
    pass through via ``extra="allow"`` so nothing is silently dropped.
    """

    model_config = ConfigDict(extra="allow")

    mode: str | None = None
    final_output: dict[str, Any] = Field(default_factory=dict)
    critic_score: dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0
    escalated: bool = False
    resolved_drugs: list[dict[str, Any]] = Field(default_factory=list)
    fused_results: list[dict[str, Any]] = Field(default_factory=list)
