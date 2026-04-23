"""LangGraph node: 5-way mode classification.

The router reads ``query`` and ``resolved_drugs`` from AgentState, hands
both to a small chat LLM with temperature=0, parses the result into a
RouterOutput model, and writes the three routing fields back onto state.

LLM provider is pluggable via ``ROUTER_PROVIDER`` env var:

* ``ROUTER_PROVIDER=groq`` (default, production) — Llama-3.3-70B on Groq
  with ``response_format=json_object`` enforced server-side.
* ``ROUTER_PROVIDER=anthropic`` — Claude Sonnet (or whatever
  ``ROUTER_MODEL_ANTHROPIC`` names). Used when Groq's free-tier TPD is
  exhausted (e.g. during Phase B benchmarks). Anthropic has no JSON-mode
  equivalent so output is run through ``_extract_json`` defensively.

All LLM traffic flows through ``groq_retry`` (provider-agnostic in
practice — it handles both Groq and Anthropic 429s).
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from agent import token_tracker
from agent.nodes.reasoning_agent import _extract_json
from agent.prompts.router_prompt import ROUTER_SYSTEM_PROMPT
from agent.rate_limit import groq_retry
from agent.schemas import AgentState, RouterOutput

DEFAULT_ROUTER_MODEL = "llama-3.3-70b-versatile"
DEFAULT_ROUTER_MODEL_ANTHROPIC = "claude-sonnet-4-6"
MAX_TOKENS = 256

_client: ChatGroq | ChatAnthropic | None = None


def _provider() -> str:
    """``"anthropic"`` or ``"groq"`` (default)."""
    return os.environ.get("ROUTER_PROVIDER", "groq").strip().lower()


def _get_client() -> ChatGroq | ChatAnthropic:
    global _client
    if _client is None:
        if _provider() == "anthropic":
            _client = ChatAnthropic(  # type: ignore[call-arg]
                model=os.environ.get(
                    "ROUTER_MODEL_ANTHROPIC", DEFAULT_ROUTER_MODEL_ANTHROPIC
                ),
                temperature=0,
                max_tokens=MAX_TOKENS,
                default_request_timeout=30.0,
            )
        else:
            _client = ChatGroq(
                model=os.environ.get("ROUTER_MODEL", DEFAULT_ROUTER_MODEL),
                temperature=0,
                max_tokens=MAX_TOKENS,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
    return _client


def _format_user_message(query: str, resolved_drugs: list[dict[str, Any]]) -> str:
    ids = [r["drug_id"] for r in resolved_drugs if r.get("drug_id")]
    # Match the EXAMPLES block exactly: the `drugs: [...]` line must be
    # present even when empty, so the model learns to ignore that slot
    # rather than pattern-match on its absence.
    return f"Q: {query.strip()}\ndrugs: [{', '.join(ids)}]"


@groq_retry
def _classify(system: str, user: str) -> str:
    client = _get_client()
    response = client.invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    # Anthropic path bills against the $10 budget; track tokens so the
    # benchmark cost ledger stays accurate. Groq path is on a separate
    # quota and outside the cost model, so we skip tracking there.
    if _provider() == "anthropic":
        token_tracker.record("router", response)
    content = response.content
    if isinstance(content, list):
        # langchain occasionally returns content blocks; flatten for JSON
        content = "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
        )
    return _extract_json(str(content))


def classify(query: str, resolved_drugs: list[dict[str, Any]]) -> RouterOutput:
    """Pure helper: query + resolved_drugs -> RouterOutput. Testable."""
    user = _format_user_message(query, resolved_drugs)
    raw = _classify(ROUTER_SYSTEM_PROMPT, user)
    parsed = json.loads(raw)
    return RouterOutput.model_validate(parsed)


def query_router(state: AgentState) -> AgentState:
    """LangGraph node: populate mode + graph_filter + semantic_constraint."""
    query = state.get("query", "")
    resolved = state.get("resolved_drugs", []) or []
    result = classify(query, resolved)
    return {
        "mode": result.mode,
        "graph_filter": result.graph_filter,
        "semantic_constraint": result.semantic_constraint,
    }
