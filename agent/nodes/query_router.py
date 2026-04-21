"""LangGraph node: 5-way mode classification via Groq.

The router reads ``query`` and ``resolved_drugs`` from AgentState, hands
both to Llama-3.3-70B with temperature=0 and ``response_format=json``,
parses the result into a RouterOutput model, and writes the three
routing fields back onto state.

All LLM traffic flows through ``groq_retry`` so a transient 429 is
absorbed rather than crashing the graph mid-run.
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from agent.prompts.router_prompt import ROUTER_SYSTEM_PROMPT
from agent.rate_limit import groq_retry
from agent.schemas import AgentState, RouterOutput

DEFAULT_ROUTER_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 256

_client: ChatGroq | None = None


def _get_client() -> ChatGroq:
    global _client
    if _client is None:
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
    content = response.content
    if isinstance(content, list):
        # langchain occasionally returns content blocks; flatten for JSON
        content = "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
        )
    return str(content)


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
