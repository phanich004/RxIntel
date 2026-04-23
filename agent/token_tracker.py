"""Process-wide, per-agent token usage log for Anthropic calls.

The reasoning and critic nodes call into this module after each
``ChatAnthropic.invoke`` to record ``input_tokens`` and ``output_tokens``
from the response's ``usage_metadata`` (LangChain >=0.3 convention,
populated by the langchain-anthropic wrapper).

Design note: this is a module-level list, not a callback handler. The
calibration script needs to attribute token usage to (query_id,
agent_name) and LangGraph nodes don't accept per-invocation callbacks
without threading them through state, which would bloat the node API.
A module-level log is the smallest thing that works.

Usage:

    from agent import token_tracker

    token_tracker.reset()
    ...run one query through the graph...
    for rec in token_tracker.records():
        print(rec["agent"], rec["input_tokens"], rec["output_tokens"])
"""

from __future__ import annotations

import threading
from typing import Any

_LOCK = threading.Lock()
_RECORDS: list[dict[str, Any]] = []


def record(agent: str, response: Any) -> None:
    """Extract usage from a LangChain AIMessage response and append a row.

    Supports both ``usage_metadata`` (preferred, LangChain >=0.3) and
    ``response_metadata["usage"]`` (older shape). Silently records zeros
    if neither is present — we don't want instrumentation failures to
    abort a benchmark run.
    """
    input_tokens = 0
    output_tokens = 0

    usage = getattr(response, "usage_metadata", None) or {}
    if usage:
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
    else:
        meta = getattr(response, "response_metadata", {}) or {}
        u = meta.get("usage") or {}
        input_tokens = int(u.get("input_tokens", 0) or 0)
        output_tokens = int(u.get("output_tokens", 0) or 0)

    with _LOCK:
        _RECORDS.append(
            {
                "agent": agent,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )


def reset() -> None:
    """Clear the log — call before each benchmark query."""
    with _LOCK:
        _RECORDS.clear()


def records() -> list[dict[str, Any]]:
    """Return a snapshot copy of the log."""
    with _LOCK:
        return list(_RECORDS)
