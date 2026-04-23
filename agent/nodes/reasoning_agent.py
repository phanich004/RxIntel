"""LangGraph node: clinical reasoning over fused retrieval evidence.

Dispatches to one of five mode-specific system prompts
(``agent.prompts.reasoning_prompts``) and validates the JSON output
against ``ReasoningOutput``. The node enforces CLAUDE.md invariant #7
BEFORE invoking Groq: if no evidence was retrieved, we short-circuit
to a canonical ``severity="unknown"`` / ``insufficient_evidence=True``
output. The LLM is never asked to hallucinate an answer from empty
evidence.

Evidence is packed from ``state["fused_results"]`` (which already
carries RRF-sorted per-drug lineage) and capped at
``MAX_EVIDENCE_TOKENS`` to keep attention quality high — polypharmacy
regimens with five drugs and ten pair rows would otherwise burn ~3000
tokens of evidence before the LLM even reads the question.

``tiktoken`` cl100k_base is used as a tokenizer approximation for the
Llama model — Llama uses SentencePiece BPE, which is close enough for
budget accounting. Exact counts aren't required; what matters is that
the evidence block doesn't balloon unboundedly.

When the critic rejects a draft, ``state["critic_feedback"]`` carries
the reviewer's notes and ``state["reasoning_output"]`` carries the
prior draft. Both are prepended to the user message via
``REVISION_INSTRUCTION_TEMPLATE`` so the model patches the draft
rather than regenerating from scratch.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from agent import token_tracker
from agent.prompts.reasoning_prompts import (
    REASONING_SYSTEM_PROMPTS,
    REVISION_INSTRUCTION_TEMPLATE,
)
from agent.rate_limit import groq_retry
from agent.schemas import AgentState, Mode, ReasoningOutput

logger = logging.getLogger(__name__)

DEFAULT_REASONING_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048
MAX_EVIDENCE_TOKENS = 2500
_TRUNCATION_FOOTER_TOKEN_HEADROOM = 20
# Each rendered line is "- {line}\n" — the "- " + newline adds ~2 tokens
# that _count_tokens(line) doesn't see.
_PER_LINE_RENDER_OVERHEAD = 2
# Each block has a label + newline ("INTERACTION / ENZYME EVIDENCE:\n") ≈ 10
# tokens, plus the "\n\n" separator between blocks (~2 tokens).
_BLOCK_LABEL_OVERHEAD = 12

_GRAPH_MODES: frozenset[str] = frozenset({"ddi_check", "polypharmacy", "hybrid"})
_VECTOR_MODES: frozenset[str] = frozenset({"alternatives", "describe", "hybrid"})

_client: ChatAnthropic | None = None


def _get_client() -> ChatAnthropic:
    global _client
    if _client is None:
        _client = ChatAnthropic(  # type: ignore[call-arg]
            model=os.environ.get("REASONING_MODEL", DEFAULT_REASONING_MODEL),
            temperature=0,
            max_tokens=MAX_TOKENS,
            default_request_timeout=60.0,
        )
    return _client


@lru_cache(maxsize=1)
def _tiktoken_enc() -> Any:
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_tiktoken_enc().encode(text))


# ------------------------------------------------------------------- #
# Evidence formatters
# ------------------------------------------------------------------- #
def _pack_pair_row(row: dict[str, Any]) -> str:
    """Format a direct/multi_hop pairwise graph row."""
    a_id = row.get("drug_a_id", "")
    a_name = row.get("drug_a_name", "") or a_id
    b_id = row.get("drug_b_id", "")
    b_name = row.get("drug_b_name", "") or b_id
    qtype = row.get("query_type", "")
    if qtype == "direct":
        sev = row.get("severity", "unclassified") or "unclassified"
        desc = (row.get("description") or "").strip().replace("\n", " ")
        return f"{a_name} ({a_id}) + {b_name} ({b_id}) [direct]: severity={sev}; {desc}"
    # multi_hop
    enz = row.get("enzyme_name") or row.get("enzyme_id") or "unknown enzyme"
    a_act = ",".join(row.get("a_actions") or []) or "?"
    b_act = ",".join(row.get("b_actions") or []) or "?"
    return (
        f"{a_name} ({a_id}) + {b_name} ({b_id}) "
        f"[multi_hop via {enz}]: {a_name} {a_act}; {b_name} {b_act}"
    )


def _pack_enzyme_row(row: dict[str, Any]) -> str:
    """Format an enzyme_filter graph row (one drug, one enzyme)."""
    did = row.get("drug_id", "")
    name = row.get("drug_name", "") or did
    enz = row.get("enzyme_name") or row.get("enzyme_id") or "unknown enzyme"
    actions = ",".join(row.get("actions") or []) or "?"
    return f"{name} ({did}) — enzyme_filter: {actions} of {enz}"


def _pack_vector_chunk(row: dict[str, Any]) -> str:
    """Format a single vector retrieval chunk."""
    did = row.get("drug_id", "")
    name = row.get("drug_name", "") or did
    coll = row.get("collection", "")
    text = (row.get("text") or "").strip().replace("\n", " ")
    return f"{name} ({did}) [{coll}]: {text}"


def _collect_lines(
    fused: list[dict[str, Any]], mode: Mode | None
) -> tuple[list[str], list[str]]:
    """Walk fused_results in RRF order; return (graph_lines, vector_lines).

    De-duplication is by formatted line content — if two drugs share
    the same INTERACTS_WITH row (pairwise expansion), we emit it once.
    """
    graph_lines: list[str] = []
    vector_lines: list[str] = []
    seen_g: set[str] = set()
    seen_v: set[str] = set()

    use_graph = mode in _GRAPH_MODES
    use_vector = mode in _VECTOR_MODES

    for rec in fused:
        if use_graph:
            for g in rec.get("graph", []) or []:
                qtype = g.get("query_type", "")
                if qtype in ("direct", "multi_hop"):
                    line = _pack_pair_row(g)
                elif qtype == "enzyme_filter":
                    line = _pack_enzyme_row(g)
                else:
                    continue
                if line not in seen_g:
                    seen_g.add(line)
                    graph_lines.append(line)
        if use_vector:
            for v in rec.get("vector", []) or []:
                line = _pack_vector_chunk(v)
                if line not in seen_v:
                    seen_v.add(line)
                    vector_lines.append(line)

    return graph_lines, vector_lines


def _trim_to_budget(
    lines: list[str], budget_remaining: int
) -> tuple[list[str], int]:
    """Take as many lines as fit in ``budget_remaining`` tokens.

    Returns (kept_lines, truncated_count). Budget accounting includes
    the ``- `` bullet prefix + newline per line, and reserves
    ``_TRUNCATION_FOOTER_TOKEN_HEADROOM`` tokens for the footer so a
    maxed-out block doesn't exceed budget once the footer is appended.
    """
    kept: list[str] = []
    used = 0
    for idx, line in enumerate(lines):
        t = _count_tokens(line) + _PER_LINE_RENDER_OVERHEAD
        if used + t + _TRUNCATION_FOOTER_TOKEN_HEADROOM > budget_remaining:
            return kept, len(lines) - idx
        kept.append(line)
        used += t
    return kept, 0


def _render_block(label: str, lines: list[str], truncated: int) -> str:
    body = "\n".join(f"- {ln}" for ln in lines)
    block = f"{label}\n{body}"
    if truncated:
        block += (
            f"\n- [... {truncated} additional entries truncated "
            f"for context budget ...]"
        )
    return block


def _pack_evidence(state: AgentState) -> str:
    """Format fused_results into a token-capped, mode-shaped evidence block."""
    fused = state.get("fused_results") or []
    if not fused:
        return "<no evidence available>"

    mode: Mode | None = state.get("mode")
    constraint = state.get("semantic_constraint") or ""

    graph_lines, vector_lines = _collect_lines(fused, mode)

    if not graph_lines and not vector_lines:
        return "<no evidence available>"

    blocks: list[str] = []
    header = f"Semantic constraint: {constraint}" if constraint else ""
    used_header = _count_tokens(header) if header else 0
    budget = MAX_EVIDENCE_TOKENS - used_header
    if header:
        blocks.append(header)

    if graph_lines:
        kept, trunc = _trim_to_budget(
            graph_lines, budget - _BLOCK_LABEL_OVERHEAD
        )
        if kept:
            block = _render_block(
                "INTERACTION / ENZYME EVIDENCE:", kept, trunc
            )
            blocks.append(block)
            budget -= _count_tokens(block)

    if vector_lines and budget > _BLOCK_LABEL_OVERHEAD:
        kept, trunc = _trim_to_budget(
            vector_lines, budget - _BLOCK_LABEL_OVERHEAD
        )
        if kept:
            blocks.append(_render_block("DRUG TEXT EVIDENCE:", kept, trunc))

    return "\n\n".join(blocks) if blocks else "<no evidence available>"


# ------------------------------------------------------------------- #
# Invariant #7 guard
# ------------------------------------------------------------------- #
def _has_evidence(state: AgentState) -> bool:
    return bool(state.get("fused_results"))


def _derive_sources(state: AgentState) -> list[str]:
    """Drug IDs worth naming in the insufficient-evidence output.

    We prefer resolved_drugs so the refusal message still cites the
    drugs the user asked about, even though no evidence came back.
    """
    resolved = state.get("resolved_drugs") or []
    return [r["drug_id"] for r in resolved if r.get("drug_id")]


def _insufficient_output(
    mode: Mode | None, sources: list[str]
) -> ReasoningOutput:
    """Canonical refusal per CLAUDE.md invariant #7."""
    effective_mode: Mode = mode or "describe"
    if mode == "describe":
        return ReasoningOutput(
            mode=effective_mode,
            severity="n/a",
            insufficient_evidence=True,
            summary="No descriptive chunks for this drug were retrieved.",
            mechanism="",
            recommendation=(
                "Escalate to human clinical review — "
                "automated evidence is insufficient."
            ),
            confidence=0.0,
            sources=sources,
        )
    return ReasoningOutput(
        mode=effective_mode,
        severity="unknown",
        insufficient_evidence=True,
        mechanism=(
            "No retrieval evidence was available to ground a clinical claim."
        ),
        recommendation=(
            "Escalate to human clinical review — "
            "automated evidence is insufficient."
        ),
        confidence=0.0,
        sources=sources,
    )


# ------------------------------------------------------------------- #
# User message + LLM invocation
# ------------------------------------------------------------------- #
def _format_user_message(state: AgentState) -> str:
    """Build the user turn — revision preface + question + evidence."""
    parts: list[str] = []
    feedback = state.get("critic_feedback")
    prior = state.get("reasoning_output")
    if feedback and prior:
        parts.append(
            REVISION_INSTRUCTION_TEMPLATE.format(
                feedback=feedback,
                prior_draft=json.dumps(prior, ensure_ascii=False),
            )
        )
    query = state.get("query", "") or ""
    parts.append(f"User question: {query.strip()}")
    parts.append(f"Evidence:\n{_pack_evidence(state)}")
    return "\n\n".join(parts)


def _extract_json(text: str) -> str:
    """Extract a JSON object from Claude output that may wrap it in prose.

    Groq's ``response_format={"type":"json_object"}`` forced plain JSON.
    Anthropic has no equivalent so the critic/reasoning prompts rely on
    the model to "respond with JSON". At ``temperature=0`` this is
    usually clean, but occasionally Claude prefixes a sentence or wraps
    the output in a `````json`` fence, which crashes
    ``json.loads`` at column 0. This helper handles:

    * plain JSON: ``{"foo": 1}``
    * fenced JSON: `````json\\n{"foo": 1}\\n`````
    * prose prefix: ``Here is the analysis:\\n\\n{"foo": 1}``
    * prose suffix: ``{"foo": 1}\\n\\nNote: draft.``
    * a false-positive ``{`` earlier in the prose that isn't a JSON object

    Returns the raw extracted text. If nothing valid is found the
    unchanged input is returned so the caller's ``json.loads`` raises
    the canonical ``JSONDecodeError`` at the original offset.
    """
    t = text.strip()

    # Peel ```json / ``` fences first.
    if t.startswith("```"):
        lines = t.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()

    # Fast path: already clean top-level JSON object.
    if t.startswith("{") and t.endswith("}"):
        try:
            json.loads(t)
            return t
        except json.JSONDecodeError:
            pass  # fall through to balanced-brace scan

    # Slow path: scan for a balanced ``{...}`` block. TEMP diagnostic log
    # — we want to measure how often Claude produces non-trivial wrapping
    # during Phase B so we know whether structured_output is worth the
    # migration cost. Remove after Day 5-6 decision.
    logger.warning("_extract_json: bracket-matching path used (len=%d)", len(t))

    start = t.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(t)):
            ch = t[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = t[start : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break  # this candidate was junk, try the next '{'
        start = t.find("{", start + 1)

    return t


@groq_retry
def _invoke(system: str, user: str) -> str:
    client = _get_client()
    response = client.invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    token_tracker.record("reasoning", response)
    content = response.content
    if isinstance(content, list):
        content = "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
        )
    return _extract_json(str(content))


def reason(state: AgentState) -> ReasoningOutput:
    """Pure helper: AgentState -> validated ReasoningOutput.

    Short-circuits to the insufficient-evidence output whenever
    fused_results is empty — the LLM is never asked to hallucinate an
    answer from nothing.
    """
    mode: Mode | None = state.get("mode")
    if not _has_evidence(state):
        return _insufficient_output(mode, _derive_sources(state))

    if mode not in REASONING_SYSTEM_PROMPTS:
        logger.warning(
            "reasoning_agent invoked with unknown mode %r; returning refusal",
            mode,
        )
        return _insufficient_output(mode, _derive_sources(state))

    system = REASONING_SYSTEM_PROMPTS[mode]
    user = _format_user_message(state)
    raw = _invoke(system, user)
    parsed = json.loads(raw)
    return ReasoningOutput.model_validate(parsed)


def reasoning_agent(state: AgentState) -> AgentState:
    """LangGraph node: populate ``reasoning_output`` for the critic."""
    out = reason(state)
    return {"reasoning_output": out.model_dump()}
