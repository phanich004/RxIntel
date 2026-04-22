"""End-to-end LangGraph wiring tests.

Five offline tests mock the five node-module imports on ``agent.graph``
so the full topology runs without Neo4j, ChromaDB, or Groq. One live
test (``@pytest.mark.skip`` by default) exercises the real pipeline
against warfarin + aspirin.

Offline tests build a fresh graph per test via ``build_graph()`` rather
than using the module-level ``AGENT_GRAPH`` singleton so patches take
effect deterministically.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, cast
from unittest.mock import MagicMock

import pytest

from agent import graph as graph_module
from agent.graph import build_graph, run_graph
from agent.schemas import AgentState


# ---------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------- #
NodeFn = Callable[[AgentState], AgentState]


def _node(update: dict[str, Any], recorder: list[AgentState] | None = None) -> NodeFn:
    """Build a deterministic fake node that returns a fixed partial update."""

    def _fn(state: AgentState) -> AgentState:
        if recorder is not None:
            recorder.append(cast(AgentState, dict(state)))
        return cast(AgentState, dict(update))

    return _fn


def _sequence_node(updates: list[dict[str, Any]], recorder: list[AgentState]) -> NodeFn:
    """Fake node whose return value advances through ``updates`` on each call."""
    idx = {"i": 0}

    def _fn(state: AgentState) -> AgentState:
        recorder.append(cast(AgentState, dict(state)))
        u = updates[min(idx["i"], len(updates) - 1)]
        idx["i"] += 1
        return cast(AgentState, dict(u))

    return _fn


def _patch_nodes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    entity_resolver: NodeFn,
    query_router: NodeFn,
    retrieval_dispatch: NodeFn,
    reasoning_agent: NodeFn,
    critic_agent: NodeFn,
) -> None:
    monkeypatch.setattr(graph_module, "entity_resolver", entity_resolver)
    monkeypatch.setattr(graph_module, "query_router", query_router)
    monkeypatch.setattr(graph_module, "retrieval_dispatch", retrieval_dispatch)
    monkeypatch.setattr(graph_module, "reasoning_agent", reasoning_agent)
    monkeypatch.setattr(graph_module, "critic_agent", critic_agent)


def _approved_critic() -> dict[str, Any]:
    return {
        "critic_score": {
            "approved": True,
            "factual_accuracy": 0.95,
            "safety_score": 0.95,
            "completeness_score": 1.0,
            "composite": 0.95,
            "issues": [],
            "revision_prompt": "",
        },
        "critic_feedback": None,
    }


def _rejected_critic(issues: list[str] | None = None) -> dict[str, Any]:
    return {
        "critic_score": {
            "approved": False,
            "factual_accuracy": 0.4,
            "safety_score": 0.5,
            "completeness_score": 0.7,
            "composite": 0.48,
            "issues": issues or ["severity unsupported"],
            "revision_prompt": "Cite a specific evidence row for severity.",
        },
        "critic_feedback": "Cite a specific evidence row for severity.",
    }


def _escalated_critic(issues: list[str] | None = None) -> dict[str, Any]:
    """Last-rejection update emitted by critic_agent past MAX_RETRIES."""
    return {
        "critic_score": {
            "approved": False,
            "factual_accuracy": 0.4,
            "safety_score": 0.5,
            "completeness_score": 0.7,
            "composite": 0.48,
            "issues": issues or ["still unsupported"],
            "revision_prompt": "",
        },
        "critic_feedback": None,
        "escalated": True,
    }


def _reasoning_output(mode: str = "ddi_check", severity: str = "Major") -> dict[str, Any]:
    return {
        "mode": mode,
        "severity": severity,
        "interacting_pairs": [
            {"drug_a": "Warfarin", "drug_b": "Aspirin", "severity": severity}
        ],
        "candidates": [],
        "summary": "",
        "mechanism": "additive anticoagulant effect",
        "recommendation": "avoid co-administration; monitor INR closely",
        "confidence": 0.9,
        "insufficient_evidence": False,
        "sources": ["DB00682", "DB00945"],
    }


# ---------------------------------------------------------------- #
# Offline tests
# ---------------------------------------------------------------- #
def test_happy_path_approved_first_try(monkeypatch: pytest.MonkeyPatch) -> None:
    """Approved on attempt 0 — finalize populates envelope, retry_count stays at 0."""
    reasoning_calls: list[AgentState] = []
    critic_calls: list[AgentState] = []

    _patch_nodes(
        monkeypatch,
        entity_resolver=_node({"resolved_drugs": [{"id": "DB00682"}, {"id": "DB00945"}]}),
        query_router=_node({"mode": "ddi_check"}),
        retrieval_dispatch=_node({"fused_results": [{"drug_id": "DB00682"}]}),
        reasoning_agent=_node(
            {"reasoning_output": _reasoning_output()}, recorder=reasoning_calls
        ),
        critic_agent=_node(_approved_critic(), recorder=critic_calls),
    )

    g = build_graph()
    final_state = g.invoke(
        {
            "query": "Is warfarin safe with aspirin?",
            "resolved_drugs": [],
            "graph_results": [],
            "vector_results": [],
            "fused_results": [],
            "retry_count": 0,
            "escalated": False,
        },
        config={"recursion_limit": 50},
    )

    assert len(reasoning_calls) == 1
    assert len(critic_calls) == 1

    # final_output is a flat reasoning output + escalation metadata.
    final_output = final_state["final_output"]
    assert final_output["mode"] == "ddi_check"
    assert final_output["severity"] == "Major"
    assert final_output["escalated"] is False
    # Escalation-only field must not be present on approved runs.
    assert "final_critic_issues" not in final_output

    # Pipeline metadata lives on the outer merged state.
    assert final_state["retry_count"] == 0
    assert final_state["escalated"] is False
    assert final_state["critic_score"]["approved"] is True


def test_retry_then_approve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject once, then approve — retry_count=1, reasoning runs twice."""
    reasoning_calls: list[AgentState] = []
    critic_calls: list[AgentState] = []

    _patch_nodes(
        monkeypatch,
        entity_resolver=_node({"resolved_drugs": [{"id": "DB00682"}]}),
        query_router=_node({"mode": "ddi_check"}),
        retrieval_dispatch=_node({"fused_results": [{"drug_id": "DB00682"}]}),
        reasoning_agent=_node(
            {"reasoning_output": _reasoning_output()}, recorder=reasoning_calls
        ),
        critic_agent=_sequence_node(
            [_rejected_critic(), _approved_critic()], recorder=critic_calls
        ),
    )

    g = build_graph()
    final_state = g.invoke(
        {
            "query": "Is warfarin safe with aspirin?",
            "resolved_drugs": [],
            "graph_results": [],
            "vector_results": [],
            "fused_results": [],
            "retry_count": 0,
            "escalated": False,
        },
        config={"recursion_limit": 50},
    )

    assert len(reasoning_calls) == 2
    assert len(critic_calls) == 2
    # Second reasoning call sees the incremented retry_count.
    assert reasoning_calls[1]["retry_count"] == 1

    final_output = final_state["final_output"]
    assert final_output["mode"] == "ddi_check"
    assert final_output["escalated"] is False
    assert final_state["retry_count"] == 1
    assert final_state["critic_score"]["approved"] is True


def test_max_retries_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Always reject — retry_count=2, escalated=True, issues surfaced."""
    reasoning_calls: list[AgentState] = []
    critic_calls: list[AgentState] = []

    # Attempts 0 and 1: reject-with-feedback. Attempt 2: critic_agent sees
    # retry_count >= MAX_RETRIES and emits the escalated update.
    _patch_nodes(
        monkeypatch,
        entity_resolver=_node({"resolved_drugs": []}),
        query_router=_node({"mode": "ddi_check"}),
        retrieval_dispatch=_node({"fused_results": []}),
        reasoning_agent=_node(
            {"reasoning_output": _reasoning_output()}, recorder=reasoning_calls
        ),
        critic_agent=_sequence_node(
            [
                _rejected_critic(["first"]),
                _rejected_critic(["second"]),
                _escalated_critic(["final: still unsupported"]),
            ],
            recorder=critic_calls,
        ),
    )

    g = build_graph()
    final_state = g.invoke(
        {
            "query": "bogus",
            "resolved_drugs": [],
            "graph_results": [],
            "vector_results": [],
            "fused_results": [],
            "retry_count": 0,
            "escalated": False,
        },
        config={"recursion_limit": 50},
    )

    assert len(reasoning_calls) == 3
    assert len(critic_calls) == 3

    final_output = final_state["final_output"]
    assert final_output["escalated"] is True
    assert final_output["final_critic_issues"] == ["final: still unsupported"]
    # Pipeline metadata for escalated runs.
    assert final_state["retry_count"] == 2
    assert final_state["escalated"] is True
    assert final_state["critic_score"]["approved"] is False


def test_describe_mode_routes_through_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mode=describe set by router reaches retrieval_dispatch unchanged."""
    dispatch_calls: list[AgentState] = []

    _patch_nodes(
        monkeypatch,
        entity_resolver=_node({"resolved_drugs": [{"id": "DB01234"}]}),
        query_router=_node({"mode": "describe"}),
        retrieval_dispatch=_node(
            {"fused_results": [{"drug_id": "DB01234", "vector": [{"text": "..."}]}]},
            recorder=dispatch_calls,
        ),
        reasoning_agent=_node(
            {"reasoning_output": _reasoning_output(mode="describe", severity="n/a")}
        ),
        critic_agent=_node(_approved_critic()),
    )

    g = build_graph()
    final_state = g.invoke(
        {
            "query": "What is semaglutide?",
            "resolved_drugs": [],
            "graph_results": [],
            "vector_results": [],
            "fused_results": [],
            "retry_count": 0,
            "escalated": False,
        },
        config={"recursion_limit": 50},
    )

    assert len(dispatch_calls) == 1
    # Dispatch was invoked with the router's mode already present in state,
    # which is the routing contract the real retrieval_dispatch depends on.
    assert dispatch_calls[0].get("mode") == "describe"
    assert final_state["mode"] == "describe"
    assert final_state["final_output"]["mode"] == "describe"
    assert final_state["final_output"]["severity"] == "n/a"


def test_mlflow_fallback_on_unreachable_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MLflow failure must fall back to local JSON without breaking the pipeline."""
    # Force fallback: the wrapper catches any Exception from mlflow paths.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://invalid.example:9999")

    import mlflow  # type: ignore[import-untyped]

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise ConnectionError("mlflow unreachable")

    monkeypatch.setattr(mlflow, "start_run", _boom)

    # Redirect local trace dir so the test doesn't pollute ~/.rxintel.
    monkeypatch.setattr(graph_module, "_LOCAL_TRACE_DIR", tmp_path)

    _patch_nodes(
        monkeypatch,
        entity_resolver=_node({"resolved_drugs": [{"id": "DB00682"}]}),
        query_router=_node({"mode": "ddi_check"}),
        retrieval_dispatch=_node({"fused_results": [{"drug_id": "DB00682"}]}),
        reasoning_agent=_node({"reasoning_output": _reasoning_output()}),
        critic_agent=_node(_approved_critic()),
    )

    # Force a fresh graph so the patched nodes are wired in.
    monkeypatch.setattr(graph_module, "AGENT_GRAPH", None)

    result = run_graph("Is warfarin safe with aspirin?")

    # run_graph returns the full merged final state.
    assert result["mode"] == "ddi_check"
    assert result["critic_score"]["approved"] is True
    assert result["final_output"]["mode"] == "ddi_check"
    assert result["final_output"]["severity"] == "Major"

    # A trace file must exist in the redirected local trace dir.
    traces = list(tmp_path.glob("*.json"))
    assert len(traces) == 1
    trace = json.loads(traces[0].read_text())
    assert "params" in trace and "metrics" in trace and "artifacts" in trace
    assert trace["params"]["query"] == "Is warfarin safe with aspirin?"
    assert "final_output.json" in trace["artifacts"]
    assert trace["artifacts"]["final_output.json"]["severity"] == "Major"
    assert trace["metrics"]["retry_count"] == 0.0
    assert "total_latency_s" in trace["metrics"]


def test_pipeline_error_surfaces_correctly_through_mlflow_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: pipeline exceptions must propagate from the
    _mlflow_run_or_fallback contextmanager without being masked as
    'generator didn't stop after throw()'.
    """
    from agent import graph as graph_module

    # Force MLflow success path (not fallback), so the bug fix is tested
    # against real mlflow context manager semantics.

    # Patch the graph's invoke to raise a specific error
    class PipelineError(Exception):
        pass

    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = PipelineError("simulated pipeline failure")

    monkeypatch.setattr(graph_module, "_get_graph", lambda: mock_graph)

    # The exception MUST propagate as PipelineError, not a contextmanager runtime error
    with pytest.raises(PipelineError, match="simulated pipeline failure"):
        graph_module.run_graph("test query")


# ---------------------------------------------------------------- #
# Live end-to-end — skipped by default; enable manually.
# ---------------------------------------------------------------- #
@pytest.mark.skip(reason="live end-to-end — run manually with real Neo4j/Chroma/Groq")
def test_live_end_to_end_warfarin_aspirin() -> None:
    """Real pipeline: expect Major/Moderate DDI, approved on first or second try.

    The verbose print block below only runs when the decorator is
    stripped for a manual run — in CI / default test runs it's skipped
    and the prints never fire. Kept in the function body so future
    manual runs get the same debug surface without having to re-type it.
    """
    t0 = time.perf_counter()
    result = run_graph("Is warfarin safe with aspirin?")
    elapsed = time.perf_counter() - t0

    assert result["mode"] == "ddi_check"
    assert result["final_output"]["severity"] in {"Major", "Moderate"}
    assert result["critic_score"]["approved"] is True
    assert result["retry_count"] <= 1
    assert elapsed < 15.0

    import json as _json

    print("\n====== END-TO-END LIVE RESULT ======")
    print("Query: 'Is warfarin safe with aspirin?'")
    print(f"Mode: {result['mode']}")
    print(f"Severity: {result['final_output'].get('severity')}")
    print(f"Approved: {result['critic_score'].get('approved')}")
    print(f"Retry count: {result['retry_count']}")
    print(f"Total latency (s): {elapsed:.3f}")
    print(
        f"Resolved drugs: "
        f"{[r.get('drug_id') for r in result.get('resolved_drugs', [])]}"
    )
    print("\nFinal output:")
    print(_json.dumps(result["final_output"], indent=2))
    print("\nCritic score breakdown:")
    print(_json.dumps(result["critic_score"], indent=2))
    print("====================================")
