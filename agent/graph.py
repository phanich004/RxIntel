"""End-to-end LangGraph wiring for the drug-intelligence pipeline.

This module is the orchestration center. Nodes are implemented and
unit-tested independently; this file is the only place that knows the
full flow and the retry loop.

Topology::

    entity_resolver
        → query_router
        → retrieval_dispatch
        → reasoning_agent
        → critic_agent
        → (conditional)
              ├── approved OR escalated → finalize → END
              └── rejected + retries left → increment_retry → reasoning_agent

Retry semantics live in ``critic_agent`` (sets ``critic_feedback`` and
``escalated``); ``_should_retry`` only reads those flags and routes
accordingly. ``_increment_retry`` is a separate pass-through node so the
retry bookkeeping is visible in traces, not buried in an edge.

Observability:

* Every invoke runs under ``_mlflow_run_or_fallback``. If MLflow is
  unreachable we write a local JSON trace to
  ``~/.rxintel/traces/<unix-millis>.json`` so dev loops still have a
  tracing record — the pipeline never fails because telemetry failed.
* A ``_NodeLatencyHandler`` sums per-node wall time across retry
  re-entries so the trace shows, e.g., ``reasoning_agent: 4.21s``
  covering all attempts.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, Iterator, Literal, cast
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes.critic_agent import critic_agent
from agent.nodes.entity_resolver import entity_resolver
from agent.nodes.query_router import query_router
from agent.nodes.reasoning_agent import reasoning_agent
from agent.nodes.retrieval_dispatch import retrieval_dispatch
from agent.schemas import AgentState

logger = logging.getLogger(__name__)

# Names that match ``add_node`` registrations below. The callback filters
# by this set so LangGraph's internal bookkeeping chains don't pollute
# the per-node latency report.
_TRACKED_NODES: Final[frozenset[str]] = frozenset(
    {
        "entity_resolver",
        "query_router",
        "retrieval_dispatch",
        "reasoning_agent",
        "critic_agent",
        "increment_retry",
        "finalize",
    }
)

# LangGraph recursion_limit counts every node transition including the
# retry loop. With MAX_RETRIES=2 the longest legitimate path is ~15 hops;
# 50 gives ample headroom without masking a real infinite loop.
_RECURSION_LIMIT: Final[int] = 50

_LOCAL_TRACE_DIR: Final[Path] = Path.home() / ".rxintel" / "traces"


def _increment_retry(state: AgentState) -> AgentState:
    """Pass-through node that bumps ``retry_count`` between attempts.

    Kept as an explicit node (rather than an edge side-effect) so retry
    hops show up in LangGraph traces as a named step instead of an
    invisible state mutation.
    """
    current = state.get("retry_count", 0) or 0
    return cast(AgentState, {"retry_count": current + 1})


def _finalize(state: AgentState) -> AgentState:
    """Build final_output: reasoning output + escalation envelope.

    final_output is what's returned to API consumers. It IS the
    reasoning output (flat, matching ReasoningOutput schema), plus
    two metadata fields for escalation. Other pipeline state
    (graph/vector results, critic scores, latencies) is available
    on the outer state for consumers that need it.
    """
    reasoning_output = state.get("reasoning_output") or {}
    critic_score = state.get("critic_score") or {}
    escalated = bool(state.get("escalated", False))

    final_output: dict[str, Any] = dict(reasoning_output)
    final_output["escalated"] = escalated
    if escalated:
        final_output["final_critic_issues"] = list(
            critic_score.get("issues", []) or []
        )

    return cast(AgentState, {"final_output": final_output})


def _should_retry(state: AgentState) -> Literal["finalize", "retry"]:
    """Conditional edge: approved or escalated → finalize; else retry.

    ``critic_agent`` has already decided retry vs. escalation by the
    time we read this: ``critic_feedback`` is ``None`` when we're done
    (either approved or over the cap) and a revision prompt otherwise.
    We do not re-derive the decision here to avoid drift.
    """
    critic_score = state.get("critic_score") or {}
    if critic_score.get("approved"):
        return "finalize"
    if state.get("escalated"):
        return "finalize"
    if state.get("critic_feedback"):
        return "retry"
    # Defensive: critic ran but left no feedback and didn't approve —
    # treat as terminal rather than spin.
    return "finalize"


def build_graph() -> CompiledStateGraph:
    """Construct and compile the pipeline graph.

    Factored so tests can build fresh graphs against mocked nodes
    without touching the module-level ``AGENT_GRAPH`` singleton.
    """
    graph = StateGraph(AgentState)

    graph.add_node("entity_resolver", entity_resolver)
    graph.add_node("query_router", query_router)
    graph.add_node("retrieval_dispatch", retrieval_dispatch)
    graph.add_node("reasoning_agent", reasoning_agent)
    graph.add_node("critic_agent", critic_agent)
    graph.add_node("increment_retry", _increment_retry)
    graph.add_node("finalize", _finalize)

    graph.set_entry_point("entity_resolver")
    graph.add_edge("entity_resolver", "query_router")
    graph.add_edge("query_router", "retrieval_dispatch")
    graph.add_edge("retrieval_dispatch", "reasoning_agent")
    graph.add_edge("reasoning_agent", "critic_agent")

    graph.add_conditional_edges(
        "critic_agent",
        _should_retry,
        {"finalize": "finalize", "retry": "increment_retry"},
    )
    graph.add_edge("increment_retry", "reasoning_agent")
    graph.add_edge("finalize", END)

    return graph.compile()


class _NodeLatencyHandler(BaseCallbackHandler):
    """Sum per-node wall time across the whole run, including retries.

    LangGraph emits ``on_chain_start`` / ``on_chain_end`` for every
    node invocation; we key in-flight timers by ``run_id`` so
    concurrent nodes (hybrid retrieval uses a thread pool) don't clobber
    each other. Filtering by ``_TRACKED_NODES`` skips the outer graph
    chain and any library-internal chains.
    """

    def __init__(self) -> None:
        self._starts: dict[str, tuple[str, float]] = {}
        self.durations: dict[str, float] = defaultdict(float)

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or ""
        if name in _TRACKED_NODES:
            self._starts[str(run_id)] = (name, time.perf_counter())

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        entry = self._starts.pop(str(run_id), None)
        if entry is not None:
            name, t0 = entry
            self.durations[name] += time.perf_counter() - t0


class _MLflowTracker:
    """Context handle for a live MLflow run — logs params and metrics."""

    def __init__(self, mlflow_module: Any) -> None:
        self._mlflow = mlflow_module

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._mlflow.log_metrics(metrics)

    def log_dict(self, d: dict[str, Any], artifact_path: str) -> None:
        self._mlflow.log_dict(d, artifact_path)


class _LocalFileTracker:
    """Fallback tracker: buffers trace data, writes one JSON at exit.

    Used when MLflow is unreachable. The goal is not parity with MLflow
    — just that a dev loop with no tracking server still produces a
    persistent artifact. Path is
    ``~/.rxintel/traces/<unix-millis>.json``.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.payload: dict[str, Any] = {
            "params": {},
            "metrics": {},
            "artifacts": {},
        }

    def log_params(self, params: dict[str, Any]) -> None:
        self.payload["params"].update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.payload["metrics"].update(metrics)

    def log_dict(self, d: dict[str, Any], artifact_path: str) -> None:
        self.payload["artifacts"][artifact_path] = d

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.payload, default=_jsonify, indent=2))


def _jsonify(obj: Any) -> Any:
    """json.dumps default hook — best-effort stringify for trace dumps."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


@contextmanager
def _mlflow_run_or_fallback(
    run_name: str,
) -> Iterator[_MLflowTracker | _LocalFileTracker]:
    """Yield an MLflow-backed tracker when possible, local JSON otherwise.

    The try/except covers ONLY the mlflow setup phase. Once we've yielded
    (either an MLflowTracker or the local fallback), exceptions in the
    caller's with block propagate normally — we don't swallow them here.
    Previous version caught body exceptions in the outer except, then
    yielded a second time, producing an opaque "generator didn't stop
    after throw()" that masked every pipeline error.
    """
    mlflow_ctx: Any = None
    mlflow_tracker: _MLflowTracker | None = None
    try:
        import mlflow
        mlflow_ctx = mlflow.start_run(run_name=run_name)
        mlflow_ctx.__enter__()
        mlflow_tracker = _MLflowTracker(mlflow)
    except Exception as exc:
        logger.warning(
            "MLflow unavailable (%s); falling back to local JSON trace", exc
        )
        mlflow_ctx = None

    if mlflow_ctx is not None and mlflow_tracker is not None:
        try:
            yield mlflow_tracker
        finally:
            try:
                mlflow_ctx.__exit__(None, None, None)
            except Exception as exc:
                logger.warning("MLflow end_run failed: %s", exc)
        return

    # Fallback path: local JSON file tracker.
    path = _LOCAL_TRACE_DIR / f"{int(time.time() * 1000)}.json"
    tracker = _LocalFileTracker(path)
    try:
        yield tracker
    finally:
        try:
            tracker.flush()
        except Exception as exc:
            logger.warning("Failed to write local trace to %s: %s", path, exc)


AGENT_GRAPH: CompiledStateGraph | None = None


def _get_graph() -> CompiledStateGraph:
    """Lazy singleton so importing the module doesn't compile the graph."""
    global AGENT_GRAPH
    if AGENT_GRAPH is None:
        AGENT_GRAPH = build_graph()
    return AGENT_GRAPH


def run_graph(query: str, thread_id: str | None = None) -> dict[str, Any]:
    """Run one end-to-end pipeline invocation.

    ``thread_id`` is accepted for API compatibility with future
    checkpointer-backed runs; it's currently a passthrough hint for
    callers and is echoed into the trace params.
    """
    graph = _get_graph()

    # Pre-populate list-typed channels so downstream nodes can always
    # append without first checking for None. ``retry_count`` and
    # ``escalated`` are initialized here for the same reason.
    initial_state: AgentState = cast(
        AgentState,
        {
            "query": query,
            "resolved_drugs": [],
            "graph_results": [],
            "vector_results": [],
            "fused_results": [],
            "retry_count": 0,
            "escalated": False,
        },
    )

    latency_handler = _NodeLatencyHandler()
    t_start = time.perf_counter()

    with _mlflow_run_or_fallback(run_name="rxintel-query") as tracker:
        tracker.log_params(
            {
                "query": query,
                "thread_id": thread_id or "",
            }
        )
        final_state = graph.invoke(
            initial_state,
            config={
                "callbacks": [latency_handler],
                "recursion_limit": _RECURSION_LIMIT,
            },
        )
        total = time.perf_counter() - t_start

        metrics: dict[str, float] = {
            "total_latency_s": total,
            "retry_count": float(final_state.get("retry_count", 0) or 0),
        }
        for name, secs in latency_handler.durations.items():
            metrics[f"latency_{name}_s"] = secs
        tracker.log_metrics(metrics)

        final_output = final_state.get("final_output") or {}
        tracker.log_dict(final_output, "final_output.json")

    return cast(dict[str, Any], dict(final_state))


__all__ = [
    "AGENT_GRAPH",
    "build_graph",
    "run_graph",
]
