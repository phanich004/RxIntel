"""LangGraph orchestration node: dispatch to the right retriever(s).

Pure composition — no LLM or DB calls originate here; this node just
invokes the graph retriever, the vector retriever, or both in parallel
depending on ``state["mode"]``, then runs fusion on whatever they
returned. Each downstream node owns its own IO and connection pool.

Dispatch table:

* ``ddi_check`` / ``polypharmacy`` — graph only, then fusion wraps it.
* ``alternatives`` / ``describe``  — vector only, then fusion wraps it.
* ``hybrid`` — graph + vector in parallel (ThreadPoolExecutor), then
  fusion merges via RRF.

Why threads, not asyncio: the Neo4j Python driver and ChromaDB client
in the current install are both sync-only, so ``asyncio.gather`` gives
us no concurrency. A two-worker thread pool lets the ~100ms Cypher
call and the ~100–300ms Chroma call overlap instead of running
serially.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import cast

from agent.nodes.fusion import fusion
from agent.nodes.graph_retriever import graph_retriever
from agent.nodes.vector_retriever import vector_retriever
from agent.schemas import AgentState


def _merge(*updates: AgentState) -> AgentState:
    """Dict-union a sequence of partial state updates into one AgentState."""
    out: dict[str, object] = {}
    for u in updates:
        out.update(u)
    return cast(AgentState, out)


def retrieval_dispatch(state: AgentState) -> AgentState:
    """Route ``state["mode"]`` to the correct retriever(s) and fuse."""
    mode = state.get("mode")

    if mode == "hybrid":
        with ThreadPoolExecutor(max_workers=2) as pool:
            graph_future = pool.submit(graph_retriever, state)
            vector_future = pool.submit(vector_retriever, state)
            graph_update = graph_future.result()
            vector_update = vector_future.result()
        merged = _merge(state, graph_update, vector_update)
        return _merge(graph_update, vector_update, fusion(merged))

    if mode in ("ddi_check", "polypharmacy"):
        graph_update = graph_retriever(state)
        merged = _merge(state, graph_update)
        return _merge(graph_update, fusion(merged))

    if mode in ("alternatives", "describe"):
        vector_update = vector_retriever(state)
        merged = _merge(state, vector_update)
        return _merge(vector_update, fusion(merged))

    return cast(
        AgentState,
        {"graph_results": [], "vector_results": [], "fused_results": []},
    )
