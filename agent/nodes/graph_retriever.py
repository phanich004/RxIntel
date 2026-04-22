"""Neo4j graph retriever with action-compatibility filter (invariant #5).

Three dispatch branches keyed off ``state["mode"]``:

* ``ddi_check``   — DIRECT first; if zero rows, fall back to MULTI_HOP
                    so inhibitor↔substrate paths still surface when no
                    explicit INTERACTS_WITH edge was recorded.
* ``polypharmacy`` — DIRECT only, all pairs. No MULTI_HOP fallback (we
                     want a clean pairwise matrix, not enzyme lore).
* ``hybrid``      — ENZYME_FILTER using state["graph_filter"].
* ``alternatives`` / ``describe`` — no-op; the vector retriever owns
                     these modes.

MULTI_HOP has EXACTLY 4 action-pair clauses: inhibitor↔substrate (both
directions) and inducer↔substrate (both directions). Two substrates of
the same enzyme are NOT a clinical interaction — do not add a
substrate/substrate clause. This is non-negotiable (CLAUDE.md invariant
#5); breaking it would re-introduce the tens-of-thousands-of-false-
positives failure mode we designed the filter to prevent.

Each row carries a ``query_type`` field ("direct" | "multi_hop" |
"enzyme_filter") so the reasoning agent can tell lineage at a glance.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

from agent.env import _require_env
from agent.schemas import AgentState

load_dotenv()
logger = logging.getLogger(__name__)

DIRECT_CYPHER = """
MATCH (a:Drug)-[r:INTERACTS_WITH]-(b:Drug)
WHERE a.id IN $drug_ids AND b.id IN $drug_ids AND a.id < b.id
RETURN a.id AS drug_a_id, a.name AS drug_a_name,
       b.id AS drug_b_id, b.name AS drug_b_name,
       r.severity AS severity, r.description AS description
"""

MULTI_HOP_CYPHER = """
MATCH (a:Drug {id:$a})-[ra:VIA_ENZYME]->(e:Enzyme)<-[rb:VIA_ENZYME]-(b:Drug {id:$b})
WHERE (
  (ANY(x IN ra.actions WHERE x = "inhibitor") AND ANY(y IN rb.actions WHERE y = "substrate")) OR
  (ANY(x IN ra.actions WHERE x = "substrate") AND ANY(y IN rb.actions WHERE y = "inhibitor")) OR
  (ANY(x IN ra.actions WHERE x = "inducer")   AND ANY(y IN rb.actions WHERE y = "substrate")) OR
  (ANY(x IN ra.actions WHERE x = "substrate") AND ANY(y IN rb.actions WHERE y = "inducer"))
)
RETURN a.id AS drug_a_id, a.name AS drug_a_name,
       b.id AS drug_b_id, b.name AS drug_b_name,
       e.id AS enzyme_id, e.name AS enzyme_name,
       ra.actions AS a_actions, rb.actions AS b_actions
"""

ENZYME_FILTER_CYPHER = """
MATCH (d:Drug)-[r:VIA_ENZYME]->(e:Enzyme {id:$enzyme})
WHERE $action IN r.actions
RETURN d.id AS drug_id, d.name AS drug_name, r.actions AS actions
"""

_driver_singleton: Driver | None = None


def _driver() -> Driver:
    """Lazy singleton. Test runs that never hit the graph pay nothing.

    Credentials are resolved here (not at import time) so modules can be
    imported without ``NEO4J_*`` set — tests that skip on missing creds
    never trigger the lookup.
    """
    global _driver_singleton
    if _driver_singleton is None:
        _driver_singleton = GraphDatabase.driver(
            _require_env("NEO4J_URI"),
            auth=(_require_env("NEO4J_USERNAME"), _require_env("NEO4J_PASSWORD")),
        )
    return _driver_singleton


def close_driver() -> None:
    """Release the underlying connection pool (for test teardown)."""
    global _driver_singleton
    if _driver_singleton is not None:
        _driver_singleton.close()
        _driver_singleton = None


def _rows(cypher: str, **params: Any) -> tuple[dict[str, Any], ...]:
    with _driver().session() as s:
        result = s.run(cypher, **params)
        return tuple(dict(r) for r in result)


@lru_cache(maxsize=10_000)
def _direct_cached(drug_ids: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    """Pairwise INTERACTS_WITH edges among the given drug ids."""
    return _rows(DIRECT_CYPHER, drug_ids=list(drug_ids))


@lru_cache(maxsize=10_000)
def _multi_hop_cached(pair: tuple[str, str]) -> tuple[dict[str, Any], ...]:
    """Action-compatible enzyme paths between two drugs (order fixed)."""
    a, b = pair
    return _rows(MULTI_HOP_CYPHER, a=a, b=b)


@lru_cache(maxsize=10_000)
def _enzyme_filter_cached(key: tuple[str, str]) -> tuple[dict[str, Any], ...]:
    """Drugs with the given action on the given enzyme."""
    enzyme, action = key
    return _rows(ENZYME_FILTER_CYPHER, enzyme=enzyme, action=action)


def query_direct(drug_ids: list[str]) -> list[dict[str, Any]]:
    """DIRECT all-pairs INTERACTS_WITH query, stamped with query_type."""
    if not drug_ids:
        return []
    key = tuple(sorted(set(drug_ids)))
    rows = _direct_cached(key)
    return [{**dict(r), "query_type": "direct"} for r in rows]


def query_multi_hop(a: str, b: str) -> list[dict[str, Any]]:
    """MULTI_HOP query with invariant-#5 filter, stamped with query_type.

    We lock the pair ordering to ``(min, max)`` for cache locality; the
    underlying Cypher is symmetric in its matches so the result set is
    the same regardless of which direction we pass in.
    """
    if a == b:
        return []
    pair = (a, b) if a < b else (b, a)
    rows = _multi_hop_cached(pair)
    return [{**dict(r), "query_type": "multi_hop"} for r in rows]


def query_enzyme_filter(enzyme: str, action: str) -> list[dict[str, Any]]:
    """ENZYME_FILTER query, stamped with query_type."""
    rows = _enzyme_filter_cached((enzyme, action))
    return [{**dict(r), "query_type": "enzyme_filter"} for r in rows]


def _drug_ids(state: AgentState) -> list[str]:
    resolved = state.get("resolved_drugs", []) or []
    return [r["drug_id"] for r in resolved if r.get("drug_id")]


def graph_retriever(state: AgentState) -> AgentState:
    """LangGraph node: dispatch to DIRECT / MULTI_HOP / ENZYME_FILTER."""
    mode = state.get("mode")
    drug_ids = _drug_ids(state)

    if mode == "ddi_check":
        results = query_direct(drug_ids)
        if not results and len(drug_ids) == 2:
            results = query_multi_hop(drug_ids[0], drug_ids[1])
        return {"graph_results": results}

    if mode == "polypharmacy":
        return {"graph_results": query_direct(drug_ids)}

    if mode == "hybrid":
        gf = state.get("graph_filter") or {}
        enzyme = gf.get("enzyme")
        action = gf.get("action")
        if not enzyme or not action:
            logger.warning(
                "hybrid mode invoked without graph_filter.enzyme/action; "
                "returning empty graph_results"
            )
            return {"graph_results": []}
        return {"graph_results": query_enzyme_filter(enzyme, action)}

    # alternatives, describe, or unset -> vector retriever owns these modes
    return {"graph_results": []}
