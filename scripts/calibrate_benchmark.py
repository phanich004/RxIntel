"""Phase A calibration: run 2 queries through the pipeline and measure
Anthropic token usage before committing to a 50-query Phase B benchmark.

Reads ``evaluation/calibration_queries.yaml``, runs each query through
``agent.graph.run_graph``, reads per-agent token usage from
``agent.token_tracker``, and prints per-query + aggregate cost using the
Sonnet prices supplied in the Step 17 brief
(input $3.00 / 1M, output $15.00 / 1M).

This script performs real Anthropic API calls. Two queries expected to
cost roughly $0.10-$0.20 combined. Stops on the first exception rather
than silently swallowing: a bad call during calibration is a signal to
abort, not to mask.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

# Import after dotenv so ANTHROPIC_API_KEY is in env before ChatAnthropic
# constructs its underlying anthropic.Anthropic() client.
from agent import token_tracker  # noqa: E402
from agent.graph import run_graph  # noqa: E402

QUERIES_PATH = ROOT / "evaluation" / "calibration_queries.yaml"

# Anthropic Sonnet pricing per Step 17 brief (USD per 1M tokens).
INPUT_PRICE_PER_M = 3.00
OUTPUT_PRICE_PER_M = 15.00

# Phase B parameters (for the projection only — Phase B is not run here).
PHASE_B_QUERIES = 50
PHASE_B_SAFETY_BUFFER = 0.20  # +20% for reasoning retries
BUDGET_USD = 10.00


def _cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * INPUT_PRICE_PER_M + output_tokens * OUTPUT_PRICE_PER_M
    ) / 1_000_000


def _sum_by_agent(
    records: list[dict[str, Any]],
) -> dict[str, tuple[int, int, int]]:
    """Return {agent_name: (calls, input_tokens, output_tokens)}."""
    agg: dict[str, tuple[int, int, int]] = {}
    for rec in records:
        calls, i, o = agg.get(rec["agent"], (0, 0, 0))
        agg[rec["agent"]] = (
            calls + 1,
            i + int(rec.get("input_tokens", 0)),
            o + int(rec.get("output_tokens", 0)),
        )
    return agg


def _drug_ids_from_fused(fused: list[dict[str, Any]]) -> set[str]:
    """Collect drug IDs that appeared in any graph row or vector chunk."""
    ids: set[str] = set()
    for rec in fused:
        for g in rec.get("graph", []) or []:
            for k in ("drug_id", "drug_a_id", "drug_b_id"):
                v = g.get(k)
                if v:
                    ids.add(v)
        for v_chunk in rec.get("vector", []) or []:
            did = v_chunk.get("drug_id")
            if did:
                ids.add(did)
        if rec.get("drug_id"):
            ids.add(rec["drug_id"])
    return ids


def _check(spec: dict[str, Any], result: dict[str, Any]) -> list[str]:
    """Return a list of human-readable pass/fail lines."""
    lines: list[str] = []

    # Mode
    actual_mode = result.get("mode")
    expected_mode = spec.get("expected_mode")
    ok_mode = actual_mode == expected_mode
    lines.append(
        f"  mode:      expected={expected_mode!r:18} "
        f"actual={actual_mode!r:18} {'PASS' if ok_mode else 'FAIL'}"
    )

    # Severity (only meaningful when expected is non-null)
    final = result.get("final_output") or {}
    actual_sev = final.get("severity")
    expected_sev = spec.get("expected_severity")
    if expected_sev is None:
        lines.append(
            f"  severity:  expected=(n/a)            actual={actual_sev!r} "
            "SKIP"
        )
    else:
        ok_sev = (
            actual_sev is not None
            and str(actual_sev).lower() == str(expected_sev).lower()
        )
        lines.append(
            f"  severity:  expected={expected_sev!r:18} "
            f"actual={actual_sev!r:18} {'PASS' if ok_sev else 'FAIL'}"
        )

    # Drug presence — we check both resolved drugs and anything that
    # surfaced through retrieval (fused). A query passes if every
    # expected drug shows up somewhere.
    expected_drugs = set(spec.get("expected_drugs") or [])
    resolved_ids = {
        r.get("drug_id") for r in (result.get("resolved_drugs") or [])
    }
    retrieved_ids = _drug_ids_from_fused(result.get("fused_results") or [])
    seen = resolved_ids | retrieved_ids
    missing = expected_drugs - seen
    ok_drugs = not missing
    lines.append(
        f"  drugs:     expected={sorted(expected_drugs)} "
        f"{'PASS' if ok_drugs else f'FAIL (missing {sorted(missing)})'}"
    )
    lines.append(
        f"             resolved={sorted(x for x in resolved_ids if x)}  "
        f"retrieved_n={len(retrieved_ids)}"
    )

    return lines


def run_one(spec: dict[str, Any]) -> dict[str, Any]:
    """Execute one calibration query and return a summary record."""
    qid = spec["id"]
    text = spec["query"]

    print("=" * 78)
    print(f"Query {qid}:  {text!r}")

    token_tracker.reset()
    t0 = time.perf_counter()
    result = run_graph(text)
    elapsed = time.perf_counter() - t0

    usage_by_agent = _sum_by_agent(token_tracker.records())

    reasoning_calls, r_in, r_out = usage_by_agent.get("reasoning", (0, 0, 0))
    critic_calls, c_in, c_out = usage_by_agent.get("critic", (0, 0, 0))
    r_cost = _cost_usd(r_in, r_out)
    c_cost = _cost_usd(c_in, c_out)
    total_cost = r_cost + c_cost

    print(
        f"  reasoning: calls={reasoning_calls} "
        f"in={r_in:5d} out={r_out:5d}  ${r_cost:.4f}"
    )
    print(
        f"  critic:    calls={critic_calls} "
        f"in={c_in:5d} out={c_out:5d}  ${c_cost:.4f}"
    )
    print(f"  retries:   {result.get('retry_count', 0)}")
    print(f"  elapsed:   {elapsed:.2f}s")
    for line in _check(spec, result):
        print(line)
    print(f"  TOTAL COST: ${total_cost:.4f}")

    return {
        "id": qid,
        "mode": spec.get("expected_mode"),
        "reasoning_calls": reasoning_calls,
        "reasoning_in": r_in,
        "reasoning_out": r_out,
        "critic_calls": critic_calls,
        "critic_in": c_in,
        "critic_out": c_out,
        "total_in": r_in + c_in,
        "total_out": r_out + c_out,
        "cost_usd": total_cost,
        "elapsed_s": elapsed,
        "retries": int(result.get("retry_count", 0) or 0),
    }


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-ant"):
        print(
            "ANTHROPIC_API_KEY is not set (expected sk-ant-...). "
            "Aborting calibration.",
            file=sys.stderr,
        )
        return 2

    with QUERIES_PATH.open() as fh:
        loaded = yaml.safe_load(fh)
    queries = loaded.get("queries", [])
    if len(queries) != 2:
        print(
            f"Expected exactly 2 calibration queries, found {len(queries)}.",
            file=sys.stderr,
        )
        return 2

    results = [run_one(spec) for spec in queries]

    # Aggregate
    total_in = sum(r["total_in"] for r in results)
    total_out = sum(r["total_out"] for r in results)
    total_cost = sum(r["cost_usd"] for r in results)
    avg_cost = total_cost / len(results)

    # Phase B projection: 50 × avg × (1 + safety buffer)
    projected = PHASE_B_QUERIES * avg_cost * (1 + PHASE_B_SAFETY_BUFFER)
    remaining = BUDGET_USD - projected

    print("=" * 78)
    print("AGGREGATE (Phase A, 2 queries)")
    print(f"  input_tokens:  {total_in}")
    print(f"  output_tokens: {total_out}")
    print(f"  total cost:    ${total_cost:.4f}")
    print(f"  avg/query:     ${avg_cost:.4f}")
    print()
    print("PHASE B PROJECTION")
    print(
        f"  {PHASE_B_QUERIES} queries × ${avg_cost:.4f}/query "
        f"× {1 + PHASE_B_SAFETY_BUFFER:.2f} (retry buffer) = "
        f"${projected:.2f}"
    )
    print(
        f"  remaining budget after Phase B: "
        f"${BUDGET_USD:.2f} − ${projected:.2f} = ${remaining:.2f}"
    )
    print(
        "  NOTE: reasoning-heavy modes (hybrid, polypharmacy) are expected "
        "to cost more; projection assumes Phase B mode mix is close to the "
        "2-query calibration mix."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
