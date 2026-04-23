"""Phase B benchmark runner over ``evaluation/queries.yaml``.

Runs each query through ``agent.graph.run_graph`` and records per-query
metrics (mode, severity, retrieved drugs, critic score, retries,
latency, Anthropic token usage + cost). Writes a per-row
``evaluation/results.csv`` and a human-readable ``evaluation/summary.md``,
and logs one MLflow run to the ``rxintel-benchmark-v1`` experiment.

Hard cost ceiling: cumulative estimated spend is tracked as queries
complete and the run is aborted cleanly if it crosses $5.00. Partial
results are still written so a post-mortem is possible.

Usage
-----
    python scripts/run_benchmark.py            # run the full set
    python scripts/run_benchmark.py --dry-run  # parse yaml + wire the
                                                # runner, no API calls

Metrics
-------
    routing_accuracy     -- frac(predicted_mode == expected_mode)
    severity_f1          -- macro f1 across severities on the
                            ddi_check + polypharmacy subset only
    drug_recall_at_5     -- avg fraction of expected_drugs appearing in
                            the top-5 RRF-ranked drug_ids per query
    critic_approval_rate -- frac(approved AND NOT escalated)
    latency_p50 / p95    -- percentiles of per-query elapsed seconds
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from agent import token_tracker  # noqa: E402

# Anthropic Sonnet pricing per Step 17 brief (USD per 1M tokens).
INPUT_PRICE_PER_M = 3.00
OUTPUT_PRICE_PER_M = 15.00
COST_CEILING_USD = 5.00

QUERIES_PATH = ROOT / "evaluation" / "queries.yaml"
RESULTS_CSV = ROOT / "evaluation" / "results.csv"
SUMMARY_MD = ROOT / "evaluation" / "summary.md"

MLFLOW_EXPERIMENT = "rxintel-benchmark-v1"

CSV_COLUMNS = [
    "id",
    "mode_expected",
    "mode_actual",
    "severity_expected",
    "severity_actual",
    "severity_pass",
    "expected_drugs",
    "retrieved_top5",
    "drug_recall",
    "critic_composite",
    "approved",
    "escalated",
    "retries",
    "elapsed_s",
    "reasoning_in",
    "reasoning_out",
    "critic_in",
    "critic_out",
    "cost_usd",
    "cum_cost_usd",
]


def _cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * INPUT_PRICE_PER_M + output_tokens * OUTPUT_PRICE_PER_M
    ) / 1_000_000


def _top5_drug_ids(fused: list[dict[str, Any]]) -> list[str]:
    """``fused_results`` is already RRF-sorted; take the first 5 drug_ids."""
    ids: list[str] = []
    for rec in fused:
        did = rec.get("drug_id")
        if did and did not in ids:
            ids.append(did)
        if len(ids) == 5:
            break
    return ids


def _expected_set_for_recall(spec: dict[str, Any]) -> list[str]:
    """Drug IDs the top-K retrieval should contain, chosen per mode.

    ddi_check / describe / polypharmacy → ``expected_drugs``
    alternatives / hybrid                → ``expected_candidates``

    Alternatives and hybrid intentionally surface DIFFERENT drugs than
    the ones named in the question, so scoring them against
    ``expected_drugs`` (which tends to hold input drugs for traceability)
    produces a systematic 0% recall.
    """
    mode = spec.get("mode")
    if mode in ("alternatives", "hybrid"):
        candidates = spec.get("expected_candidates")
        return list(candidates) if candidates else []
    return list(spec.get("expected_drugs") or [])


def _severity_match(predicted: Any, expected: Any) -> bool:
    """Pass/fail for severity — expected may be None, str, or list of str."""
    if expected is None:
        return True  # mode has no severity concept
    pred = str(predicted).lower() if predicted is not None else ""
    if isinstance(expected, str):
        return pred == expected.lower()
    if isinstance(expected, list):
        return pred in [str(e).lower() for e in expected]
    return False


def _severity_canonical(expected: Any) -> str | None:
    """First-choice label for sklearn f1_score.

    When ``expected`` is a list, the first entry is the preferred label
    — sklearn sees only that one, so a prediction of a non-first
    acceptable label counts as a mismatch for F1 even though pass/fail
    passes. This gives F1 a stricter "hit the top-choice label" signal.
    """
    if expected is None:
        return None
    if isinstance(expected, str):
        return expected.lower()
    if isinstance(expected, list) and expected:
        return str(expected[0]).lower()
    return None


def _sum_by_agent(
    records: list[dict[str, Any]],
) -> dict[str, tuple[int, int, int]]:
    agg: dict[str, tuple[int, int, int]] = {}
    for rec in records:
        calls, i, o = agg.get(rec["agent"], (0, 0, 0))
        agg[rec["agent"]] = (
            calls + 1,
            i + int(rec.get("input_tokens", 0)),
            o + int(rec.get("output_tokens", 0)),
        )
    return agg


def _load_queries() -> list[dict[str, Any]]:
    with QUERIES_PATH.open() as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, list):
        raise ValueError(
            f"{QUERIES_PATH} must be a top-level YAML list of query specs"
        )
    required = {"id", "mode", "query", "expected_mode"}
    for i, spec in enumerate(loaded):
        missing = required - set(spec or {})
        if missing:
            raise ValueError(
                f"query {i} ({spec.get('id', '<?>')}) missing fields: "
                f"{sorted(missing)}"
            )
    return loaded


def _run_one(spec: dict[str, Any], cum_cost: float) -> dict[str, Any]:
    """Execute one query; return a CSV-ready row."""
    from agent.graph import run_graph  # lazy to keep --dry-run import-free

    qid = spec["id"]
    token_tracker.reset()
    t0 = time.perf_counter()
    result = run_graph(spec["query"])
    elapsed = time.perf_counter() - t0

    usage = _sum_by_agent(token_tracker.records())
    _, r_in, r_out = usage.get("reasoning", (0, 0, 0))
    _, c_in, c_out = usage.get("critic", (0, 0, 0))
    cost = _cost_usd(r_in + c_in, r_out + c_out)

    final = result.get("final_output") or {}
    critic = result.get("critic_score") or {}
    fused = result.get("fused_results") or []
    top5 = _top5_drug_ids(fused)

    expected_set = _expected_set_for_recall(spec)
    if expected_set:
        hits = sum(1 for d in expected_set if d in top5)
        drug_recall = hits / len(expected_set)
    else:
        drug_recall = 1.0  # no expected set for this query (e.g. hybrid with no candidates)

    sev_expected_raw = spec.get("expected_severity")
    sev_actual = final.get("severity")
    sev_pass = _severity_match(sev_actual, sev_expected_raw)

    return {
        "id": qid,
        "mode_expected": spec.get("expected_mode"),
        "mode_actual": result.get("mode"),
        "severity_expected": json.dumps(sev_expected_raw) if sev_expected_raw is not None else "",
        "severity_actual": sev_actual,
        "severity_pass": sev_pass,
        "expected_drugs": json.dumps(expected_set),
        "retrieved_top5": json.dumps(top5),
        "drug_recall": round(drug_recall, 4),
        "critic_composite": round(float(critic.get("composite", 0.0) or 0.0), 4),
        "approved": bool(critic.get("approved", False)),
        "escalated": bool(result.get("escalated", False)),
        "retries": int(result.get("retry_count", 0) or 0),
        "elapsed_s": round(elapsed, 3),
        "reasoning_in": r_in,
        "reasoning_out": r_out,
        "critic_in": c_in,
        "critic_out": c_out,
        "cost_usd": round(cost, 6),
        "cum_cost_usd": round(cum_cost + cost, 6),
    }


def _compute_metrics(
    rows: list[dict[str, Any]], specs: list[dict[str, Any]]
) -> dict[str, float]:
    """Five aggregate metrics per brief.

    ``specs`` is the raw list of query dicts parallel to ``rows``. It's
    passed separately because ``rows`` (CSV-shaped) serializes
    ``expected_severity`` to JSON for the list case, so the canonical
    label for F1 has to come from the raw spec.
    """
    from sklearn.metrics import f1_score

    total = len(rows)
    if total == 0:
        return {}

    routing = sum(1 for r in rows if r["mode_expected"] == r["mode_actual"]) / total

    # severity_f1 — macro, only on ddi_check + polypharmacy where both labels exist.
    # When expected is a list, ``_severity_canonical`` picks its first entry
    # as the strict-match label seen by sklearn.
    sev_modes = {"ddi_check", "polypharmacy"}
    y_true: list[str] = []
    y_pred: list[str] = []
    for row, spec in zip(rows, specs, strict=True):
        if row["mode_expected"] not in sev_modes:
            continue
        canonical = _severity_canonical(spec.get("expected_severity"))
        pred = row["severity_actual"]
        if canonical is None or pred is None:
            continue
        y_true.append(canonical)
        y_pred.append(str(pred).lower())

    if y_true:
        severity_f1 = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
    else:
        severity_f1 = 0.0

    drug_recall = sum(r["drug_recall"] for r in rows) / total
    critic_approval = (
        sum(1 for r in rows if r["approved"] and not r["escalated"]) / total
    )

    latencies = sorted(r["elapsed_s"] for r in rows)
    p50 = statistics.median(latencies)
    # p95 by nearest-rank (matches our ops convention, no interpolation)
    p95 = latencies[min(len(latencies) - 1, max(0, int(round(0.95 * len(latencies))) - 1))]

    return {
        "routing_accuracy": round(routing, 4),
        "severity_f1_ddi_poly": round(severity_f1, 4),
        "drug_recall_at_5": round(drug_recall, 4),
        "critic_approval_rate": round(critic_approval, 4),
        "latency_p50_s": round(p50, 3),
        "latency_p95_s": round(p95, 3),
        "total_queries": float(total),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
    }


def _write_csv(rows: list[dict[str, Any]]) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_summary(
    rows: list[dict[str, Any]],
    metrics: dict[str, float],
    aborted: bool,
    aborted_at: int,
) -> None:
    lines: list[str] = ["# RxIntel benchmark — rxintel-benchmark-v1", ""]
    if aborted:
        lines.append(
            f"> **Aborted** at query {aborted_at}/{aborted_at} after "
            f"crossing ${COST_CEILING_USD:.2f} cost ceiling."
        )
        lines.append("")

    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | ---: |")
    for k in (
        "total_queries",
        "routing_accuracy",
        "severity_f1_ddi_poly",
        "drug_recall_at_5",
        "critic_approval_rate",
        "latency_p50_s",
        "latency_p95_s",
        "total_cost_usd",
    ):
        if k in metrics:
            lines.append(f"| {k} | {metrics[k]} |")
    lines.append("")

    lines.append("## Per-query results")
    lines.append("")
    lines.append(
        "| id | mode (exp→act) | sev (exp→act / pass) | recall | "
        "critic | retries | elapsed | $ |"
    )
    lines.append(
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"
    )
    for r in rows:
        sev_exp_raw = r["severity_expected"] or "(n/a)"
        try:
            parsed = json.loads(sev_exp_raw) if sev_exp_raw != "(n/a)" else None
        except (json.JSONDecodeError, TypeError):
            parsed = sev_exp_raw
        if isinstance(parsed, list):
            sev_exp_disp = "{" + ",".join(parsed) + "}"
        elif parsed is None:
            sev_exp_disp = "—"
        else:
            sev_exp_disp = str(parsed)
        sev_pass = "✓" if r["severity_pass"] else "✗"
        lines.append(
            f"| {r['id']} | {r['mode_expected']}→{r['mode_actual']} "
            f"| {sev_exp_disp}→{r['severity_actual']} {sev_pass} "
            f"| {r['drug_recall']:.2f} | {r['critic_composite']:.2f}"
            f"{'' if not r['escalated'] else ' ESC'} "
            f"| {r['retries']} | {r['elapsed_s']:.1f}s "
            f"| ${r['cost_usd']:.4f} |"
        )

    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def _mlflow_log(
    rows: list[dict[str, Any]], metrics: dict[str, float]
) -> None:
    """Log metrics + artifacts under rxintel-benchmark-v1. Silent on failure."""
    try:
        import mlflow

        mlflow.set_tracking_uri(
            os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
        )
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="phase-b-50q"):
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))
            mlflow.log_artifact(str(RESULTS_CSV))
            mlflow.log_artifact(str(SUMMARY_MD))
    except Exception as exc:  # noqa: BLE001 — logging is best-effort
        print(f"WARNING: MLflow logging failed: {exc}", file=sys.stderr)


def _run(dry_run: bool) -> int:
    queries = _load_queries()
    print(f"Loaded {len(queries)} queries from {QUERIES_PATH}")
    if dry_run:
        print("[--dry-run] skipping API calls; validating schema + loader only")
        # Touch sklearn import so a missing dep fails dry-run too.
        from sklearn.metrics import f1_score  # noqa: F401

        # Smoke-test the lazy agent import so we catch import errors early.
        from agent.graph import run_graph  # noqa: F401

        print("[--dry-run] imports OK, yaml OK — runner ready.")
        return 0

    if not os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-ant"):
        print(
            "ANTHROPIC_API_KEY is not set (expected sk-ant-...). Aborting.",
            file=sys.stderr,
        )
        return 2

    rows: list[dict[str, Any]] = []
    cum_cost = 0.0
    aborted = False

    for i, spec in enumerate(queries, start=1):
        row = _run_one(spec, cum_cost)
        cum_cost = row["cum_cost_usd"]
        rows.append(row)

        print(
            f"[{i:3d}/{len(queries)}] {row['id']:10s} "
            f"mode={row['mode_actual']:12s} "
            f"recall={row['drug_recall']:.2f} "
            f"critic={row['critic_composite']:.2f} "
            f"{'ESC' if row['escalated'] else 'ok ':3s} "
            f"t={row['elapsed_s']:6.2f}s "
            f"${row['cost_usd']:.4f}  cum=${cum_cost:.4f}"
        )

        if i % 10 == 0:
            print(f"   -- cumulative spend after {i} queries: ${cum_cost:.4f}")

        if cum_cost >= COST_CEILING_USD:
            print(
                f"HIT COST CEILING at query {i}/{len(queries)}. "
                f"Partial results in {RESULTS_CSV}."
            )
            aborted = True
            break

    _write_csv(rows)
    metrics = _compute_metrics(rows, queries[: len(rows)])
    _write_summary(rows, metrics, aborted, len(rows))
    _mlflow_log(rows, metrics)

    print()
    print("=" * 78)
    print("FINAL METRICS")
    for k, v in metrics.items():
        print(f"  {k:24s} = {v}")
    print(f"Results:  {RESULTS_CSV}")
    print(f"Summary:  {SUMMARY_MD}")
    return 1 if aborted else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate yaml + imports; do not hit the API.",
    )
    args = parser.parse_args(argv)
    return _run(args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
