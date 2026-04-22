"""Seed the demo-mode response cache by running the 4 example queries.

Run once on the deploy host AFTER the services are up and healthy::

    python -m scripts.seed_demo

The script hits the live pipeline (so Neo4j / Chroma / Groq must be
available) and writes ``data/demo_responses.json`` — the path the API
container reads when ``DEMO_MODE=true``. The file itself is gitignored
because it's generated artifact data.

Pre-seeded demo responses decouple screenshot-quality UX from the live
path: the demo survives a Groq rate-limit or a Neo4j hiccup without
visible degradation, and every canned response was already critic-
approved when it was recorded.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from agent.graph import run_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("seed_demo")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "data" / "demo_responses.json"

# Matches the example buttons in ui/app.py. Keys here MUST match the
# exact query strings the UI sends so the API short-circuit fires.
DEMO_QUERIES: list[str] = [
    "Is warfarin safe with aspirin?",
    "Alternatives to ibuprofen for a patient with chronic kidney disease",
    "What is semaglutide and how does it work?",
    "Which CYP3A4 inhibitors interact with atorvastatin?",
]


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Ensure the file exists before anything else so the Dockerfile's
    # ``COPY data/demo_responses.json`` always has something to grab
    # — the image build must not require a live pipeline to succeed.
    if not OUTPUT_PATH.exists():
        OUTPUT_PATH.write_text("{}")
        logger.info("created placeholder %s (empty dict)", OUTPUT_PATH)

    seeded: dict[str, dict] = {}

    for i, query in enumerate(DEMO_QUERIES, start=1):
        logger.info("[%d/%d] running: %s", i, len(DEMO_QUERIES), query)
        try:
            result = run_graph(query)
        except Exception:
            logger.exception("query failed; skipping: %s", query)
            continue

        approved = (result.get("critic_score") or {}).get("approved")
        if not approved:
            logger.warning(
                "query did not approve (approved=%s); skipping seeding: %s",
                approved,
                query,
            )
            continue

        # Strip non-JSON-serializable bits defensively; run_graph's
        # return is already a plain dict but some state keys (e.g.
        # chroma chunk objects) may not be. We accept lossiness here
        # — the UI only needs final_output + critic_score + metadata.
        seeded[query] = json.loads(json.dumps(result, default=str))

    if not seeded:
        logger.error("no queries approved; not writing %s", OUTPUT_PATH)
        return 1

    OUTPUT_PATH.write_text(json.dumps(seeded, indent=2))
    logger.info("wrote %d seeded responses to %s", len(seeded), OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
