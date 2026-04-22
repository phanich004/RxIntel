"""RxIntel clinician UI — single-page Streamlit front end.

Layout follows progressive disclosure:

1. Input: query textarea + Ask button + 4 one-click example queries.
2. On submit: POST to the API, show a spinner while the pipeline runs.
3. Result screen:
   * Escalation banner (if escalated) or mode/severity badges.
   * Recommendation — the headline a clinician reads first.
   * Copy-JSON download button under the recommendation.
   * Expanders (collapsed by default): Mechanism, Pairs/Candidates,
     Sources, Raw evidence, Critic scores.
   * Footer: latency + retry count + approval status.

``API_URL`` is injected by docker-compose in the Streamlit container;
for local development without compose it falls back to localhost.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests
import streamlit as st

from ui.components import (
    escalation_banner,
    mode_badge,
    render_critic,
    render_fused_results,
    render_pairs_or_candidates,
    render_sources,
    severity_badge,
)
from ui.styles import inject_css

API_URL = os.environ.get("API_URL", "http://localhost:8000")
ASK_HTTP_TIMEOUT_S = 35.0  # a touch above the API's 30s hard cap.

# Must match scripts/seed_demo.py's DEMO_QUERIES so demo-mode hits.
EXAMPLE_QUERIES: dict[str, str] = {
    "Interaction check": "Is warfarin safe with aspirin?",
    "Find an alternative": (
        "Alternatives to ibuprofen for a patient with chronic kidney disease"
    ),
    "Describe a drug": "What is semaglutide and how does it work?",
    "Pharmacology + class": (
        "Which CYP3A4 inhibitors interact with atorvastatin?"
    ),
}


def _post_ask(query: str) -> dict[str, Any]:
    """POST to /ask and return the parsed response dict.

    Raises RuntimeError with a user-friendly message on any non-200.
    The caller displays the message verbatim — do not leak server
    internals from here.
    """
    try:
        resp = requests.post(
            f"{API_URL}/ask",
            json={"query": query},
            timeout=ASK_HTTP_TIMEOUT_S,
        )
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(f"The API took longer than {ASK_HTTP_TIMEOUT_S:.0f}s to respond.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not reach the API at {API_URL}. Is the container running?"
        ) from exc

    if resp.status_code == 429:
        raise RuntimeError("Rate limit exceeded — please wait a minute before retrying.")
    if resp.status_code == 504:
        raise RuntimeError("The pipeline timed out on the server side. Try a simpler query.")
    if resp.status_code >= 400:
        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        raise RuntimeError(body.get("message", f"API error (status {resp.status_code})"))

    return resp.json()  # type: ignore[no-any-return]


def _render_result(result: dict[str, Any], elapsed: float) -> None:
    escalated = bool(result.get("escalated", False))
    final_output = result.get("final_output") or {}
    critic = result.get("critic_score") or {}
    mode = result.get("mode")
    severity = final_output.get("severity")

    # ---------- Row 1: escalation banner OR mode/severity badges ----------
    if escalated:
        escalation_banner()
    else:
        col_mode, col_sev, _spacer = st.columns([1, 1, 4])
        with col_mode:
            mode_badge(mode)
        with col_sev:
            severity_badge(severity)

    # ---------- Recommendation (headline) ----------
    st.subheader("Recommendation")
    rec = final_output.get("recommendation") or "(no recommendation returned)"
    st.write(rec)

    # Copy-JSON button — serializes the full response envelope so the
    # clinician / reviewer can paste it into a ticket or log.
    st.download_button(
        label="Copy JSON",
        data=json.dumps(result, indent=2, default=str),
        file_name="rxintel_response.json",
        mime="application/json",
    )

    # ---------- Mechanism ----------
    with st.expander("Mechanism"):
        mech = final_output.get("mechanism") or "(no mechanism provided)"
        st.write(mech)

    # ---------- Pairs or candidates ----------
    with st.expander("Interacting pairs / Candidates"):
        render_pairs_or_candidates(final_output)

    # ---------- Sources ----------
    with st.expander("Sources"):
        sources = final_output.get("sources") or []
        render_sources(sources)

    # ---------- Raw fused evidence ----------
    with st.expander("Raw evidence"):
        render_fused_results(result.get("fused_results") or [])

    # ---------- Critic scores — expanded when escalated ----------
    with st.expander("Critic scores", expanded=escalated):
        render_critic(critic)

    # ---------- Footer ----------
    retry_count = int(result.get("retry_count", 0) or 0)
    approved = bool(critic.get("approved", False))
    status_word = "Approved" if approved else ("Escalated" if escalated else "Rejected")
    st.caption(
        f"Answered in {elapsed:.1f}s · {retry_count} retry"
        f"{'s' if retry_count != 1 else ''} · {status_word}"
    )


def main() -> None:
    st.set_page_config(
        page_title="RxIntel — Drug Intelligence",
        layout="wide",
        page_icon="💊",
    )
    inject_css()

    st.title("RxIntel — Drug Intelligence")
    st.caption(
        "Ask a clinical question. The pipeline routes across a DrugBank "
        "knowledge graph and vector store, then a critic audits the answer "
        "before it reaches you."
    )

    # Seed initial state.
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = ""

    query = st.text_area(
        "Your question",
        value=st.session_state.pending_query,
        placeholder="Is warfarin safe with aspirin?",
        height=90,
        key="query_input",
    )

    ask_col, *ex_cols = st.columns([1, 1, 1, 1, 1])
    with ask_col:
        submit = st.button("Ask", type="primary", use_container_width=True)

    for (label, example_q), col in zip(EXAMPLE_QUERIES.items(), ex_cols):
        with col:
            if st.button(label, key=f"ex-{label}", use_container_width=True):
                st.session_state.pending_query = example_q
                st.session_state.auto_submit = True
                st.rerun()

    # After a rerun triggered by an example button, this flag is set.
    # pop() clears it so re-running main() after the result renders
    # doesn't auto-resubmit the same query.
    auto_submit = st.session_state.pop("auto_submit", False)
    should_run = (submit or auto_submit) and query.strip()

    if should_run:
        with st.spinner("Reasoning over evidence…"):
            t0 = time.perf_counter()
            try:
                result = _post_ask(query)
            except RuntimeError as exc:
                st.error(str(exc))
                return
            elapsed = time.perf_counter() - t0

        _render_result(result, elapsed)


if __name__ == "__main__":
    main()
