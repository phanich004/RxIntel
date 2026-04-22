"""Small Streamlit components for the RxIntel result screen.

Each function either renders directly (st.* calls) or returns an HTML
string meant for ``st.markdown(..., unsafe_allow_html=True)``. The
split is because badges are inline HTML but expanders / dataframes
need the native Streamlit API.
"""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from ui.styles import MODE_COLORS, SEVERITY_COLORS


def mode_badge(mode: str | None) -> None:
    if not mode:
        return
    color = MODE_COLORS.get(mode, "#475569")
    st.markdown(
        f'<span class="rxi-badge" style="background-color:{color}">'
        f"{html.escape(mode)}</span>",
        unsafe_allow_html=True,
    )


def severity_badge(severity: str | None) -> None:
    """Render a colored severity pill.

    ``n/a`` is a deliberate non-render: in ``describe`` / ``alternatives``
    / ``hybrid`` modes the severity concept doesn't apply, and showing
    an "n/a" pill adds visual noise without meaning.
    """
    if not severity or severity == "n/a":
        return
    color = SEVERITY_COLORS.get(severity, "#475569")
    if not color:
        return
    st.markdown(
        f'<span class="rxi-badge" style="background-color:{color}">'
        f"severity: {html.escape(severity)}</span>",
        unsafe_allow_html=True,
    )


def escalation_banner() -> None:
    st.markdown(
        '<div class="rxi-escalation-banner">'
        "⚠ Escalated to human review — the critic could not verify this "
        "response after 3 attempts. See the Critic scores section below "
        "for flagged issues."
        "</div>",
        unsafe_allow_html=True,
    )


def source_link(drug_id: str) -> str:
    """Return an anchor HTML string for a DrugBank ID."""
    safe = html.escape(drug_id)
    url = f"https://go.drugbank.com/drugs/{safe}"
    return (
        f'<a class="rxi-source-link" href="{url}" target="_blank" '
        f'rel="noopener noreferrer">{safe}</a>'
    )


def render_sources(sources: list[str]) -> None:
    if not sources:
        st.markdown('<span class="rxi-muted">No sources cited.</span>',
                    unsafe_allow_html=True)
        return
    html_str = " ".join(source_link(s) for s in sources)
    st.markdown(html_str, unsafe_allow_html=True)


def render_pairs_or_candidates(final_output: dict[str, Any]) -> None:
    pairs = final_output.get("interacting_pairs") or []
    cands = final_output.get("candidates") or []

    if pairs:
        st.markdown("**Interacting pairs**")
        for p in pairs:
            st.json(p, expanded=False)
    if cands:
        st.markdown("**Candidate drugs**")
        for c in cands:
            st.json(c, expanded=False)
    if not pairs and not cands:
        st.markdown('<span class="rxi-muted">None.</span>',
                    unsafe_allow_html=True)


def render_fused_results(fused: list[dict[str, Any]]) -> None:
    if not fused:
        st.markdown('<span class="rxi-muted">No retrieved evidence.</span>',
                    unsafe_allow_html=True)
        return
    # Cap at 20 rows to keep the UI from becoming a wall of JSON.
    for row in fused[:20]:
        drug_id = row.get("drug_id", "?")
        score = row.get("score", 0.0)
        st.markdown(f"**{html.escape(str(drug_id))}** · rrf={score:.4f}")
        st.json(row, expanded=False)
    if len(fused) > 20:
        st.caption(f"… {len(fused) - 20} more rows truncated")


def render_critic(critic: dict[str, Any]) -> None:
    composite = critic.get("composite", 0.0)
    approved = critic.get("approved", False)
    st.metric(
        "Composite",
        f"{composite:.2f}",
        delta="approved" if approved else "rejected",
        delta_color="normal" if approved else "inverse",
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Factual accuracy", f"{critic.get('factual_accuracy', 0.0):.2f}")
    col2.metric("Safety", f"{critic.get('safety_score', 0.0):.2f}")
    col3.metric("Completeness", f"{critic.get('completeness_score', 0.0):.2f}")

    issues = critic.get("issues") or []
    if issues:
        st.markdown("**Issues flagged**")
        for issue in issues:
            st.markdown(f"- {html.escape(str(issue))}")

    revision = critic.get("revision_prompt") or ""
    if revision:
        with st.expander("Revision prompt the critic proposed"):
            st.code(revision, language="text")
