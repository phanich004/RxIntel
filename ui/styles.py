"""Color tokens + CSS injection for the Streamlit UI.

The palette is centralized here so mode and severity badges share a
single source of truth. If you change a hex value, both badge
components and any future overlays pick it up on the next rerun.

Palette rationale:
* Mode colors are categorical, not ordinal — no red in the set, to
  leave red unambiguously tied to severity/escalation.
* Severity colors track clinical convention (red=danger,
  orange=caution, yellow=minor, gray=unknown).
"""

from __future__ import annotations

from typing import Final

import streamlit as st

MODE_COLORS: Final[dict[str, str]] = {
    "ddi_check": "#2563eb",      # blue
    "alternatives": "#16a34a",   # green
    "describe": "#7c3aed",       # purple
    "hybrid": "#ea580c",         # orange
    "polypharmacy": "#be123c",   # rose/red
}

SEVERITY_COLORS: Final[dict[str, str]] = {
    "Major": "#dc2626",          # red
    "Moderate": "#ea580c",       # orange
    "Minor": "#ca8a04",          # amber
    "unknown": "#6b7280",        # gray
    "n/a": "",                   # empty — badge is hidden for n/a
}

ESCALATION_RED: Final[str] = "#dc2626"
TEXT_ON_COLORED: Final[str] = "#ffffff"


_BADGE_CSS = """
<style>
.rxi-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
    color: var(--rxi-text-on-colored);
    letter-spacing: 0.02em;
    margin-right: 6px;
}
.rxi-escalation-banner {
    background-color: var(--rxi-escalation-red);
    color: var(--rxi-text-on-colored);
    padding: 12px 18px;
    border-radius: 8px;
    font-weight: 600;
    margin-bottom: 14px;
}
.rxi-source-link {
    display: inline-block;
    padding: 2px 10px;
    margin: 2px 4px 2px 0;
    border-radius: 6px;
    background-color: #f1f5f9;
    color: #0f172a;
    font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    font-size: 0.82rem;
    text-decoration: none;
    border: 1px solid #e2e8f0;
}
.rxi-source-link:hover { background-color: #e2e8f0; }
.rxi-muted { color: #64748b; font-size: 0.82rem; }
</style>
"""


def inject_css() -> None:
    """Inject the RxIntel CSS once per Streamlit rerun.

    The ``:root`` custom properties let badge colors be computed at
    render time from the Python-side palette without embedding each
    hex literal in every HTML string.
    """
    vars_block = (
        ":root {"
        f" --rxi-escalation-red: {ESCALATION_RED};"
        f" --rxi-text-on-colored: {TEXT_ON_COLORED};"
        " }"
    )
    st.markdown(
        f"<style>{vars_block}</style>{_BADGE_CSS}",
        unsafe_allow_html=True,
    )
