"""Fail-loud accessor for credential env vars.

Credentials must never silently fall back to factory defaults. Neo4j
ships with ``neo4j/neo4j`` and a default bolt URI; if our ``.env`` is
broken, we want a loud import-time failure rather than a confusing
authentication error deep inside a driver call.

Non-credential env vars (model ids, persist dirs, retry counts) keep
using ``os.environ.get(..., default)`` directly — they have sensible
fallbacks and are not sensitive.
"""

from __future__ import annotations

import os


def _require_env(name: str) -> str:
    """Return the env var, or raise RuntimeError with setup guidance."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and "
            f"fill in your credentials."
        )
    return value
