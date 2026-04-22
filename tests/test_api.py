"""Offline tests for the FastAPI service.

We mock ``agent.graph.run_graph`` at the ``api.main`` import site so the
tests never touch Neo4j / Chroma / Groq. TestClient exercises the real
FastAPI routing, middleware, and error handlers.

Each test builds its own app via the FastAPI test pattern (import the
module after patches are in place so the lifespan doesn't compile the
real graph); we still patch ``_get_graph`` to a no-op to be defensive.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api import main as api_main
from api.main import app


@pytest.fixture(autouse=True)
def _noop_lifespan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip eager graph compile during tests."""
    monkeypatch.setattr(api_main, "_get_graph", lambda: None)


def _sample_result() -> dict[str, Any]:
    return {
        "mode": "ddi_check",
        "final_output": {
            "mode": "ddi_check",
            "severity": "Major",
            "interacting_pairs": [
                {"drug_a": "Warfarin", "drug_b": "Aspirin", "severity": "Major"}
            ],
            "candidates": [],
            "summary": "",
            "mechanism": "additive anticoagulant effect",
            "recommendation": "avoid co-administration",
            "confidence": 0.9,
            "insufficient_evidence": False,
            "sources": ["DB00682", "DB00945"],
            "escalated": False,
        },
        "critic_score": {
            "approved": True,
            "factual_accuracy": 1.0,
            "safety_score": 1.0,
            "completeness_score": 1.0,
            "composite": 1.0,
            "issues": [],
            "revision_prompt": "",
        },
        "retry_count": 0,
        "escalated": False,
        "resolved_drugs": [{"drug_id": "DB00682"}, {"drug_id": "DB00945"}],
        "fused_results": [],
    }


def test_health_returns_ok() -> None:
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_ask_happy_path_returns_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_main, "run_graph", lambda q, tid: _sample_result())
    # Disable demo-mode branch to force the real path.
    monkeypatch.setattr(api_main, "DEMO_MODE", False)

    with TestClient(app) as client:
        r = client.post("/ask", json={"query": "Is warfarin safe with aspirin?"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "ddi_check"
    assert body["final_output"]["severity"] == "Major"
    assert body["critic_score"]["approved"] is True
    assert body["retry_count"] == 0
    assert body["escalated"] is False


def test_ask_empty_query_returns_400(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_main, "run_graph", lambda q, tid: _sample_result())
    monkeypatch.setattr(api_main, "DEMO_MODE", False)

    with TestClient(app) as client:
        r = client.post("/ask", json={"query": "   "})
    # Pydantic passes (min_length=1 matches 3 spaces); our strip() catches it.
    assert r.status_code == 400
    assert r.json()["error"] == "http_error"


def test_ask_pydantic_validation_fires_on_missing_query() -> None:
    with TestClient(app) as client:
        r = client.post("/ask", json={})
    assert r.status_code == 422  # FastAPI's unprocessable-entity


def test_ask_pipeline_error_returns_sanitized_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(q: str, tid: str | None) -> dict[str, Any]:
        raise RuntimeError("neo4j exploded; SECRET=abc123")

    monkeypatch.setattr(api_main, "run_graph", _boom)
    monkeypatch.setattr(api_main, "DEMO_MODE", False)

    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.post("/ask", json={"query": "anything"})
    assert r.status_code == 500
    body = r.json()
    assert body["error"] == "internal_error"
    # Internal error details must not leak the exception message.
    assert "SECRET" not in body["message"]
    assert "neo4j exploded" not in body["message"]


def test_ask_timeout_returns_504(monkeypatch: pytest.MonkeyPatch) -> None:
    def _slow(q: str, tid: str | None) -> dict[str, Any]:
        import time as _t

        _t.sleep(2.0)
        return _sample_result()

    monkeypatch.setattr(api_main, "run_graph", _slow)
    monkeypatch.setattr(api_main, "DEMO_MODE", False)
    # Cut the timeout to 0.2s so this test runs fast.
    monkeypatch.setattr(api_main, "ASK_TIMEOUT_SECONDS", 0.2)

    with TestClient(app) as client:
        r = client.post("/ask", json={"query": "anything"})
    assert r.status_code == 504
    assert r.json()["error"] == "http_error"


def test_ask_demo_mode_short_circuits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Patch the loader so the lifespan populates _DEMO_CACHE with our
    # canned map. Setting _DEMO_CACHE directly would be overwritten
    # when the TestClient enters and triggers lifespan startup.
    canned = _sample_result()
    monkeypatch.setattr(api_main, "DEMO_MODE", True)
    monkeypatch.setattr(
        api_main,
        "_load_demo_responses",
        lambda: {"Is warfarin safe with aspirin?": canned},
    )
    monkeypatch.setattr(api_main, "DEMO_SIMULATED_LATENCY_S", 0.0)

    def _should_not_run(q: str, tid: str | None) -> dict[str, Any]:
        raise AssertionError("run_graph must not be called in DEMO_MODE short-circuit")

    monkeypatch.setattr(api_main, "run_graph", _should_not_run)

    with TestClient(app) as client:
        r = client.post("/ask", json={"query": "Is warfarin safe with aspirin?"})
    assert r.status_code == 200
    assert r.json()["final_output"]["severity"] == "Major"


def test_ask_demo_mode_falls_through_for_unseeded_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_main, "DEMO_MODE", True)
    monkeypatch.setattr(
        api_main, "_load_demo_responses", lambda: {"seeded-only": _sample_result()}
    )
    monkeypatch.setattr(api_main, "run_graph", lambda q, tid: _sample_result())

    with TestClient(app) as client:
        r = client.post("/ask", json={"query": "not seeded"})
    assert r.status_code == 200
    assert r.json()["mode"] == "ddi_check"


def test_load_demo_responses_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(api_main, "DEMO_RESPONSES_PATH", tmp_path / "nope.json")
    assert api_main._load_demo_responses() == {}


def test_load_demo_responses_malformed_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[1,2,3]")  # list, not dict
    monkeypatch.setattr(api_main, "DEMO_RESPONSES_PATH", bad)
    assert api_main._load_demo_responses() == {}


def test_rate_limit_triggers_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_main, "run_graph", lambda q, tid: _sample_result())
    monkeypatch.setattr(api_main, "DEMO_MODE", False)
    # Reset the in-memory limiter counters so prior tests don't poison us.
    api_main.limiter.reset()

    with TestClient(app) as client:
        # 10/minute is the declared limit; the 11th call must be rejected.
        for _ in range(10):
            assert client.post("/ask", json={"query": "q"}).status_code == 200
        r = client.post("/ask", json={"query": "q"})
    assert r.status_code == 429
    assert r.json()["error"] == "rate_limited"
    api_main.limiter.reset()
