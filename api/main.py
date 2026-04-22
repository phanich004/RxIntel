"""FastAPI service layer for RxIntel.

Endpoints:

* ``GET /health``  — liveness probe used by the Streamlit UI and the
  docker-compose healthcheck. Cheap — no graph calls.
* ``POST /ask``    — primary entry. Wraps ``agent.graph.run_graph`` with
  a 30s hard timeout, a 10/minute per-IP rate limit, and an optional
  demo-mode short-circuit that returns pre-seeded results for the
  demo queries so the live Groq/Neo4j/Chroma path isn't on the demo
  critical path.

Design notes:

* ``run_graph`` is sync (Groq / Neo4j / Chroma SDKs are sync). Running
  it on the FastAPI event loop would block every other request, so we
  dispatch it to a worker thread via ``asyncio.to_thread`` and wrap
  the whole thing in ``asyncio.wait_for`` for the timeout.
* The graph is eagerly compiled on the lifespan startup. Compilation
  is milliseconds, but the side-effect of importing the nodes (loading
  the 140k-entry gazetteer, warming the Groq/Chroma clients) surfaces
  any wiring error at container-healthcheck time instead of on the
  first user request.
* Unhandled exceptions are caught by a global handler that logs the
  full traceback server-side and returns a sanitized 500 body — no
  stack traces escape the API surface.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.exceptions import HTTPException as StarletteHTTPException

from agent.graph import _get_graph, run_graph
from api.models import AskRequest, AskResponse, ErrorResponse, HealthResponse

logger = logging.getLogger(__name__)

VERSION = os.environ.get("VERSION", "0.1")
ASK_TIMEOUT_SECONDS = float(os.environ.get("ASK_TIMEOUT_SECONDS", "30"))

# Demo-mode plumbing. DEMO_MODE is opt-in and is left off by default in
# docker-compose — a real demo operator flips it via .env after running
# scripts/seed_demo.py. The 3.5s sleep mimics typical pipeline latency
# so screenshots / screen recordings look natural.
DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() in {"1", "true", "yes"}
DEMO_RESPONSES_PATH = Path(
    os.environ.get("DEMO_RESPONSES_PATH", "/app/demo_responses.json")
)
DEMO_SIMULATED_LATENCY_S = 3.5

# Only the Streamlit container needs to hit this API. External access
# goes through the host port mapping, not CORS. If you later front this
# with a tunnel or a hosted UI, widen this allow-list explicitly.
_CORS_ALLOW_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://streamlit:8501",
]


def _load_demo_responses() -> dict[str, dict[str, Any]]:
    """Return the seeded demo-response map; empty dict if not present.

    Keys are the exact query strings that should short-circuit. Values
    are full ``run_graph`` return envelopes. Missing / malformed files
    are logged and treated as empty — demo mode simply falls through
    to the live pipeline for unseeded queries.
    """
    if not DEMO_RESPONSES_PATH.exists():
        return {}
    try:
        raw = json.loads(DEMO_RESPONSES_PATH.read_text())
        if not isinstance(raw, dict):
            logger.warning(
                "demo_responses.json is not a dict; ignoring (got %s)", type(raw)
            )
            return {}
        return cast(dict[str, dict[str, Any]], raw)
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", DEMO_RESPONSES_PATH, exc)
        return {}


_DEMO_CACHE: dict[str, dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Eagerly compile the LangGraph pipeline and cache demo responses."""
    logger.info("Compiling AGENT_GRAPH on startup (eager)…")
    _get_graph()
    if DEMO_MODE:
        global _DEMO_CACHE
        _DEMO_CACHE = _load_demo_responses()
        logger.info(
            "DEMO_MODE=on; %d seeded responses loaded from %s",
            len(_DEMO_CACHE),
            DEMO_RESPONSES_PATH,
        )
    yield


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="RxIntel", version=VERSION, lifespan=lifespan)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ALLOW_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content=ErrorResponse(
            error="rate_limited",
            message=f"Too many requests: {exc.detail}. Try again shortly.",
        ).model_dump(),
    )


@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Reshape FastAPI HTTPExceptions into our ErrorResponse envelope."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="http_error", message=str(exc.detail)
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for pipeline errors — logs full trace, returns sanitized 500."""
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            message="Pipeline failed. Check server logs for details.",
        ).model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=VERSION)


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
@limiter.limit("10/minute")
async def ask(request: Request, body: AskRequest) -> AskResponse:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    if DEMO_MODE and query in _DEMO_CACHE:
        logger.info("DEMO_MODE short-circuit for query: %r", query)
        await asyncio.sleep(DEMO_SIMULATED_LATENCY_S)
        return AskResponse.model_validate(_DEMO_CACHE[query])

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(run_graph, query, body.thread_id),
            timeout=ASK_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"pipeline exceeded {ASK_TIMEOUT_SECONDS:.0f}s timeout",
        ) from exc

    return AskResponse.model_validate(result)
