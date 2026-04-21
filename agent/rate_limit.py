"""Tenacity retry wrapper for Groq LLM calls.

Groq free tier is ~30 req/min. An end-to-end graph run can touch the LLM
up to ~8 times (router + 2 * 3 reasoning retries + final), so a single
429 must not abort the pipeline. We retry on:

  * ``httpx.HTTPStatusError`` with status 429
  * ``groq.RateLimitError`` when the groq SDK is installed
  * generic ``ConnectionError``

Exponential backoff 2s -> 30s, capped at 5 attempts. ``reraise=True`` so
the original exception surfaces after exhaustion instead of being wrapped
in a ``RetryError``.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

try:
    from groq import RateLimitError as _GroqRateLimit
except ImportError:  # pragma: no cover - groq is a transitive dep of langchain-groq
    _GroqRateLimit = None  # type: ignore[assignment,misc]


F = TypeVar("F", bound=Callable[..., Any])


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, ConnectionError):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        return True
    if _GroqRateLimit is not None and isinstance(exc, _GroqRateLimit):
        return True
    return False


groq_retry: Callable[[F], F] = retry(
    wait=wait_exponential(min=2, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
