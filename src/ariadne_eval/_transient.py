"""Transient-error retry primitives shared across LLM-calling components.

Provider APIs (Anthropic, OpenAI, Groq, etc.) occasionally return HTTP
5xx / rate-limit / connection errors on otherwise-valid calls. Bounded
exponential backoff handles those without polluting reports or aborting
long runs. Identification is by class name (provider-portable across
litellm exception types).
"""

from __future__ import annotations

__all__ = [
    "MAX_TRANSIENT_RETRIES",
    "TRANSIENT_BACKOFF_BASE",
    "TRANSIENT_EXC_NAMES",
    "is_transient",
]


MAX_TRANSIENT_RETRIES: int = 4
TRANSIENT_BACKOFF_BASE: float = 2.0  # seconds; doubles per attempt: 2, 4, 8, 16
TRANSIENT_EXC_NAMES: tuple[str, ...] = (
    "InternalServerError",
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "ServiceUnavailableError",
)


def is_transient(exc: BaseException) -> bool:
    """Return True if ``exc`` is a known provider-side transient error.

    Identifies by exact class name (no inheritance walk) so the contract
    is stable across litellm / openai / anthropic exception hierarchies.
    """
    return type(exc).__name__ in TRANSIENT_EXC_NAMES
