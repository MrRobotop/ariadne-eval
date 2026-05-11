"""Fail-mode policy for unattached tracing recordings.

When user code calls a recorder (``record_llm_call``, ``record_tool_call``,
``@trace_step``) without an active trajectory in context, the configured
``FailMode`` decides what happens:

- ``STRICT`` (default): raise ``RuntimeError`` so the bug surfaces loudly.
- ``WARN``: emit ``UnattachedTracingWarning`` exactly once per process,
  then no-op. The operation succeeds with no effect.
- ``SILENT``: no-op silently. Use in production when orphaned LLM calls
  must not halt the agent.

The mode is read from ``ARIADNE_FAIL_MODE`` on each call so test monkeypatch
+ reload patterns work; the cost of a single ``os.environ.get`` per recorder
call is negligible.
"""

from __future__ import annotations

import os
import warnings
from enum import StrEnum
from typing import Final

__all__ = [
    "FailMode",
    "UnattachedTracingWarning",
    "handle_unattached",
]


class FailMode(StrEnum):
    """Behaviour when a recording is attempted with no active trajectory."""

    STRICT = "strict"
    WARN = "warn"
    SILENT = "silent"


class UnattachedTracingWarning(UserWarning):
    """Emitted under ``FailMode.WARN`` when a recorder runs unattached."""


_ENV_VAR: Final[str] = "ARIADNE_FAIL_MODE"
_warned_once: bool = False


def _resolve_fail_mode() -> FailMode:
    """Read the fail mode from the environment, defaulting to STRICT."""
    raw = os.environ.get(_ENV_VAR)
    if raw is None:
        return FailMode.STRICT
    try:
        return FailMode(raw.lower())
    except ValueError as exc:
        raise ValueError(
            f"{_ENV_VAR}={raw!r} is invalid; expected one of {[m.value for m in FailMode]}"
        ) from exc


def handle_unattached(call_site: str) -> None:
    """Apply the configured fail-mode policy.

    ``call_site`` names the recorder that hit the unattached state (e.g.
    ``"record_llm_call"``) so error / warning messages are actionable.
    """
    global _warned_once
    mode = _resolve_fail_mode()
    if mode is FailMode.STRICT:
        raise RuntimeError(
            f"no active trajectory: {call_site} called outside "
            "start_trajectory(...). Set ARIADNE_FAIL_MODE=silent or 'warn' "
            "to opt out."
        )
    if mode is FailMode.WARN:
        if not _warned_once:
            warnings.warn(
                f"ariadne-eval: {call_site} called with no active trajectory; "
                "subsequent occurrences will be silent.",
                UnattachedTracingWarning,
                stacklevel=3,
            )
            _warned_once = True
        return
    # SILENT
    return
