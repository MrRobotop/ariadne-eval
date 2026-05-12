"""LiteLLM auto-trace adapter.

Registers a callback with LiteLLM so that every successful (and failed)
completion is recorded as an llm_call Step in the active trajectory.

Lazy-imports LiteLLM so ``import ariadne_eval`` does not pull it in.
"""

from __future__ import annotations

from typing import Any

from ariadne_eval.core.trajectory import Message
from ariadne_eval.tracing import _fail_mode
from ariadne_eval.tracing.context import current_trajectory
from ariadne_eval.tracing.decorator import record_llm_call

__all__ = ["enable_litellm_autotrace"]


_registered = False


def enable_litellm_autotrace() -> None:
    """Register the auto-trace callbacks with LiteLLM. Idempotent.

    LiteLLM has multiple callback registries. The unified ``callbacks``
    list handles both sync and async callbacks for both ``completion`` and
    ``acompletion`` — litellm dispatches based on callable shape. The
    legacy ``success_callback`` / ``failure_callback`` are sync-only;
    we still append to them so sync ``completion`` callers also get
    traces.
    """
    global _registered
    if _registered:
        return
    import litellm  # lazy

    # Newer litellm exposes a unified ``callbacks`` list that handles both
    # sync ``completion`` and async ``acompletion`` paths. Fall back gracefully
    # for older versions or stub modules in tests.
    callbacks = getattr(litellm, "callbacks", None)
    if callbacks is not None:
        if _on_success not in callbacks:
            callbacks.append(_on_success)
        if _on_failure not in callbacks:
            callbacks.append(_on_failure)

    if _on_success not in litellm.success_callback:
        litellm.success_callback.append(_on_success)
    if _on_failure not in litellm.failure_callback:
        litellm.failure_callback.append(_on_failure)

    _registered = True


def _messages_from_kwargs(kwargs: dict[str, Any]) -> list[Message]:
    raw = kwargs.get("messages", []) or []
    out: list[Message] = []
    for m in raw:
        if isinstance(m, Message):
            out.append(m)
            continue
        role = m.get("role", "user")
        content = m.get("content", "")
        out.append(Message(role=role, content=content))
    return out


def _cost_from_response(response: Any) -> float:
    try:
        import litellm

        cost = litellm.completion_cost(response)
        return float(cost) if cost is not None else 0.0
    except Exception:  # pragma: no cover - upstream model coverage varies
        return 0.0


def _latency_ms(start_time: Any, end_time: Any) -> float:
    """Compute latency in ms, tolerating both float and datetime inputs.

    LiteLLM passes ``datetime.datetime`` objects to its callbacks; their
    difference is a ``timedelta`` which would fail Pydantic validation on
    ``LLMCallPayload.latency_ms: float``. Plain numeric inputs go through
    the float multiplication path.
    """
    delta = end_time - start_time
    if hasattr(delta, "total_seconds"):
        return float(delta.total_seconds()) * 1000.0
    return float(delta) * 1000.0


async def _on_success(
    kwargs: dict[str, Any],
    response: Any,
    start_time: Any,
    end_time: Any,
) -> None:
    """LiteLLM success callback: record an llm_call Step."""
    traj = current_trajectory()
    if traj is None:
        _fail_mode.handle_unattached("litellm.success_callback")
        return
    if traj.is_noop:
        return

    try:
        completion = response.choices[0].message.content or ""
    except (AttributeError, IndexError):  # pragma: no cover - defensive
        completion = ""
    try:
        input_tokens = int(response.usage.prompt_tokens)
        output_tokens = int(response.usage.completion_tokens)
    except (AttributeError, TypeError):  # pragma: no cover - defensive
        input_tokens = 0
        output_tokens = 0

    await record_llm_call(
        model_id=str(kwargs.get("model", "unknown")),
        prompt_messages=_messages_from_kwargs(kwargs),
        completion=completion,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=_cost_from_response(response),
        temperature=kwargs.get("temperature"),
        latency_ms=_latency_ms(start_time, end_time),
    )


async def _on_failure(
    kwargs: dict[str, Any],
    response: Any,
    start_time: Any,
    end_time: Any,
) -> None:
    """LiteLLM failure callback: record a failed llm_call Step."""
    traj = current_trajectory()
    if traj is None:
        _fail_mode.handle_unattached("litellm.failure_callback")
        return
    if traj.is_noop:
        return

    await record_llm_call(
        model_id=str(kwargs.get("model", "unknown")),
        prompt_messages=_messages_from_kwargs(kwargs),
        completion="",
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        temperature=kwargs.get("temperature"),
        latency_ms=_latency_ms(start_time, end_time),
    )
