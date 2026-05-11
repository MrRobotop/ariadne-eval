"""@trace_step decorator and the explicit record_* recorders."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Literal, TypeVar, cast

from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import StepStatus
from ariadne_eval.core.trajectory import (
    InternalPayload,
    JsonValue,
    LLMCallPayload,
    Message,
    Step,
    StepError,
    ToolCallPayload,
)
from ariadne_eval.tracing import _fail_mode
from ariadne_eval.tracing.context import (
    TrajectoryHandle,
    _current_step,
    current_step,
    current_trajectory,
)

__all__ = ["record_llm_call", "record_tool_call", "trace_step"]


F = TypeVar("F", bound=Callable[..., Any])

_AllowedStepTypes = Literal["internal"]


def trace_step(
    name: str,
    *,
    step_type: _AllowedStepTypes = "internal",
) -> Callable[[F], F]:
    """Wrap a function so each call appears as a Step in the current trajectory.

    In v0.0.4, only ``step_type="internal"`` is supported. For LLM and tool
    calls, use :func:`record_llm_call` and :func:`record_tool_call` directly.
    """
    if step_type != "internal":
        raise ValueError(
            f"@trace_step only supports step_type='internal' in v0.0.4; "
            f"got {step_type!r}"
        )

    def decorator(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                traj = current_trajectory()
                if traj is None:
                    _fail_mode.handle_unattached("@trace_step")
                    return await fn(*args, **kwargs)
                if traj.is_noop:
                    return await fn(*args, **kwargs)
                step, token = _begin_step(name=name, traj=traj)
                try:
                    result = await fn(*args, **kwargs)
                except BaseException as exc:
                    _finish_step(step, status=StepStatus.FAILED, exc=exc)
                    traj.append_step(step)
                    _current_step.reset(token)
                    raise
                _finish_step(step, status=StepStatus.SUCCEEDED)
                traj.append_step(step)
                _current_step.reset(token)
                return result

            return cast(F, async_wrapper)

        @wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            traj = current_trajectory()
            if traj is None:
                _fail_mode.handle_unattached("@trace_step")
                return fn(*args, **kwargs)
            if traj.is_noop:
                return fn(*args, **kwargs)
            step, token = _begin_step(name=name, traj=traj)
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                _finish_step(step, status=StepStatus.FAILED, exc=exc)
                traj.append_step(step)
                _current_step.reset(token)
                raise
            _finish_step(step, status=StepStatus.SUCCEEDED)
            traj.append_step(step)
            _current_step.reset(token)
            return result

        return cast(F, sync_wrapper)

    return decorator


def _begin_step(*, name: str, traj: TrajectoryHandle) -> tuple[Step, Any]:
    """Build a RUNNING Step and set the current_step ContextVar."""
    parent = current_step()
    started = datetime.now(tz=timezone.utc)
    step = Step(
        id=new_id(),
        trajectory_id=traj.id,
        parent_step_id=parent.id if parent is not None else None,
        name=name,
        started_at=started,
        finished_at=None,
        status=StepStatus.RUNNING,
        payload=InternalPayload(kind=name),
    )
    token = _current_step.set(step)
    return step, token


def _finish_step(
    step: Step,
    *,
    status: StepStatus,
    exc: BaseException | None = None,
) -> None:
    """Mutate a step in place to its terminal state.

    Uses ``object.__setattr__`` to bypass the Pydantic validator chain that
    would otherwise reject a partial transition to ``FAILED`` before
    ``error`` is also set. The model validator only runs at construction.
    """
    object.__setattr__(step, "finished_at", datetime.now(tz=timezone.utc))
    object.__setattr__(step, "status", status)
    if exc is not None:
        object.__setattr__(
            step,
            "error",
            StepError(type=type(exc).__name__, message=str(exc)),
        )


async def record_llm_call(
    *,
    model_id: str,
    prompt_messages: list[Message],
    completion: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    temperature: float | None = None,
    latency_ms: float,
    ttft_ms: float | None = None,
    tool_calls_emitted: list[str] | None = None,
    name: str = "llm_call",
) -> str:
    """Record an LLM call as a Step in the current trajectory.

    Returns the step id. If no trajectory is active, the configured
    ``ARIADNE_FAIL_MODE`` policy applies.
    """
    traj = current_trajectory()
    if traj is None:
        _fail_mode.handle_unattached("record_llm_call")
        return ""
    if traj.is_noop:
        return ""

    parent = current_step()
    now = datetime.now(tz=timezone.utc)
    payload = LLMCallPayload(
        model_id=model_id,
        prompt_messages=prompt_messages,
        completion=completion,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        temperature=temperature,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        tool_calls_emitted=tool_calls_emitted or [],
    )
    step = Step(
        id=new_id(),
        trajectory_id=traj.id,
        parent_step_id=parent.id if parent is not None else None,
        name=name,
        started_at=now,
        finished_at=now,
        status=StepStatus.SUCCEEDED,
        payload=payload,
    )
    traj.append_step(step)
    return step.id


async def record_tool_call(
    *,
    tool_name: str,
    arguments: dict[str, JsonValue],
    result: JsonValue = None,
    latency_ms: float,
    name: str | None = None,
    error: StepError | None = None,
) -> str:
    """Record a tool call as a Step in the current trajectory.

    Returns the step id. If ``error`` is set, the step's status is FAILED.
    """
    traj = current_trajectory()
    if traj is None:
        _fail_mode.handle_unattached("record_tool_call")
        return ""
    if traj.is_noop:
        return ""

    parent = current_step()
    now = datetime.now(tz=timezone.utc)
    payload = ToolCallPayload(
        tool_name=tool_name,
        arguments=arguments,
        result=result,
        latency_ms=latency_ms,
    )
    step = Step(
        id=new_id(),
        trajectory_id=traj.id,
        parent_step_id=parent.id if parent is not None else None,
        name=name or tool_name,
        started_at=now,
        finished_at=now,
        status=StepStatus.FAILED if error is not None else StepStatus.SUCCEEDED,
        payload=payload,
        error=error,
    )
    traj.append_step(step)
    return step.id
