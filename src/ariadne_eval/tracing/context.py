"""TrajectoryHandle, start_trajectory CM, and ContextVar accessors."""

from __future__ import annotations

import traceback as _tb
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import TrajectoryStatus
from ariadne_eval.core.trajectory import JsonValue, Step, Trajectory
from ariadne_eval.tracing.sampler import AlwaysSampler, Sampler

if TYPE_CHECKING:
    from ariadne_eval.storage.base import Store

__all__ = [
    "TrajectoryHandle",
    "current_step",
    "current_trajectory",
    "start_trajectory",
]


_current_trajectory: ContextVar[TrajectoryHandle | None] = ContextVar(
    "ariadne_current_trajectory", default=None
)
_current_step: ContextVar[Step | None] = ContextVar("ariadne_current_step", default=None)


def current_trajectory() -> TrajectoryHandle | None:
    """Return the active trajectory handle in this async context, or None."""
    return _current_trajectory.get()


def current_step() -> Step | None:
    """Return the active step in this async context, or None."""
    return _current_step.get()


@dataclass
class TrajectoryHandle:
    """Mutable builder for an in-flight trajectory.

    The context manager yields one of these. Recorders append to ``_steps``.
    Call ``snapshot(...)`` to produce the frozen ``Trajectory`` model for
    storage.
    """

    id: str
    task: str
    agent_name: str
    agent_version: str
    model_id: str
    started_at: datetime
    _steps: list[Step] = field(default_factory=list)
    _metadata: dict[str, JsonValue] = field(default_factory=dict)
    _final_answer: JsonValue = None
    _final_status_override: TrajectoryStatus | None = None
    is_noop: bool = False

    def add_metadata(self, key: str, value: JsonValue) -> None:
        """Attach a free-form metadata key-value pair to the trajectory."""
        if self.is_noop:
            return
        self._metadata[key] = value

    def set_final_answer(self, answer: JsonValue) -> None:
        """Record the trajectory's final answer."""
        if self.is_noop:
            return
        self._final_answer = answer

    def set_final_status(self, status: TrajectoryStatus) -> None:
        """Override the trajectory's terminal status (SUCCEEDED by default)."""
        if self.is_noop:
            return
        self._final_status_override = status

    def append_step(self, step: Step) -> None:
        """Internal: append a recorded Step. No-op handles drop the step."""
        if self.is_noop:
            return
        self._steps.append(step)

    def snapshot(
        self,
        *,
        finished_at: datetime,
        default_status: TrajectoryStatus,
    ) -> Trajectory:
        """Freeze the current handle state into a Trajectory model."""
        root_step_id = self._steps[0].id if self._steps else None
        return Trajectory(
            id=self.id,
            task=self.task,
            agent_name=self.agent_name,
            agent_version=self.agent_version,
            model_id=self.model_id,
            started_at=self.started_at,
            finished_at=finished_at,
            final_status=self._final_status_override or default_status,
            final_answer=self._final_answer,
            root_step_id=root_step_id,
            metadata=dict(self._metadata),
        )


@asynccontextmanager
async def start_trajectory(
    task: str,
    *,
    agent_name: str,
    agent_version: str,
    model_id: str,
    store: Store | None = None,
    sampler: Sampler | None = None,
    metadata: dict[str, JsonValue] | None = None,
) -> AsyncIterator[TrajectoryHandle]:
    """Open an async trajectory context.

    See ``docs/superpowers/specs/2026-05-11-tracing-instrumentation-design.md``
    for the design.
    """
    chosen_sampler = sampler or AlwaysSampler()
    initial_metadata = dict(metadata) if metadata else {}

    sampled = chosen_sampler.should_sample(
        task=task,
        agent_name=agent_name,
        agent_version=agent_version,
        model_id=model_id,
        metadata=initial_metadata,
    )

    started_at = datetime.now(tz=UTC)
    handle = TrajectoryHandle(
        id=new_id(),
        task=task,
        agent_name=agent_name,
        agent_version=agent_version,
        model_id=model_id,
        started_at=started_at,
        _metadata=initial_metadata,
        is_noop=not sampled,
    )

    token = _current_trajectory.set(handle)
    raised = False
    try:
        yield handle
    except BaseException as exc:
        raised = True
        if not handle.is_noop:
            handle._metadata["_trajectory_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(_tb.format_exception(type(exc), exc, exc.__traceback__)),
            }
            handle._final_status_override = TrajectoryStatus.FAILED
        raise
    finally:
        _current_trajectory.reset(token)
        if not handle.is_noop and store is not None:
            default_status = TrajectoryStatus.FAILED if raised else TrajectoryStatus.SUCCEEDED
            snap = handle.snapshot(
                finished_at=datetime.now(tz=UTC),
                default_status=default_status,
            )
            await store.save_trajectory(snap, list(handle._steps))
