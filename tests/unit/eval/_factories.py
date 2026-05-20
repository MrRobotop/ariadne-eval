"""Test-only factories for Trajectory / Step instances.

Keep these tiny and dumb. If a test needs a wildly different shape, build
the model directly in the test rather than expanding these helpers.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import StepStatus, TrajectoryStatus
from ariadne_eval.core.trajectory import (
    JsonValue,
    Step,
    ToolCallPayload,
    Trajectory,
)


def make_trajectory(
    *,
    final_answer: JsonValue = "ok",
    final_status: TrajectoryStatus = TrajectoryStatus.SUCCEEDED,
    task: str = "demo",
    traj_id: str | None = None,
) -> Trajectory:
    started = datetime(2026, 5, 14, 12, 0, 0, tzinfo=UTC)
    return Trajectory(
        id=traj_id or new_id(),
        task=task,
        agent_name="test",
        agent_version="0.0.0",
        model_id="test/model",
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        final_status=final_status,
        final_answer=final_answer,
    )


def make_tool_step(
    *,
    trajectory_id: str,
    tool_name: str,
    arguments: dict[str, JsonValue] | None = None,
    result: JsonValue = None,
    started_at: datetime | None = None,
    parent_step_id: str | None = None,
) -> Step:
    started = started_at or datetime(2026, 5, 14, 12, 0, 0, tzinfo=UTC)
    return Step(
        id=new_id(),
        trajectory_id=trajectory_id,
        parent_step_id=parent_step_id,
        name=f"tool:{tool_name}",
        started_at=started,
        finished_at=started + timedelta(milliseconds=10),
        status=StepStatus.SUCCEEDED,
        payload=ToolCallPayload(
            tool_name=tool_name,
            arguments=arguments or {},
            result=result,
            latency_ms=10.0,
        ),
    )
