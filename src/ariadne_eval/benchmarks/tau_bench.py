"""tau-bench adapter and trajectory converter.

The adapter class lazy-imports the ``tau_bench`` package; users who
don't install the ``[tau-bench]`` extra never trigger the import. The
converter below is pure-Python and does not depend on the tau_bench
package — it accepts the EnvRunResult shape as a dict so it can be
exercised from unit tests without the extra installed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import StepStatus, TrajectoryStatus
from ariadne_eval.core.trajectory import (
    JsonValue,
    LLMCallPayload,
    Message,
    Step,
    ToolCallPayload,
    Trajectory,
    UserInputPayload,
)

__all__ = ["_convert_tau_traj"]


_SUCCESS_THRESHOLD = 1.0 - 1e-6


def _convert_tau_traj(
    env_result: dict[str, object],
    *,
    instruction: str,
    model_id: str,
    agent_name: str,
    agent_version: str,
) -> tuple[Trajectory, list[Step]]:
    """Convert a tau-bench ``EnvRunResult``-shaped dict to ``(Trajectory, list[Step])``.

    Accepts a plain dict with keys ``task_id``, ``reward``, ``info``,
    ``traj`` (a list of message dicts). This lets the converter be
    tested without installing the ``tau_bench`` package.
    """
    traj_id = new_id()
    started = datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC)

    reward = float(cast(float, env_result["reward"]))
    task_id = str(env_result["task_id"])
    raw_messages = cast(list[dict[str, Any]], env_result["traj"])

    steps: list[Step] = []
    final_answer: str | None = None
    # tool_call_id → the child Step (so we can fill result once the tool-result message arrives)
    pending_tool_payloads: dict[str, Step] = {}
    msg_index = 0

    for raw_msg in raw_messages:
        role = raw_msg.get("role")
        content = raw_msg.get("content", "") or ""
        step_started = started + timedelta(seconds=msg_index)
        step_finished = step_started + timedelta(milliseconds=10)

        if role == "user":
            step = Step(
                id=new_id(),
                trajectory_id=traj_id,
                parent_step_id=None,
                name="user_input",
                started_at=step_started,
                finished_at=step_finished,
                status=StepStatus.SUCCEEDED,
                payload=UserInputPayload(message=str(content)),
            )
            steps.append(step)

        elif role == "assistant":
            tool_calls = raw_msg.get("tool_calls") or []
            llm_step = Step(
                id=new_id(),
                trajectory_id=traj_id,
                parent_step_id=None,
                name="llm_call",
                started_at=step_started,
                finished_at=step_finished,
                status=StepStatus.SUCCEEDED,
                payload=LLMCallPayload(
                    model_id=model_id,
                    prompt_messages=[Message(role="user", content=instruction)],
                    completion=str(content),
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=10.0,
                    cost_usd=0.0,
                ),
            )
            steps.append(llm_step)

            if not tool_calls:
                # Final-answer assistant turn (no tool calls)
                if content:
                    final_answer = str(content)
            else:
                # Emit one child ToolCall step per tool_call
                for tool_call in tool_calls:
                    fn = tool_call.get("function", {})
                    tool_name = str(fn.get("name", "unknown"))
                    arguments_raw = fn.get("arguments", "{}")
                    try:
                        arguments: dict[str, Any] = (
                            json.loads(arguments_raw)
                            if isinstance(arguments_raw, str)
                            else dict(arguments_raw)
                        )
                    except (json.JSONDecodeError, TypeError):
                        arguments = {"_raw": str(arguments_raw)}
                    tool_step = Step(
                        id=new_id(),
                        trajectory_id=traj_id,
                        parent_step_id=llm_step.id,
                        name=f"tool_{tool_name}",
                        started_at=step_started + timedelta(milliseconds=1),
                        finished_at=step_finished,
                        status=StepStatus.SUCCEEDED,
                        payload=ToolCallPayload(
                            tool_name=tool_name,
                            arguments=cast("dict[str, JsonValue]", arguments),
                            result=None,
                            latency_ms=10.0,
                        ),
                    )
                    steps.append(tool_step)
                    pending_tool_payloads[str(tool_call.get("id", ""))] = tool_step

        elif role == "tool":
            tool_call_id = str(raw_msg.get("tool_call_id", ""))
            if tool_call_id in pending_tool_payloads:
                child_step = pending_tool_payloads.pop(tool_call_id)
                # Use model_copy(update=...) rather than direct mutation to re-run field validators.
                new_payload = cast(ToolCallPayload, child_step.payload).model_copy(
                    update={"result": str(content)}
                )
                # Replace the step in `steps` with one carrying the updated payload
                idx = steps.index(child_step)
                steps[idx] = child_step.model_copy(update={"payload": new_payload})

        # All other roles (system, etc.) are ignored
        msg_index += 1

    final_status = (
        TrajectoryStatus.SUCCEEDED if reward >= _SUCCESS_THRESHOLD else TrajectoryStatus.FAILED
    )

    trajectory = Trajectory(
        id=traj_id,
        task=instruction,
        agent_name=agent_name,
        agent_version=agent_version,
        model_id=model_id,
        started_at=started,
        finished_at=started + timedelta(seconds=max(msg_index, 1)),
        final_status=final_status,
        final_answer=final_answer,
        metadata={
            "tau_bench_reward": reward,
            "tau_bench_task_id": task_id,
        },
    )

    return trajectory, steps
