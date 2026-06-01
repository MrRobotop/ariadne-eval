"""Tests for the tau-bench → ariadne trajectory converter."""

from __future__ import annotations

import pytest

from ariadne_eval.benchmarks.tau_bench import _convert_tau_traj
from ariadne_eval.core.status import TrajectoryStatus
from ariadne_eval.core.trajectory import (
    LLMCallPayload,
    ToolCallPayload,
    UserInputPayload,
)

pytestmark = pytest.mark.fast


# A minimal EnvRunResult-shaped dict. We don't import tau_bench's types;
# the converter accepts a plain dict with the same keys.
def _make_env_result(
    *, reward: float, traj: list[dict[str, object]], task_id: str
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "reward": reward,
        "info": {},
        "traj": traj,
    }


def test_convert_zero_tool_calls_final_answer() -> None:
    """An assistant turn with no tool_calls becomes the final answer."""
    env_result = _make_env_result(
        task_id="task-1",
        reward=1.0,
        traj=[
            {"role": "user", "content": "Add 1 and 2."},
            {"role": "assistant", "content": "The answer is 3."},
        ],
    )
    traj, steps = _convert_tau_traj(
        env_result,
        instruction="Add 1 and 2.",
        model_id="anthropic/claude-haiku-4-5",
        agent_name="tau-bench/tool-calling",
        agent_version="59a200c",
    )
    assert traj.task == "Add 1 and 2."
    assert traj.final_answer == "The answer is 3."
    assert traj.final_status == TrajectoryStatus.SUCCEEDED  # reward >= 1.0
    assert traj.metadata["tau_bench_reward"] == 1.0
    assert traj.metadata["tau_bench_task_id"] == "task-1"
    # 1 user step + 1 llm step = 2 steps total
    assert len(steps) == 2
    assert isinstance(steps[0].payload, UserInputPayload)
    assert isinstance(steps[1].payload, LLMCallPayload)


def test_convert_multi_step_tool_calls() -> None:
    """Assistant with tool_calls → parent LLM Step + child ToolCall Steps."""
    env_result = _make_env_result(
        task_id="task-2",
        reward=1.0,
        traj=[
            {"role": "user", "content": "What is 17 * 23?"},
            {
                "role": "assistant",
                "content": "I'll use the calculator.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "calculator", "arguments": '{"expression":"17*23"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "391"},
            {"role": "assistant", "content": "The answer is 391."},
        ],
    )
    traj, steps = _convert_tau_traj(
        env_result,
        instruction="What is 17 * 23?",
        model_id="anthropic/claude-haiku-4-5",
        agent_name="tau-bench/tool-calling",
        agent_version="59a200c",
    )
    # Steps: user, llm_call (parent) → tool_call (child), final llm_call = 4 steps
    assert len(steps) == 4
    user, llm_parent, tool_child, llm_final = steps
    assert isinstance(user.payload, UserInputPayload)
    assert isinstance(llm_parent.payload, LLMCallPayload)
    assert isinstance(tool_child.payload, ToolCallPayload)
    assert isinstance(llm_final.payload, LLMCallPayload)
    assert tool_child.parent_step_id == llm_parent.id
    assert tool_child.payload.tool_name == "calculator"
    assert tool_child.payload.arguments == {"expression": "17*23"}
    assert tool_child.payload.result == "391"
    assert traj.final_answer == "The answer is 391."


def test_convert_failure_records_status() -> None:
    """reward < 1.0 → final_status=FAILED, traj.metadata records reward."""
    env_result = _make_env_result(
        task_id="task-3",
        reward=0.0,
        traj=[
            {"role": "user", "content": "Place an order."},
            {"role": "assistant", "content": "I cannot help with that."},
        ],
    )
    traj, _ = _convert_tau_traj(
        env_result,
        instruction="Place an order.",
        model_id="anthropic/claude-haiku-4-5",
        agent_name="tau-bench/tool-calling",
        agent_version="59a200c",
    )
    assert traj.final_status == TrajectoryStatus.FAILED
    assert traj.metadata["tau_bench_reward"] == 0.0
