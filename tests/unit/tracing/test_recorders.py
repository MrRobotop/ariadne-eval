"""record_llm_call and record_tool_call attachment behavior."""

from __future__ import annotations

import pytest

from ariadne_eval.core.status import StepStatus
from ariadne_eval.core.trajectory import (
    LLMCallPayload,
    Message,
    StepError,
    ToolCallPayload,
)
from ariadne_eval.tracing.context import start_trajectory
from ariadne_eval.tracing.decorator import (
    record_llm_call,
    record_tool_call,
    trace_step,
)


@pytest.mark.fast
async def test_record_llm_call_attaches_to_trajectory_root_when_no_step():
    async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m") as traj:
        sid = await record_llm_call(
            model_id="claude",
            prompt_messages=[Message(role="user", content="hi")],
            completion="hi back",
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
            latency_ms=1.0,
        )
        assert sid != ""
    [step] = traj._steps
    assert step.parent_step_id is None
    assert isinstance(step.payload, LLMCallPayload)
    assert step.status == StepStatus.SUCCEEDED


@pytest.mark.fast
async def test_record_llm_call_attaches_to_current_step():
    @trace_step("outer")
    async def outer() -> None:
        await record_llm_call(
            model_id="claude",
            prompt_messages=[Message(role="user", content="hi")],
            completion="hi back",
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
            latency_ms=1.0,
        )

    async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m") as traj:
        await outer()

    outer_step = next(s for s in traj._steps if s.name == "outer")
    llm_step = next(s for s in traj._steps if s.name == "llm_call")
    assert llm_step.parent_step_id == outer_step.id


@pytest.mark.fast
async def test_record_tool_call_with_error_marks_failed():
    async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m") as traj:
        await record_tool_call(
            tool_name="search",
            arguments={"q": "x"},
            result=None,
            latency_ms=1.0,
            error=StepError(type="TimeoutError", message="took too long"),
        )
    [step] = traj._steps
    assert step.status == StepStatus.FAILED
    assert step.error is not None
    assert step.error.type == "TimeoutError"
    assert isinstance(step.payload, ToolCallPayload)


@pytest.mark.fast
async def test_record_tool_call_default_name_is_tool_name():
    async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m") as traj:
        await record_tool_call(
            tool_name="calculator",
            arguments={"expr": "2+2"},
            result=4,
            latency_ms=1.0,
        )
    assert traj._steps[0].name == "calculator"


@pytest.mark.fast
async def test_recorders_return_empty_string_under_noop():
    from ariadne_eval.tracing.sampler import RateSampler

    async with start_trajectory(
        "t",
        agent_name="a",
        agent_version="0.1",
        model_id="m",
        sampler=RateSampler(rate=0.0),
    ) as traj:
        sid = await record_llm_call(
            model_id="claude",
            prompt_messages=[Message(role="user", content="hi")],
            completion="x",
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
            latency_ms=1.0,
        )
    assert sid == ""
    assert traj._steps == []
