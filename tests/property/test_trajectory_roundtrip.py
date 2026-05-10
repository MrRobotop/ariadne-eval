"""Property-based round-trip: any Trajectory we can construct must serialize
and deserialize to an equal value."""

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import StepStatus, TrajectoryStatus
from ariadne_eval.core.trajectory import (
    InternalPayload,
    LLMCallPayload,
    Message,
    Step,
    ToolCallPayload,
    Trajectory,
    UserInputPayload,
)


_BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


@st.composite
def _datetimes(draw):
    seconds = draw(st.integers(min_value=0, max_value=10_000_000))
    return _BASE_TIME + timedelta(seconds=seconds)


@st.composite
def _llm_payloads(draw):
    return LLMCallPayload(
        model_id=draw(st.sampled_from(["claude-sonnet", "gpt-4o", "haiku"])),
        prompt_messages=[Message(role="user", content=draw(st.text(max_size=64)))],
        completion=draw(st.text(max_size=128)),
        input_tokens=draw(st.integers(min_value=0, max_value=10_000)),
        output_tokens=draw(st.integers(min_value=0, max_value=10_000)),
        cost_usd=draw(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        ),
        latency_ms=draw(
            st.floats(min_value=0.0, max_value=5_000.0, allow_nan=False, allow_infinity=False)
        ),
    )


@st.composite
def _tool_payloads(draw):
    return ToolCallPayload(
        tool_name=draw(st.sampled_from(["search", "calculator", "fetch"])),
        arguments={"q": draw(st.text(max_size=32))},
        result=draw(st.one_of(st.none(), st.integers(), st.text(max_size=64))),
        latency_ms=draw(
            st.floats(min_value=0.0, max_value=5_000.0, allow_nan=False, allow_infinity=False)
        ),
    )


@st.composite
def _user_payloads(draw):
    return UserInputPayload(message=draw(st.text(max_size=64)))


@st.composite
def _internal_payloads(draw):
    return InternalPayload(kind=draw(st.text(min_size=1, max_size=16, alphabet="abcdef")))


_payloads = st.one_of(_llm_payloads(), _tool_payloads(), _user_payloads(), _internal_payloads())


@st.composite
def _steps(draw):
    started = draw(_datetimes())
    return Step(
        id=new_id(),
        trajectory_id=new_id(),
        parent_step_id=None,
        name=draw(st.text(min_size=1, max_size=24)),
        started_at=started,
        finished_at=started + timedelta(milliseconds=draw(st.integers(min_value=0, max_value=10_000))),
        status=StepStatus.SUCCEEDED,
        payload=draw(_payloads),
    )


@st.composite
def _trajectories(draw):
    started = draw(_datetimes())
    return Trajectory(
        id=new_id(),
        task=draw(st.text(min_size=1, max_size=64)),
        agent_name=draw(st.sampled_from(["react", "tool-use"])),
        agent_version=draw(st.sampled_from(["0.1", "0.2"])),
        model_id=draw(st.sampled_from(["claude-sonnet", "gpt-4o"])),
        started_at=started,
        finished_at=started + timedelta(seconds=draw(st.integers(min_value=0, max_value=600))),
        final_status=draw(st.sampled_from(list(TrajectoryStatus))),
        final_answer=draw(st.one_of(st.none(), st.text(max_size=64))),
        metadata={"k": draw(st.text(max_size=16))},
    )


@pytest.mark.fast
@given(t=_trajectories())
@settings(max_examples=200, deadline=None)
def test_trajectory_json_round_trip(t):
    rehydrated = Trajectory.model_validate_json(t.model_dump_json())
    assert rehydrated == t


@pytest.mark.fast
@given(s=_steps())
@settings(max_examples=200, deadline=None)
def test_step_json_round_trip(s):
    rehydrated = Step.model_validate_json(s.model_dump_json())
    assert rehydrated == s
