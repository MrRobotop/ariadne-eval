"""Payloads, Step, Trajectory, validators, truncation.

This file accumulates as later tasks add more behaviour. Each task adds
its tests to this file rather than spawning a new one.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from ariadne_eval.core.trajectory import (
    InternalPayload,
    LLMCallPayload,
    Message,
    Payload,
    StepError,
    ToolCallPayload,
    UserInputPayload,
)


@pytest.mark.fast
def test_llm_call_payload_minimal():
    p = LLMCallPayload(
        model_id="claude-sonnet",
        prompt_messages=[Message(role="user", content="hi")],
        completion="hello",
        input_tokens=10,
        output_tokens=2,
        cost_usd=0.0001,
        latency_ms=42.0,
    )
    assert p.step_type == "llm_call"
    assert p.completion_truncated is False
    assert p.tool_calls_emitted == []


@pytest.mark.fast
def test_tool_call_payload_minimal():
    p = ToolCallPayload(
        tool_name="search",
        arguments={"q": "ariadne"},
        result={"hits": 3},
        latency_ms=12.0,
    )
    assert p.step_type == "tool_call"
    assert p.result_truncated is False


@pytest.mark.fast
def test_user_input_payload_minimal():
    p = UserInputPayload(message="please continue")
    assert p.step_type == "user_input"
    assert p.channel is None


@pytest.mark.fast
def test_internal_payload_minimal():
    p = InternalPayload(kind="branch", data={"reason": "retry"})
    assert p.step_type == "internal"


@pytest.mark.fast
def test_payload_discriminator_resolves_correctly():
    """The discriminator must select the right variant on deserialization."""
    adapter = TypeAdapter(Payload)

    raw = {
        "step_type": "tool_call",
        "tool_name": "search",
        "arguments": {"q": "x"},
        "result": None,
        "latency_ms": 1.0,
    }
    inst = adapter.validate_python(raw)
    assert isinstance(inst, ToolCallPayload)


@pytest.mark.fast
def test_payload_discriminator_rejects_unknown_step_type():
    adapter = TypeAdapter(Payload)
    with pytest.raises(ValidationError):
        adapter.validate_python({"step_type": "made_up", "x": 1})


# ---------------------------------------------------------------------------
# Task 6 — StepError
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_step_error_minimal():
    e = StepError(type="TimeoutError", message="timed out after 30s")
    assert e.traceback is None


@pytest.mark.fast
def test_step_error_with_traceback():
    e = StepError(
        type="ValueError",
        message="bad input",
        traceback="Traceback (most recent call last):\n  ...\nValueError: bad input",
    )
    assert "Traceback" in (e.traceback or "")


@pytest.mark.fast
def test_step_error_round_trip():
    e = StepError(type="X", message="y")
    assert StepError.model_validate(e.model_dump()) == e


# ---------------------------------------------------------------------------
# Task 7 — Step model with validators
# ---------------------------------------------------------------------------

from datetime import UTC, datetime  # noqa: E402

from ariadne_eval.core.ids import new_id  # noqa: E402
from ariadne_eval.core.status import StepStatus  # noqa: E402
from ariadne_eval.core.trajectory import Step  # noqa: E402


def _ll() -> LLMCallPayload:
    return LLMCallPayload(
        model_id="m",
        prompt_messages=[Message(role="user", content="hi")],
        completion="hello",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        latency_ms=1.0,
    )


@pytest.mark.fast
def test_step_minimal_succeeded():
    sid, tid = new_id(), new_id()
    s = Step(
        id=sid,
        trajectory_id=tid,
        parent_step_id=None,
        name="ask llm",
        started_at=datetime.now(tz=UTC),
        finished_at=datetime.now(tz=UTC),
        status=StepStatus.SUCCEEDED,
        payload=_ll(),
    )
    assert s.error is None
    assert s.metadata == {}


@pytest.mark.fast
def test_step_rejects_naive_started_at():
    with pytest.raises(ValidationError) as exc:
        Step(
            id=new_id(),
            trajectory_id=new_id(),
            parent_step_id=None,
            name="x",
            started_at=datetime(2026, 5, 10, 12, 0, 0),  # naive
            finished_at=None,
            status=StepStatus.RUNNING,
            payload=_ll(),
        )
    msg = str(exc.value).lower()
    assert "tz-aware" in msg or "timezone" in msg


@pytest.mark.fast
def test_step_rejects_naive_finished_at():
    with pytest.raises(ValidationError):
        Step(
            id=new_id(),
            trajectory_id=new_id(),
            parent_step_id=None,
            name="x",
            started_at=datetime.now(tz=UTC),
            finished_at=datetime(2026, 5, 10, 12, 0, 0),  # naive
            status=StepStatus.SUCCEEDED,
            payload=_ll(),
        )


@pytest.mark.fast
def test_step_rejects_self_parent():
    sid = new_id()
    with pytest.raises(ValidationError) as exc:
        Step(
            id=sid,
            trajectory_id=new_id(),
            parent_step_id=sid,
            name="x",
            started_at=datetime.now(tz=UTC),
            finished_at=None,
            status=StepStatus.RUNNING,
            payload=_ll(),
        )
    msg = str(exc.value).lower()
    assert "self" in msg or "parent_step_id" in msg


@pytest.mark.fast
def test_failed_step_requires_error():
    with pytest.raises(ValidationError) as exc:
        Step(
            id=new_id(),
            trajectory_id=new_id(),
            parent_step_id=None,
            name="x",
            started_at=datetime.now(tz=UTC),
            finished_at=datetime.now(tz=UTC),
            status=StepStatus.FAILED,
            payload=_ll(),
            error=None,
        )
    assert "error" in str(exc.value).lower()


@pytest.mark.fast
def test_step_round_trip_json():
    s = Step(
        id=new_id(),
        trajectory_id=new_id(),
        parent_step_id=None,
        name="x",
        started_at=datetime.now(tz=UTC),
        finished_at=None,
        status=StepStatus.RUNNING,
        payload=_ll(),
    )
    dumped = s.model_dump_json()
    rehydrated = Step.model_validate_json(dumped)
    assert rehydrated == s


@pytest.mark.fast
def test_step_id_must_be_valid_ulid():
    with pytest.raises(ValidationError):
        Step(
            id="not-a-ulid",
            trajectory_id=new_id(),
            parent_step_id=None,
            name="x",
            started_at=datetime.now(tz=UTC),
            finished_at=None,
            status=StepStatus.RUNNING,
            payload=_ll(),
        )
