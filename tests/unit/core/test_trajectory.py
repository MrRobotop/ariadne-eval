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
