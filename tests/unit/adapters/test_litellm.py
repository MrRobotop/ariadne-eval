"""LiteLLM auto-trace adapter."""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest

from ariadne_eval.tracing.context import start_trajectory


@pytest.fixture
def fake_litellm(monkeypatch):
    """Install a stub litellm module."""
    stub = types.ModuleType("litellm")
    stub.success_callback = []
    stub.failure_callback = []
    stub.completion_cost = lambda response: 0.0
    monkeypatch.setitem(sys.modules, "litellm", stub)
    # Reset the module-level _registered flag so each test starts clean.
    from ariadne_eval.adapters import litellm as adapter

    importlib.reload(adapter)
    yield stub


@pytest.mark.fast
def test_enable_litellm_autotrace_registers_callbacks(fake_litellm):
    from ariadne_eval.adapters.litellm import enable_litellm_autotrace

    enable_litellm_autotrace()
    assert len(fake_litellm.success_callback) == 1
    assert len(fake_litellm.failure_callback) == 1


@pytest.mark.fast
async def test_callback_records_llm_call(fake_litellm):
    """When the success callback fires inside a trajectory, an llm_call step lands."""
    from ariadne_eval.adapters.litellm import _on_success, enable_litellm_autotrace

    enable_litellm_autotrace()

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "hello back"
    response.usage.prompt_tokens = 5
    response.usage.completion_tokens = 2

    kwargs = {
        "model": "claude-sonnet",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
    }

    async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m") as traj:
        await _on_success(kwargs, response, start_time=0.0, end_time=0.05)

    assert len(traj._steps) == 1
    step = traj._steps[0]
    assert step.name == "llm_call"
    assert step.payload.model_id == "claude-sonnet"
    assert step.payload.completion == "hello back"
    assert step.payload.input_tokens == 5
    assert step.payload.output_tokens == 2
    assert step.payload.latency_ms == pytest.approx(50.0)


@pytest.mark.fast
def test_enable_litellm_autotrace_is_idempotent(fake_litellm):
    from ariadne_eval.adapters.litellm import enable_litellm_autotrace

    enable_litellm_autotrace()
    enable_litellm_autotrace()
    assert len(fake_litellm.success_callback) == 1
    assert len(fake_litellm.failure_callback) == 1


@pytest.mark.fast
async def test_failure_callback_records_failed_llm_call(fake_litellm):
    from ariadne_eval.adapters.litellm import _on_failure

    kwargs = {
        "model": "claude-sonnet",
        "messages": [{"role": "user", "content": "hi"}],
    }
    async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m") as traj:
        await _on_failure(kwargs, None, start_time=0.0, end_time=0.02)
    assert len(traj._steps) == 1
    step = traj._steps[0]
    assert step.payload.model_id == "claude-sonnet"
    assert step.payload.completion == ""
    assert step.payload.latency_ms == pytest.approx(20.0)


@pytest.mark.fast
async def test_success_callback_unattached_silent(monkeypatch, fake_litellm):
    """Success callback fired outside a trajectory under FAIL_MODE=silent: no-op."""
    monkeypatch.setenv("ARIADNE_FAIL_MODE", "silent")
    import importlib

    from ariadne_eval.tracing import _fail_mode

    importlib.reload(_fail_mode)

    from ariadne_eval.adapters.litellm import _on_failure, _on_success

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "ignored"
    response.usage.prompt_tokens = 0
    response.usage.completion_tokens = 0

    # Both callbacks fire outside any active trajectory.
    await _on_success({"model": "x"}, response, 0.0, 0.0)
    await _on_failure({"model": "x"}, None, 0.0, 0.0)


@pytest.mark.fast
async def test_callback_under_noop_handle(fake_litellm):
    """When the active handle is a no-op, callbacks return without recording."""
    from ariadne_eval.adapters.litellm import _on_failure, _on_success
    from ariadne_eval.tracing.sampler import RateSampler

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "x"
    response.usage.prompt_tokens = 0
    response.usage.completion_tokens = 0

    async with start_trajectory(
        "t",
        agent_name="a",
        agent_version="0.1",
        model_id="m",
        sampler=RateSampler(rate=0.0),
    ) as traj:
        await _on_success({"model": "x"}, response, 0.0, 0.0)
        await _on_failure({"model": "x"}, None, 0.0, 0.0)
    assert traj._steps == []


@pytest.mark.fast
def test_messages_from_kwargs_passes_through_existing_message(fake_litellm):
    """Pre-built Message objects in kwargs are preserved (not re-wrapped)."""
    from ariadne_eval.adapters.litellm import _messages_from_kwargs
    from ariadne_eval.core.trajectory import Message

    msg = Message(role="user", content="hi")
    out = _messages_from_kwargs({"messages": [msg]})
    assert out == [msg]
