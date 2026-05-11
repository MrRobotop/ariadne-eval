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
    stub.completion_cost = lambda response: 0.0  # noqa: ARG005
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

    async with start_trajectory(
        "t", agent_name="a", agent_version="0.1", model_id="m"
    ) as traj:
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
