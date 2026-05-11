"""@trace_step decorator on sync and async functions."""

from __future__ import annotations

import pytest

from ariadne_eval.core.status import StepStatus
from ariadne_eval.core.trajectory import InternalPayload
from ariadne_eval.tracing.context import current_step, start_trajectory
from ariadne_eval.tracing.decorator import trace_step


@pytest.mark.fast
async def test_trace_step_on_async_function():
    @trace_step("plan")
    async def plan(x: int) -> int:
        return x * 2

    async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m") as traj:
        result = await plan(3)
        assert result == 6
    assert len(traj._steps) == 1
    step = traj._steps[0]
    assert step.name == "plan"
    assert step.status == StepStatus.SUCCEEDED
    assert isinstance(step.payload, InternalPayload)
    assert step.payload.kind == "plan"
    assert step.parent_step_id is None


@pytest.mark.fast
async def test_trace_step_on_sync_function():
    @trace_step("compute")
    def compute(x: int) -> int:
        return x + 1

    async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m") as traj:
        result = compute(2)
        assert result == 3
    assert len(traj._steps) == 1
    assert traj._steps[0].name == "compute"


@pytest.mark.fast
async def test_nested_trace_steps_attach_to_parent():
    @trace_step("inner")
    async def inner() -> int:
        return 1

    @trace_step("outer")
    async def outer() -> int:
        return await inner()

    async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m") as traj:
        await outer()

    outer_step = next(s for s in traj._steps if s.name == "outer")
    inner_step = next(s for s in traj._steps if s.name == "inner")
    assert inner_step.parent_step_id == outer_step.id
    assert outer_step.parent_step_id is None


@pytest.mark.fast
async def test_trace_step_records_failure_and_reraises():
    class BoomError(Exception):
        pass

    @trace_step("bad")
    async def bad() -> int:
        raise BoomError("nope")

    async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m") as traj:
        with pytest.raises(BoomError):
            await bad()
    step = traj._steps[0]
    assert step.status == StepStatus.FAILED
    assert step.error is not None
    assert step.error.type == "BoomError"
    assert step.error.message == "nope"


@pytest.mark.fast
async def test_current_step_resets_after_decorator_exits():
    @trace_step("only")
    async def only() -> int:
        assert current_step() is not None
        return 1

    async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m"):
        assert current_step() is None
        await only()
        assert current_step() is None


@pytest.mark.fast
def test_trace_step_rejects_non_internal_step_type():
    with pytest.raises(ValueError):
        trace_step("x", step_type="llm_call")  # type: ignore[arg-type]


@pytest.mark.fast
async def test_unattached_decorator_strict_raises(monkeypatch):
    monkeypatch.setenv("ARIADNE_FAIL_MODE", "strict")
    import importlib

    from ariadne_eval.tracing import _fail_mode

    importlib.reload(_fail_mode)

    @trace_step("orphan")
    async def orphan() -> int:
        return 1

    with pytest.raises(RuntimeError):
        await orphan()


@pytest.mark.fast
async def test_sync_trace_step_failure_records_and_reraises():
    class BoomError(Exception):
        pass

    @trace_step("bad_sync")
    def bad_sync() -> int:
        raise BoomError("sync nope")

    async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m") as traj:
        with pytest.raises(BoomError):
            bad_sync()
    step = traj._steps[0]
    assert step.status == StepStatus.FAILED
    assert step.error is not None
    assert step.error.type == "BoomError"


@pytest.mark.fast
async def test_sync_decorator_under_noop_handle_passes_through():
    """Under a no-op handle, the sync decorator must still call the wrapped fn."""
    from ariadne_eval.tracing.sampler import RateSampler

    called = []

    @trace_step("noop_sync")
    def noop_sync() -> int:
        called.append(True)
        return 7

    async with start_trajectory(
        "t",
        agent_name="a",
        agent_version="0.1",
        model_id="m",
        sampler=RateSampler(rate=0.0),
    ) as traj:
        result = noop_sync()
    assert result == 7
    assert called == [True]
    assert traj._steps == []  # no recording under no-op handle


@pytest.mark.fast
def test_sync_decorator_unattached_strict_raises(monkeypatch):
    monkeypatch.setenv("ARIADNE_FAIL_MODE", "strict")
    import importlib

    from ariadne_eval.tracing import _fail_mode

    importlib.reload(_fail_mode)

    @trace_step("sync_orphan")
    def sync_orphan() -> int:
        return 1

    with pytest.raises(RuntimeError):
        sync_orphan()


@pytest.mark.fast
async def test_unattached_recorders_under_silent_return_empty(monkeypatch):
    monkeypatch.setenv("ARIADNE_FAIL_MODE", "silent")
    import importlib

    from ariadne_eval.tracing import _fail_mode

    importlib.reload(_fail_mode)

    from ariadne_eval.core.trajectory import Message
    from ariadne_eval.tracing.decorator import record_llm_call, record_tool_call

    sid1 = await record_llm_call(
        model_id="m",
        prompt_messages=[Message(role="user", content="x")],
        completion="y",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        latency_ms=1.0,
    )
    sid2 = await record_tool_call(
        tool_name="t",
        arguments={"a": 1},
        result=None,
        latency_ms=1.0,
    )
    assert sid1 == ""
    assert sid2 == ""
