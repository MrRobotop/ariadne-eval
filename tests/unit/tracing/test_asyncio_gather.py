"""ContextVar correctness under asyncio.gather and TaskGroup."""

from __future__ import annotations

import asyncio

import pytest

from ariadne_eval.tracing.context import start_trajectory
from ariadne_eval.tracing.decorator import trace_step


@pytest.mark.fast
async def test_parallel_children_under_gather_attach_to_parent():
    @trace_step("child")
    async def child(idx: int) -> int:
        await asyncio.sleep(0)
        return idx

    @trace_step("parent")
    async def parent() -> list[int]:
        return await asyncio.gather(child(1), child(2), child(3))

    async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m") as traj:
        await parent()

    parent_step = next(s for s in traj._steps if s.name == "parent")
    child_steps = [s for s in traj._steps if s.name == "child"]
    assert len(child_steps) == 3
    for c in child_steps:
        assert c.parent_step_id == parent_step.id


@pytest.mark.fast
async def test_parallel_children_under_taskgroup_attach_to_parent():
    @trace_step("child")
    async def child(idx: int) -> int:
        await asyncio.sleep(0)
        return idx

    @trace_step("parent")
    async def parent() -> None:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(child(1))
            tg.create_task(child(2))

    async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m") as traj:
        await parent()

    parent_step = next(s for s in traj._steps if s.name == "parent")
    child_steps = [s for s in traj._steps if s.name == "child"]
    assert len(child_steps) == 2
    for c in child_steps:
        assert c.parent_step_id == parent_step.id
