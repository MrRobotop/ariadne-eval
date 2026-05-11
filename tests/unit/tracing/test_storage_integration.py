"""End-to-end: start_trajectory + decorated calls + DuckDBStore."""

from __future__ import annotations

import pytest

from ariadne_eval.core.trajectory import Message
from ariadne_eval.storage.duckdb_store import DuckDBStore
from ariadne_eval.tracing.context import start_trajectory
from ariadne_eval.tracing.decorator import (
    record_llm_call,
    record_tool_call,
    trace_step,
)


@pytest.mark.fast
async def test_end_to_end_trajectory_persisted(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:

        @trace_step("plan")
        async def plan() -> None:
            await record_llm_call(
                model_id="claude",
                prompt_messages=[Message(role="user", content="plan it")],
                completion="step 1: search",
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.001,
                latency_ms=120.0,
            )

        @trace_step("execute")
        async def execute() -> None:
            await record_tool_call(
                tool_name="search",
                arguments={"q": "ariadne"},
                result=["hit1", "hit2"],
                latency_ms=15.0,
            )

        async with start_trajectory(
            "compute",
            agent_name="react",
            agent_version="0.1",
            model_id="claude-sonnet",
            store=store,
        ) as traj:
            await plan()
            await execute()
            traj.set_final_answer("done")
            tid = traj.id

        loaded, steps = await store.get_trajectory(tid)
        assert loaded.final_answer == "done"
        assert {s.name for s in steps} == {"plan", "execute", "llm_call", "search"}
    finally:
        await store.close()
