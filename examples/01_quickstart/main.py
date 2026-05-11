"""Quickstart: trace a tiny ReAct-style loop.

Run with:
    uv run python examples/01_quickstart/main.py

The trajectory is persisted to ~/.ariadne/quickstart.duckdb.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from ariadne_eval import (
    DuckDBStore,
    Message,
    record_llm_call,
    record_tool_call,
    start_trajectory,
    trace_step,
)


@trace_step("plan")
async def plan(task: str) -> str:
    """Pretend to plan the task via an LLM call."""
    await record_llm_call(
        model_id="claude-sonnet",
        prompt_messages=[Message(role="user", content=f"plan: {task}")],
        completion="step 1: compute 17*23; step 2: divide by len('banana')",
        input_tokens=20,
        output_tokens=30,
        cost_usd=0.0002,
        latency_ms=120.0,
    )
    return "step 1: compute 17*23; step 2: divide by len('banana')"


@trace_step("execute")
async def execute() -> float:
    """Pretend to execute the plan via a tool."""
    t0 = time.perf_counter()
    answer = 17 * 23 / len("banana")
    latency = (time.perf_counter() - t0) * 1000
    await record_tool_call(
        tool_name="calculator",
        arguments={"expr": "17*23/len('banana')"},
        result=answer,
        latency_ms=latency,
    )
    return answer


async def main() -> None:
    store_path = Path("~/.ariadne/quickstart.duckdb").expanduser()
    store = DuckDBStore(path=store_path)
    try:
        async with start_trajectory(
            "compute 17*23 / len('banana')",
            agent_name="quickstart",
            agent_version="0.1",
            model_id="claude-sonnet",
            store=store,
        ) as traj:
            await plan(traj.task)
            answer = await execute()
            traj.set_final_answer(answer)
            print(f"trajectory id: {traj.id}")
            print(f"final answer: {answer}")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
