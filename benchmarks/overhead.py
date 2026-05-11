"""Measure @trace_step overhead.

Run via pytest:
    uv run pytest benchmarks/overhead.py -v -m slow

Or directly:
    uv run python benchmarks/overhead.py
"""

from __future__ import annotations

import asyncio
import time

import pytest

from ariadne_eval.tracing.context import start_trajectory
from ariadne_eval.tracing.decorator import trace_step

_N = 1000


async def _untraced_loop() -> int:
    total = 0
    for i in range(_N):
        total += i
    return total


async def _traced_loop() -> int:
    @trace_step("inner")
    async def inner(i: int) -> int:
        return i

    total = 0
    async with start_trajectory("bench", agent_name="bench", agent_version="0", model_id="none"):
        for i in range(_N):
            total += await inner(i)
    return total


async def _measure(coro_factory, repeats: int = 5) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        await coro_factory()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]  # median


async def _untraced_loop_with_io() -> int:
    """Simulate a realistic agent step: 1ms of I/O per iteration."""
    total = 0
    for i in range(_N):
        await asyncio.sleep(0.001)
        total += i
    return total


async def _traced_loop_with_io() -> int:
    @trace_step("inner")
    async def inner(i: int) -> int:
        await asyncio.sleep(0.001)
        return i

    total = 0
    async with start_trajectory("bench", agent_name="bench", agent_version="0", model_id="none"):
        for i in range(_N):
            total += await inner(i)
    return total


async def _main() -> None:
    print("=== no-op loop (1000 iterations, no I/O) ===")
    print("Worst-case microbenchmark; not representative of real agent latency.")
    baseline = await _measure(_untraced_loop)
    traced = await _measure(_traced_loop)
    overhead = (traced - baseline) / baseline * 100
    print(f"  baseline: {baseline * 1000:.2f} ms")
    print(f"  traced:   {traced * 1000:.2f} ms")
    print(f"  overhead: {overhead:.1f}% (~{(traced - baseline) * 1e6 / _N:.1f} us per step)")
    print()
    print("=== realistic loop (1000 iterations, 1ms simulated I/O each) ===")
    print("Closer to real agent latency where LLM calls dominate.")
    baseline_io = await _measure(_untraced_loop_with_io, repeats=3)
    traced_io = await _measure(_traced_loop_with_io, repeats=3)
    overhead_io = (traced_io - baseline_io) / baseline_io * 100
    print(f"  baseline: {baseline_io * 1000:.2f} ms")
    print(f"  traced:   {traced_io * 1000:.2f} ms")
    print(f"  overhead: {overhead_io:.2f}% <-- this is the real <2% target")


@pytest.mark.slow
def test_trace_step_overhead_under_threshold():
    """Tracing 1000 I/O-bound steps adds <10% latency.

    Each step does 1ms simulated I/O (proxy for a real LLM call). With
    realistic per-step latency, the absolute per-step tracing cost
    (~5μs of Pydantic model construction) is a small fraction of the
    step time. The headline target is <2%; we assert <10% here to
    tolerate CI noise.
    """

    async def _both():
        baseline = await _measure(_untraced_loop_with_io, repeats=3)
        traced = await _measure(_traced_loop_with_io, repeats=3)
        return baseline, traced

    baseline, traced = asyncio.run(_both())
    overhead = (traced - baseline) / baseline * 100
    assert overhead < 10.0, (
        f"overhead {overhead:.2f}% exceeded 10% threshold (target <2%); "
        f"baseline={baseline * 1000:.2f}ms, traced={traced * 1000:.2f}ms"
    )


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_main())
