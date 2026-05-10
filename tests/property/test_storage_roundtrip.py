"""Property-based: any (Trajectory, list[Step]) round-trips through DuckDBStore."""

from __future__ import annotations

import asyncio
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
from ariadne_eval.storage.duckdb_store import DuckDBStore


_BASE = datetime(2026, 1, 1, tzinfo=UTC)


@st.composite
def _payloads(draw):
    return draw(
        st.one_of(
            st.builds(
                LLMCallPayload,
                model_id=st.sampled_from(["claude-sonnet", "gpt-4o"]),
                prompt_messages=st.lists(
                    st.builds(
                        Message,
                        role=st.just("user"),
                        content=st.text(max_size=32),
                    ),
                    min_size=1,
                    max_size=2,
                ),
                completion=st.text(max_size=64),
                input_tokens=st.integers(min_value=0, max_value=1000),
                output_tokens=st.integers(min_value=0, max_value=1000),
                cost_usd=st.floats(
                    min_value=0.0,
                    max_value=1.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                latency_ms=st.floats(
                    min_value=0.0,
                    max_value=1000.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            st.builds(
                ToolCallPayload,
                tool_name=st.sampled_from(["search", "calculator"]),
                arguments=st.just({"k": "v"}),
                result=st.one_of(st.none(), st.integers(), st.text(max_size=32)),
                latency_ms=st.floats(
                    min_value=0.0,
                    max_value=1000.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            st.builds(UserInputPayload, message=st.text(max_size=32)),
            st.builds(
                InternalPayload,
                kind=st.text(min_size=1, max_size=12, alphabet="abcdef"),
            ),
        )
    )


@st.composite
def _traj_and_steps(draw):
    tid = new_id()
    started = _BASE + timedelta(
        seconds=draw(st.integers(min_value=0, max_value=10_000))
    )
    n_steps = draw(st.integers(min_value=1, max_value=4))
    steps = [
        Step(
            id=new_id(),
            trajectory_id=tid,
            parent_step_id=None,
            name=draw(st.text(min_size=1, max_size=12)),
            started_at=started + timedelta(milliseconds=i),
            finished_at=started + timedelta(milliseconds=i + 1),
            status=StepStatus.SUCCEEDED,
            payload=draw(_payloads()),
        )
        for i in range(n_steps)
    ]
    traj = Trajectory(
        id=tid,
        task=draw(st.text(min_size=1, max_size=64)),
        agent_name=draw(st.sampled_from(["react", "tool-use"])),
        agent_version="0.1",
        model_id=draw(st.sampled_from(["claude-sonnet", "gpt-4o"])),
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        final_status=draw(st.sampled_from(list(TrajectoryStatus))),
        final_answer=draw(st.one_of(st.none(), st.text(max_size=32))),
        root_step_id=steps[0].id,
    )
    return traj, steps


@pytest.mark.fast
@given(traj_and_steps=_traj_and_steps())
@settings(max_examples=50, deadline=None)
def test_storage_round_trip(traj_and_steps, tmp_path_factory):
    """Any (traj, steps) we generate is recovered identically from DuckDBStore."""
    traj, steps = traj_and_steps
    db_dir = tmp_path_factory.mktemp("hypothesis_store")
    db = db_dir / f"{traj.id}.duckdb"

    async def _run():
        store = DuckDBStore(path=db)
        try:
            await store.save_trajectory(traj, steps)
            loaded_traj, loaded_steps = await store.get_trajectory(traj.id)
            assert loaded_traj == traj
            assert loaded_steps == steps
        finally:
            await store.close()

    asyncio.run(_run())
