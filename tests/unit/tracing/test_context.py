"""TrajectoryHandle, start_trajectory, current_trajectory/step accessors."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ariadne_eval.core.status import TrajectoryStatus
from ariadne_eval.core.trajectory import Trajectory
from ariadne_eval.tracing.context import (
    TrajectoryHandle,
    current_step,
    current_trajectory,
    start_trajectory,
)


@pytest.mark.fast
async def test_start_trajectory_yields_handle_and_resets_on_exit():
    assert current_trajectory() is None
    async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m") as traj:
        assert isinstance(traj, TrajectoryHandle)
        assert traj.task == "t"
        assert current_trajectory() is traj
    assert current_trajectory() is None


@pytest.mark.fast
async def test_handle_id_is_a_ulid():
    from ariadne_eval.core.ids import is_valid_id

    async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m") as traj:
        assert is_valid_id(traj.id)


@pytest.mark.fast
async def test_handle_snapshot_succeeded():
    async with start_trajectory(
        "compute", agent_name="react", agent_version="0.1", model_id="claude-sonnet"
    ) as traj:
        traj.set_final_answer("42")
    snap = traj.snapshot(
        finished_at=datetime.now(tz=UTC),
        default_status=TrajectoryStatus.SUCCEEDED,
    )
    assert isinstance(snap, Trajectory)
    assert snap.task == "compute"
    assert snap.final_answer == "42"
    assert snap.final_status == TrajectoryStatus.SUCCEEDED


@pytest.mark.fast
async def test_handle_snapshot_respects_override():
    async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m") as traj:
        traj.set_final_status(TrajectoryStatus.ABORTED)
    snap = traj.snapshot(
        finished_at=datetime.now(tz=UTC),
        default_status=TrajectoryStatus.SUCCEEDED,
    )
    assert snap.final_status == TrajectoryStatus.ABORTED


@pytest.mark.fast
async def test_handle_add_metadata():
    async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m") as traj:
        traj.add_metadata("user", "alice")
    snap = traj.snapshot(
        finished_at=datetime.now(tz=UTC),
        default_status=TrajectoryStatus.SUCCEEDED,
    )
    assert snap.metadata["user"] == "alice"


@pytest.mark.fast
async def test_initial_metadata_passed_through():
    async with start_trajectory(
        "t",
        agent_name="a",
        agent_version="0.1",
        model_id="m",
        metadata={"k": "v"},
    ) as traj:
        snap_inside = traj.snapshot(
            finished_at=datetime.now(tz=UTC),
            default_status=TrajectoryStatus.RUNNING,
        )
        assert snap_inside.metadata["k"] == "v"


@pytest.mark.fast
async def test_exception_marks_failed_and_re_raises():
    class BoomError(Exception):
        pass

    with pytest.raises(BoomError):
        async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m") as traj:
            raise BoomError("kaboom")
    snap = traj.snapshot(
        finished_at=datetime.now(tz=UTC),
        default_status=TrajectoryStatus.SUCCEEDED,
    )
    assert snap.final_status == TrajectoryStatus.FAILED
    err = snap.metadata.get("_trajectory_error")
    assert err is not None
    assert "BoomError" in str(err)


@pytest.mark.fast
async def test_current_step_is_none_at_top():
    async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m"):
        assert current_step() is None


@pytest.mark.fast
async def test_sampler_returning_false_yields_noop_handle():
    from ariadne_eval.tracing.sampler import RateSampler

    async with start_trajectory(
        "t",
        agent_name="a",
        agent_version="0.1",
        model_id="m",
        sampler=RateSampler(rate=0.0),
    ) as traj:
        assert traj.is_noop is True
        traj.set_final_answer("ignored")
        traj.add_metadata("x", "y")
    assert traj.is_noop is True


@pytest.mark.fast
async def test_save_called_on_exit(tmp_path):
    from ariadne_eval.storage.duckdb_store import DuckDBStore

    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        async with start_trajectory(
            "t",
            agent_name="a",
            agent_version="0.1",
            model_id="m",
            store=store,
        ) as traj:
            traj.set_final_answer("ok")
            tid = traj.id
        loaded, steps = await store.get_trajectory(tid)
        assert loaded.final_answer == "ok"
        assert steps == []
    finally:
        await store.close()
