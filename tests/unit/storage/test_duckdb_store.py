"""End-to-end DuckDBStore tests."""

from __future__ import annotations

import duckdb
import pytest

from ariadne_eval.storage.duckdb_store import DuckDBStore


@pytest.mark.fast
async def test_init_creates_file_and_runs_migrations(tmp_path):
    db = tmp_path / "store.duckdb"
    store = DuckDBStore(path=db)
    try:
        assert db.exists()
        conn = duckdb.connect(str(db))
        try:
            rows = conn.execute("SELECT version, name FROM _meta ORDER BY version").fetchall()
            assert (1, "initial") in rows
        finally:
            conn.close()
    finally:
        await store.close()


@pytest.mark.fast
async def test_init_creates_parent_directory(tmp_path):
    db = tmp_path / "nested" / "subdir" / "store.duckdb"
    store = DuckDBStore(path=db)
    try:
        assert db.exists()
        assert db.parent.is_dir()
    finally:
        await store.close()


@pytest.mark.fast
async def test_constructor_path_overrides_env_var(tmp_path, monkeypatch):
    env_db = tmp_path / "env.duckdb"
    arg_db = tmp_path / "arg.duckdb"
    monkeypatch.setenv("ARIADNE_STORE_PATH", str(env_db))

    store = DuckDBStore(path=arg_db)
    try:
        assert arg_db.exists()
        assert not env_db.exists()
    finally:
        await store.close()


@pytest.mark.fast
async def test_env_var_used_when_no_arg(tmp_path, monkeypatch):
    db = tmp_path / "env.duckdb"
    monkeypatch.setenv("ARIADNE_STORE_PATH", str(db))
    store = DuckDBStore()
    try:
        assert db.exists()
    finally:
        await store.close()


@pytest.mark.fast
async def test_path_property_returns_resolved_path(tmp_path):
    db = tmp_path / "s.duckdb"
    store = DuckDBStore(path=db)
    try:
        assert store.path == db
    finally:
        await store.close()


@pytest.mark.fast
async def test_close_is_idempotent(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    await store.close()
    # Calling again should not raise
    await store.close()


@pytest.mark.fast
async def test_default_path_is_under_dot_ariadne(monkeypatch, tmp_path):
    """When no path or env var is set, default is ~/.ariadne/store.duckdb."""
    monkeypatch.delenv("ARIADNE_STORE_PATH", raising=False)
    fake_home = tmp_path / "fake_home"
    monkeypatch.setenv("HOME", str(fake_home))

    store = DuckDBStore()
    try:
        expected = fake_home / ".ariadne" / "store.duckdb"
        assert expected.exists()
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Task 5 — save_trajectory + get_trajectory round-trip
# ---------------------------------------------------------------------------

from datetime import UTC, datetime  # noqa: E402

from ariadne_eval.core.ids import new_id  # noqa: E402
from ariadne_eval.core.status import StepStatus, TrajectoryStatus  # noqa: E402
from ariadne_eval.core.trajectory import (  # noqa: E402
    LLMCallPayload,
    Message,
    Step,
    Trajectory,
)
from ariadne_eval.storage.base import (  # noqa: E402
    MetadataTooLargeError,
    TrajectoryNotFoundError,
)


def _llm_payload() -> LLMCallPayload:
    return LLMCallPayload(
        model_id="claude-sonnet",
        prompt_messages=[Message(role="user", content="hi")],
        completion="hello",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        latency_ms=1.0,
    )


def _make_traj_with_steps(traj_id: str | None = None) -> tuple[Trajectory, list[Step]]:
    tid = traj_id or new_id()
    started = datetime.now(tz=UTC)
    s1 = Step(
        id=new_id(),
        trajectory_id=tid,
        parent_step_id=None,
        name="ask_llm",
        started_at=started,
        finished_at=started,
        status=StepStatus.SUCCEEDED,
        payload=_llm_payload(),
    )
    traj = Trajectory(
        id=tid,
        task="2+2",
        agent_name="react",
        agent_version="0.1",
        model_id="claude-sonnet",
        started_at=started,
        finished_at=started,
        final_status=TrajectoryStatus.SUCCEEDED,
        final_answer="4",
        root_step_id=s1.id,
    )
    return traj, [s1]


@pytest.mark.fast
async def test_save_then_get_round_trip(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        traj, steps = _make_traj_with_steps()
        await store.save_trajectory(traj, steps)

        loaded_traj, loaded_steps = await store.get_trajectory(traj.id)
        assert loaded_traj == traj
        assert loaded_steps == steps
    finally:
        await store.close()


@pytest.mark.fast
async def test_get_trajectory_raises_on_missing(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        with pytest.raises(TrajectoryNotFoundError) as exc:
            await store.get_trajectory("01ARZ3NDEKTSV4RRFFQ69G5FAV")
        assert "01ARZ3NDEKTSV4RRFFQ69G5FAV" in str(exc.value)
    finally:
        await store.close()


@pytest.mark.fast
async def test_save_rejects_metadata_over_1mb(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        traj, steps = _make_traj_with_steps()
        traj = traj.model_copy(update={"metadata": {"big": "x" * 1_100_000}})
        with pytest.raises(MetadataTooLargeError) as exc:
            await store.save_trajectory(traj, steps)
        assert "1048576" in str(exc.value)
    finally:
        await store.close()


@pytest.mark.fast
async def test_save_replaces_on_same_id(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        tid = new_id()
        traj1, steps1 = _make_traj_with_steps(traj_id=tid)
        await store.save_trajectory(traj1, steps1)

        traj2, steps2 = _make_traj_with_steps(traj_id=tid)
        traj2 = traj2.model_copy(update={"task": "REVISED TASK"})
        await store.save_trajectory(traj2, steps2)

        loaded_traj, loaded_steps = await store.get_trajectory(tid)
        assert loaded_traj.task == "REVISED TASK"
        assert {s.id for s in loaded_steps} == {s.id for s in steps2}
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Task 6 — list_trajectories + count
# ---------------------------------------------------------------------------

from datetime import timedelta  # noqa: E402


async def _seed(store, *, n=5, agent="react", model="m"):
    base = datetime(2026, 1, 1, tzinfo=UTC)
    out = []
    for i in range(n):
        traj, steps = _make_traj_with_steps()
        traj = traj.model_copy(
            update={
                "agent_name": agent,
                "model_id": model,
                "started_at": base + timedelta(seconds=i),
            }
        )
        await store.save_trajectory(traj, steps)
        out.append(traj)
    return out


@pytest.mark.fast
async def test_list_returns_trajectories_most_recent_first(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        seeded = await _seed(store, n=5)
        listed = await store.list_trajectories()
        assert [t.id for t in listed] == [t.id for t in reversed(seeded)]
    finally:
        await store.close()


@pytest.mark.fast
async def test_list_filters_by_agent_name(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        await _seed(store, n=2, agent="react")
        await _seed(store, n=3, agent="tool-use")
        listed = await store.list_trajectories(agent_name="react")
        assert len(listed) == 2
        assert all(t.agent_name == "react" for t in listed)
    finally:
        await store.close()


@pytest.mark.fast
async def test_list_filters_by_model_id(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        await _seed(store, n=2, model="claude-sonnet")
        await _seed(store, n=4, model="gpt-4o")
        listed = await store.list_trajectories(model_id="gpt-4o")
        assert len(listed) == 4
    finally:
        await store.close()


@pytest.mark.fast
async def test_list_filters_by_final_status(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        await _seed(store, n=2)
        traj, steps = _make_traj_with_steps()
        traj = traj.model_copy(update={"final_status": TrajectoryStatus.FAILED})
        await store.save_trajectory(traj, steps)

        succ = await store.list_trajectories(final_status=TrajectoryStatus.SUCCEEDED)
        fail = await store.list_trajectories(final_status=TrajectoryStatus.FAILED)
        assert len(succ) == 2
        assert len(fail) == 1
    finally:
        await store.close()


@pytest.mark.fast
async def test_list_filters_by_time_range(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        seeded = await _seed(store, n=5)
        after = seeded[1].started_at
        before = seeded[3].started_at
        listed = await store.list_trajectories(started_after=after, started_before=before)
        assert {t.id for t in listed} == {t.id for t in seeded[1:4]}
    finally:
        await store.close()


@pytest.mark.fast
async def test_list_pagination(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        await _seed(store, n=7)
        page1 = await store.list_trajectories(limit=3, offset=0)
        page2 = await store.list_trajectories(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        assert {t.id for t in page1}.isdisjoint({t.id for t in page2})
    finally:
        await store.close()


@pytest.mark.fast
async def test_count_total_and_filtered(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        await _seed(store, n=2, agent="react")
        await _seed(store, n=3, agent="tool-use")
        assert await store.count() == 5
        assert await store.count(agent_name="react") == 2
        assert await store.count(agent_name="tool-use") == 3
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Task 7 — delete_trajectory
# ---------------------------------------------------------------------------


@pytest.mark.fast
async def test_delete_removes_trajectory_and_steps(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        traj, steps = _make_traj_with_steps()
        await store.save_trajectory(traj, steps)
        await store.delete_trajectory(traj.id)
        with pytest.raises(TrajectoryNotFoundError):
            await store.get_trajectory(traj.id)
        assert await store.count() == 0
    finally:
        await store.close()


@pytest.mark.fast
async def test_delete_is_idempotent_on_missing(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        await store.delete_trajectory("01ARZ3NDEKTSV4RRFFQ69G5FAV")
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Task 8 — concurrent writes are serialized by the per-instance lock
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402


@pytest.mark.fast
async def test_50_parallel_saves_all_land(tmp_path):
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        pairs = [_make_traj_with_steps() for _ in range(50)]
        await asyncio.gather(*(store.save_trajectory(t, s) for t, s in pairs))

        assert await store.count() == 50
        ids = {t.id for t, _ in pairs}
        listed = await store.list_trajectories(limit=100)
        assert {t.id for t in listed} == ids
    finally:
        await store.close()
