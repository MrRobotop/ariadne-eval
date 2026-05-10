"""JSONL export / import functions."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import StepStatus, TrajectoryStatus
from ariadne_eval.core.trajectory import (
    LLMCallPayload,
    Message,
    Step,
    Trajectory,
)
from ariadne_eval.storage.duckdb_store import DuckDBStore
from ariadne_eval.storage.jsonl_store import export_jsonl, import_jsonl


def _make_traj(agent: str = "react") -> tuple[Trajectory, list[Step]]:
    tid = new_id()
    started = datetime.now(tz=UTC)
    s1 = Step(
        id=new_id(),
        trajectory_id=tid,
        parent_step_id=None,
        name="ask",
        started_at=started,
        finished_at=started,
        status=StepStatus.SUCCEEDED,
        payload=LLMCallPayload(
            model_id="m",
            prompt_messages=[Message(role="user", content="hi")],
            completion="hello",
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
            latency_ms=1.0,
        ),
    )
    traj = Trajectory(
        id=tid,
        task="t",
        agent_name=agent,
        agent_version="0.1",
        model_id="m",
        started_at=started,
        finished_at=started,
        final_status=TrajectoryStatus.SUCCEEDED,
        root_step_id=s1.id,
    )
    return traj, [s1]


@pytest.mark.fast
async def test_export_then_import_round_trip(tmp_path):
    src = DuckDBStore(path=tmp_path / "src.duckdb")
    dst = DuckDBStore(path=tmp_path / "dst.duckdb")
    try:
        for _ in range(3):
            t, s = _make_traj()
            await src.save_trajectory(t, s)

        out = tmp_path / "dump.jsonl"
        n = await export_jsonl(src, out)
        assert n == 3

        imported = await import_jsonl(out, dst)
        assert imported == 3
        assert await dst.count() == 3
    finally:
        await src.close()
        await dst.close()


@pytest.mark.fast
async def test_export_honours_filter_kwargs(tmp_path):
    src = DuckDBStore(path=tmp_path / "src.duckdb")
    try:
        for _ in range(2):
            t, s = _make_traj(agent="react")
            await src.save_trajectory(t, s)
        for _ in range(3):
            t, s = _make_traj(agent="tool-use")
            await src.save_trajectory(t, s)

        out = tmp_path / "react_only.jsonl"
        n = await export_jsonl(src, out, agent_name="react")
        assert n == 2

        lines = out.read_text().splitlines()
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert obj["trajectory"]["agent_name"] == "react"
    finally:
        await src.close()


@pytest.mark.fast
async def test_export_format_has_trajectory_and_steps_keys(tmp_path):
    src = DuckDBStore(path=tmp_path / "src.duckdb")
    try:
        t, s = _make_traj()
        await src.save_trajectory(t, s)

        out = tmp_path / "dump.jsonl"
        await export_jsonl(src, out)

        line = out.read_text().splitlines()[0]
        obj = json.loads(line)
        assert set(obj.keys()) == {"trajectory", "steps"}
        assert isinstance(obj["steps"], list)
    finally:
        await src.close()


@pytest.mark.fast
async def test_import_rejects_corrupt_json_with_line_number(tmp_path):
    dst = DuckDBStore(path=tmp_path / "dst.duckdb")
    try:
        bad = tmp_path / "bad.jsonl"
        # Write a valid trajectory on line 1, corrupt JSON on line 2.
        t, s = _make_traj()
        good_line = json.dumps(
            {
                "trajectory": json.loads(t.model_dump_json()),
                "steps": [json.loads(step.model_dump_json()) for step in s],
            }
        )
        bad.write_text(good_line + "\nNOT VALID JSON\n")
        with pytest.raises(ValueError) as exc:
            await import_jsonl(bad, dst)
        assert "line 2" in str(exc.value).lower()
    finally:
        await dst.close()


@pytest.mark.fast
async def test_import_rejects_missing_keys_with_line_number(tmp_path):
    dst = DuckDBStore(path=tmp_path / "dst.duckdb")
    try:
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"not_a_trajectory": true}\n')
        with pytest.raises(ValueError) as exc:
            await import_jsonl(bad, dst)
        assert "line 1" in str(exc.value).lower()
    finally:
        await dst.close()
