"""Tests for scripts/build_calibration_set.py with an injected StubJudge."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import StepStatus, TrajectoryStatus
from ariadne_eval.core.trajectory import LLMCallPayload, Message, Step, Trajectory
from ariadne_eval.eval.judges.base import JudgeVerdict
from ariadne_eval.eval.judges.stub import StubJudge
from ariadne_eval.storage.duckdb_store import DuckDBStore

pytestmark = pytest.mark.fast


_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "scripts"))


def _build_traj_and_steps(case_id: str, plan: str) -> tuple[Trajectory, list[Step]]:
    started = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
    traj_id = new_id()
    traj = Trajectory(
        id=traj_id,
        task=f"task-{case_id}",
        agent_name="test",
        agent_version="0.0.0",
        model_id="test/model",
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        final_status=TrajectoryStatus.SUCCEEDED,
        final_answer="ok",
    )
    step = Step(
        id=new_id(),
        trajectory_id=traj_id,
        parent_step_id=None,
        name="llm",
        started_at=started,
        finished_at=started + timedelta(milliseconds=10),
        status=StepStatus.SUCCEEDED,
        payload=LLMCallPayload(
            model_id="test/model",
            prompt_messages=[Message(role="user", content="hi")],
            completion=plan,
            input_tokens=10,
            output_tokens=10,
            latency_ms=10.0,
            cost_usd=0.0,
        ),
    )
    return traj, [step]


def _stub_factory_passing(*_args, **_kwargs):  # type: ignore[no-untyped-def]
    return StubJudge(
        JudgeVerdict(score=1.0, label="pass", rationale="ok"),
        name="stub",
    )


async def test_build_calibration_set_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_path = tmp_path / "store.duckdb"
    store = DuckDBStore(path=store_path)
    traj_ids: list[str] = []
    for i in range(3):
        traj, steps = _build_traj_and_steps(f"c{i}", f"plan {i}")
        await store.save_trajectory(traj, steps)
        traj_ids.append(traj.id)
    await store.close()

    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        "".join(json.dumps({"trajectory_id": tid, "label": "pass"}) + "\n" for tid in traj_ids),
        encoding="utf-8",
    )

    monkeypatch.setenv(
        "ARIADNE_TEST_JUDGE_FACTORY",
        "tests.unit.scripts.test_build_calibration_set._stub_factory_passing",
    )

    out = tmp_path / "calibration.jsonl"
    from build_calibration_set import run

    await run(
        store_path=store_path,
        gold_labels=gold,
        judge_model="test/model",
        out_path=out,
        concurrency=2,
    )

    lines = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    per = [r for r in lines if r.get("_kind") != "summary"]
    summary = [r for r in lines if r.get("_kind") == "summary"]
    assert len(per) == 3
    assert len(summary) == 1
    assert all(p["judge_label"] == "pass" for p in per)
    assert summary[0]["n"] == 3
    assert summary[0]["kappa"] == 1.0
    assert summary[0]["interpretation"] == "almost_perfect"
