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


def _stub_factory_raising_parse(*_args, **_kwargs):  # type: ignore[no-untyped-def]
    from ariadne_eval.eval.judges.base import JudgeParseError

    def _raise(*_a, **_kw):  # type: ignore[no-untyped-def]
        raise JudgeParseError("synthetic")

    return StubJudge(_raise, name="raises")


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
        source="store",
        store_path=store_path,
        gold_labels=gold,
        judge_model="test/model",
        out_path=out,
        concurrency=2,
    )

    lines = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    per = [r for r in lines if "_kind" not in r]
    summary = [r for r in lines if r.get("_kind") == "summary"]
    assert len(per) == 3
    assert len(summary) == 1
    assert all(p["judge_label"] == "pass" for p in per)
    assert summary[0]["n"] == 3
    assert summary[0]["kappa"] == 1.0
    assert summary[0]["interpretation"] == "almost_perfect"


async def test_build_calibration_set_load_failure_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown trajectory_id → per-row 'error' line; summary still emitted."""
    store_path = tmp_path / "store.duckdb"
    store = DuckDBStore(path=store_path)
    await store.close()

    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps({"trajectory_id": "01J000000000000000000MISSING", "label": "pass"}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv(
        "ARIADNE_TEST_JUDGE_FACTORY",
        "tests.unit.scripts.test_build_calibration_set._stub_factory_passing",
    )

    out = tmp_path / "calibration.jsonl"
    from build_calibration_set import run

    await run(
        source="store",
        store_path=store_path,
        gold_labels=gold,
        judge_model="test/model",
        out_path=out,
        concurrency=2,
    )

    lines = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    per = [r for r in lines if "_kind" not in r]
    summary = [r for r in lines if r.get("_kind") == "summary"]
    assert len(per) == 1
    assert per[0]["error"].startswith("load: ")
    assert "judge_label" not in per[0]
    assert summary[0]["n"] == 0
    assert summary[0]["kappa"] is None


async def test_build_calibration_set_parse_error_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """JudgeParseError → per-row 'error' line (not a hard failure)."""
    store_path = tmp_path / "store.duckdb"
    store = DuckDBStore(path=store_path)
    traj, steps = _build_traj_and_steps("c0", "plan 0")
    await store.save_trajectory(traj, steps)
    await store.close()

    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps({"trajectory_id": traj.id, "label": "pass"}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv(
        "ARIADNE_TEST_JUDGE_FACTORY",
        "tests.unit.scripts.test_build_calibration_set._stub_factory_raising_parse",
    )

    out = tmp_path / "calibration.jsonl"
    from build_calibration_set import run

    await run(
        source="store",
        store_path=store_path,
        gold_labels=gold,
        judge_model="test/model",
        out_path=out,
        concurrency=2,
    )

    lines = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    per = [r for r in lines if "_kind" not in r]
    assert per[0]["error"].startswith("parse: ")


def test_resolve_test_factory_raises_on_non_dotted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ARIADNE_TEST_JUDGE_FACTORY without a module path is rejected."""
    from build_calibration_set import _resolve_test_factory

    monkeypatch.setenv("ARIADNE_TEST_JUDGE_FACTORY", "no_dots_here")
    with pytest.raises(ValueError, match="dotted path"):
        _resolve_test_factory()


def test_resolve_test_factory_returns_none_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the env var, no factory is resolved (production path)."""
    from build_calibration_set import _resolve_test_factory

    monkeypatch.delenv("ARIADNE_TEST_JUDGE_FACTORY", raising=False)
    assert _resolve_test_factory() is None


async def test_run_with_source_synth_uses_fixtures_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--source synth loads gold_plans.jsonl directly; no DuckDB needed."""
    import json as _json
    from datetime import UTC, datetime, timedelta

    from ariadne_eval.core.ids import new_id

    # Build a 1-line gold-plans JSONL inline (the production file has 51)
    traj_id = new_id()
    step_id = new_id()
    started = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
    entry = {
        "trajectory": {
            "id": traj_id,
            "task": "compute 1+2",
            "agent_name": "synth",
            "agent_version": "0.0.0",
            "model_id": "synth/agent",
            "started_at": started.isoformat().replace("+00:00", "Z"),
            "finished_at": (started + timedelta(seconds=1)).isoformat().replace("+00:00", "Z"),
            "final_status": "succeeded",
            "final_answer": "3",
        },
        "steps": [
            {
                "id": step_id,
                "trajectory_id": traj_id,
                "parent_step_id": None,
                "name": "llm",
                "started_at": started.isoformat().replace("+00:00", "Z"),
                "finished_at": (started + timedelta(milliseconds=10))
                .isoformat()
                .replace("+00:00", "Z"),
                "status": "succeeded",
                "payload": {
                    "step_type": "llm_call",
                    "model_id": "synth/agent",
                    "prompt_messages": [{"role": "user", "content": "hi"}],
                    "completion": "Step 1: use calculator(1+2).",
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "latency_ms": 1.0,
                    "cost_usd": 0.0,
                },
            }
        ],
        "gold_label": "pass",
    }
    gold = tmp_path / "gold_plans.jsonl"
    gold.write_text(_json.dumps(entry) + "\n", encoding="utf-8")

    monkeypatch.setenv(
        "ARIADNE_TEST_JUDGE_FACTORY",
        "tests.unit.scripts.test_build_calibration_set._stub_factory_passing",
    )

    out = tmp_path / "calibration.jsonl"
    from build_calibration_set import run

    await run(
        source="synth",
        store_path=None,
        gold_labels=gold,
        judge_model="test/model",
        out_path=out,
        concurrency=2,
    )

    lines = [
        _json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    per = [r for r in lines if r.get("_kind") not in ("summary", "confusion", "meta")]
    assert len(per) == 1
    assert per[0]["trajectory_id"] == traj_id
    assert per[0]["gold_label"] == "pass"
    assert per[0]["judge_label"] == "pass"


async def test_run_with_source_store_and_no_store_path_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Programmatic callers: source='store' + store_path=None raises ValueError."""
    monkeypatch.setenv(
        "ARIADNE_TEST_JUDGE_FACTORY",
        "tests.unit.scripts.test_build_calibration_set._stub_factory_passing",
    )
    gold = tmp_path / "empty.jsonl"
    gold.write_text("", encoding="utf-8")
    out = tmp_path / "calibration.jsonl"
    from build_calibration_set import run

    with pytest.raises(ValueError, match="--store"):
        await run(
            source="store",
            store_path=None,
            gold_labels=gold,
            judge_model="test/model",
            out_path=out,
            concurrency=1,
        )


def test_confusion_matrix_and_per_label_metrics() -> None:
    """Hand-computed expected on a small (gold, judge) pair list."""
    from build_calibration_set import _build_confusion_block

    # pairs: (gold, judge)
    # gold=[pass, pass, partial, fail, fail]
    # judge=[pass, fail,   pass,   fail, partial]
    # sorted label_set = [fail, partial, pass], rows=gold cols=judge
    #   fail row (gold=fail, 2 entries): (fail,fail)=1 in fail col,
    #                                    (fail,partial)=1 in partial col
    #                                    → [1, 1, 0]
    #   partial row (gold=partial, 1 entry): (partial,pass)=1 in pass col
    #                                        → [0, 0, 1]
    #   pass row (gold=pass, 2 entries): (pass,pass)=1 in pass col,
    #                                    (pass,fail)=1 in fail col
    #                                    → [1, 0, 1]
    pairs = [
        ("pass", "pass"),
        ("pass", "fail"),
        ("partial", "pass"),
        ("fail", "fail"),
        ("fail", "partial"),
    ]
    block = _build_confusion_block(pairs)
    assert block["_kind"] == "confusion"
    assert block["labels"] == ["fail", "partial", "pass"]
    assert block["matrix"] == [[1, 1, 0], [0, 0, 1], [1, 0, 1]]
    assert block["per_label"]["fail"]["support"] == 2
    assert block["per_label"]["partial"]["support"] == 1
    assert block["per_label"]["pass"]["support"] == 2
    # tp_fail=1, col_sum_fail=2 → precision=0.5
    # tp_fail=1, row_sum_fail=2 → recall=0.5
    assert block["per_label"]["fail"]["precision"] == 0.5
    assert block["per_label"]["fail"]["recall"] == 0.5
    # tp_pass=1, col_sum_pass=2, row_sum_pass=2 → both 0.5
    assert block["per_label"]["pass"]["precision"] == 0.5
    assert block["per_label"]["pass"]["recall"] == 0.5
    # tp_partial=0, col_sum_partial=1 → precision=0/1=0.0, recall=0/1=0.0
    assert block["per_label"]["partial"]["precision"] == 0.0
    assert block["per_label"]["partial"]["recall"] == 0.0


def test_confusion_block_with_empty_pairs_is_safe() -> None:
    """No judged pairs → empty matrix, empty per_label, label_set empty."""
    from build_calibration_set import _build_confusion_block

    block = _build_confusion_block([])
    assert block == {
        "_kind": "confusion",
        "labels": [],
        "matrix": [],
        "per_label": {},
    }


def test_meta_block_shape_and_prompt_hashes() -> None:
    """The meta block carries judge config + prompt hashes + ariadne version."""
    import hashlib

    from build_calibration_set import _build_meta_block

    from ariadne_eval.eval.judges.prompts import (
        PLAN_QUALITY_SYSTEM,
        PLAN_QUALITY_USER_TEMPLATE,
    )

    block = _build_meta_block(
        judge_model="anthropic/claude-sonnet-4-6",
        temperature=0.0,
        run_date="2026-05-31",
        n_gold=51,
    )
    assert block["_kind"] == "meta"
    assert block["judge_model"] == "anthropic/claude-sonnet-4-6"
    assert block["temperature"] == 0.0
    assert block["run_date"] == "2026-05-31"
    assert block["n_gold"] == 51
    assert (
        block["system_prompt_sha256"]
        == hashlib.sha256(PLAN_QUALITY_SYSTEM.encode("utf-8")).hexdigest()
    )
    assert (
        block["user_template_sha256"]
        == hashlib.sha256(PLAN_QUALITY_USER_TEMPLATE.encode("utf-8")).hexdigest()
    )
    from ariadne_eval import __version__

    assert block["ariadne_version"] == __version__
