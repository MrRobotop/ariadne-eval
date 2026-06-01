"""Tests for the synthetic gold-plans loader."""

from __future__ import annotations

from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Literal

import pytest

from ariadne_eval.core.trajectory import LLMCallPayload, Step, Trajectory
from tests.data._load_gold_plans import GoldEntry, iter_gold_plans

_GOLD_PLANS_PATH = Path(__file__).resolve().parents[2] / "data" / "gold_plans.jsonl"

pytestmark = pytest.mark.fast


_ONE_FIXTURE_JSONL = """\
{"trajectory":{"id":"01J0000000000000000000000A","task":"add 1+2","agent_name":"synth","agent_version":"0.0.0","model_id":"synth/agent","started_at":"2026-05-31T00:00:00Z","finished_at":"2026-05-31T00:00:01Z","final_status":"succeeded","final_answer":"3"},"steps":[{"id":"01J0000000000000000000000B","trajectory_id":"01J0000000000000000000000A","parent_step_id":null,"name":"llm","started_at":"2026-05-31T00:00:00Z","finished_at":"2026-05-31T00:00:00.010000Z","status":"succeeded","payload":{"step_type":"llm_call","model_id":"synth/agent","prompt_messages":[{"role":"user","content":"hi"}],"completion":"I will use the calculator on 1+2.","input_tokens":1,"output_tokens":1,"latency_ms":1.0,"cost_usd":0.0}}],"gold_label":"pass"}
"""  # noqa: E501


def test_load_single_entry_from_stream() -> None:
    entries = list(iter_gold_plans(StringIO(_ONE_FIXTURE_JSONL)))
    assert len(entries) == 1
    e = entries[0]
    assert isinstance(e, GoldEntry)
    assert isinstance(e.trajectory, Trajectory)
    assert len(e.steps) == 1
    assert isinstance(e.steps[0], Step)
    assert e.gold_label == "pass"
    assert e.trajectory.task == "add 1+2"


def test_load_skips_blank_lines() -> None:
    stream = StringIO(_ONE_FIXTURE_JSONL + "\n   \n" + _ONE_FIXTURE_JSONL)
    entries = list(iter_gold_plans(stream))
    assert len(entries) == 2


def test_load_rejects_bad_label() -> None:
    bad = _ONE_FIXTURE_JSONL.replace('"gold_label":"pass"', '"gold_label":"maybe"')
    with pytest.raises(ValueError, match="gold_label"):
        list(iter_gold_plans(StringIO(bad)))


def test_load_rejects_missing_trajectory() -> None:
    bad = '{"steps":[],"gold_label":"pass"}\n'
    with pytest.raises(ValueError):
        list(iter_gold_plans(StringIO(bad)))


def test_gold_entry_label_type() -> None:
    # Compile-time-ish check that GoldEntry.gold_label is the right Literal
    e = next(iter_gold_plans(StringIO(_ONE_FIXTURE_JSONL)))
    lbl: Literal["pass", "partial", "fail"] = e.gold_label
    assert lbl in ("pass", "partial", "fail")


def test_gold_plans_file_loads_completely() -> None:
    with _GOLD_PLANS_PATH.open(encoding="utf-8") as f:
        entries = list(iter_gold_plans(f))
    assert len(entries) == 51


def test_gold_plans_balanced_buckets() -> None:
    with _GOLD_PLANS_PATH.open(encoding="utf-8") as f:
        entries = list(iter_gold_plans(f))
    counts = Counter(e.gold_label for e in entries)
    assert counts == {"pass": 17, "partial": 17, "fail": 17}


def test_gold_plans_unique_trajectory_ids() -> None:
    with _GOLD_PLANS_PATH.open(encoding="utf-8") as f:
        entries = list(iter_gold_plans(f))
    ids = [e.trajectory.id for e in entries]
    assert len(set(ids)) == len(ids), "trajectory IDs must be unique"


def test_gold_plans_have_an_llm_step() -> None:
    """Every fixture must have at least one LLM step so PlanQuality can score it."""
    with _GOLD_PLANS_PATH.open(encoding="utf-8") as f:
        entries = list(iter_gold_plans(f))
    for e in entries:
        has_llm = any(isinstance(s.payload, LLMCallPayload) for s in e.steps)
        assert has_llm, f"entry {e.trajectory.id} has no LLM step"
