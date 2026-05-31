"""Tests for the plan-quality prompt parser."""

from __future__ import annotations

import pytest

from ariadne_eval.eval.judges.base import JudgeParseError
from ariadne_eval.eval.judges.prompts import (
    PLAN_QUALITY_SYSTEM,
    PLAN_QUALITY_USER_TEMPLATE,
    parse_plan_quality_verdict,
)

pytestmark = pytest.mark.fast


def test_prompt_constants_present() -> None:
    assert isinstance(PLAN_QUALITY_SYSTEM, str) and len(PLAN_QUALITY_SYSTEM) > 0
    assert "{task}" in PLAN_QUALITY_USER_TEMPLATE
    assert "{plan}" in PLAN_QUALITY_USER_TEMPLATE


def test_parse_happy_full() -> None:
    text = """SCORE: 0.8
LABEL: pass
RATIONALE: The plan decomposes the task into clear, executable steps.
CLARITY: 4
DECOMPOSITION: 5
EXECUTABILITY: 4"""
    v = parse_plan_quality_verdict(text)
    assert v.score == 0.8
    assert v.label == "pass"
    assert "decomposes" in v.rationale
    assert v.raw == {"clarity": 4, "decomposition": 5, "executability": 4}


def test_parse_happy_minimal() -> None:
    text = "SCORE: 0.3\nLABEL: partial\nRATIONALE: lacks decomposition"
    v = parse_plan_quality_verdict(text)
    assert v.score == 0.3
    assert v.label == "partial"
    assert v.rationale == "lacks decomposition"
    assert v.raw == {}


def test_parse_case_insensitive_keys() -> None:
    text = "score: 1.0\nlabel: pass\nrationale: ok"
    v = parse_plan_quality_verdict(text)
    assert v.score == 1.0
    assert v.label == "pass"


def test_parse_strips_whitespace_and_extra_lines() -> None:
    text = """

SCORE: 0.5
LABEL: partial
RATIONALE: meh

extra trailing content ignored
"""
    v = parse_plan_quality_verdict(text)
    assert v.score == 0.5
    assert v.rationale == "meh"


def test_parse_missing_score_raises() -> None:
    with pytest.raises(JudgeParseError, match="SCORE"):
        parse_plan_quality_verdict("LABEL: pass\nRATIONALE: x")


def test_parse_missing_label_raises() -> None:
    with pytest.raises(JudgeParseError, match="LABEL"):
        parse_plan_quality_verdict("SCORE: 0.5\nRATIONALE: x")


def test_parse_missing_rationale_raises() -> None:
    with pytest.raises(JudgeParseError, match="RATIONALE"):
        parse_plan_quality_verdict("SCORE: 0.5\nLABEL: pass")


def test_parse_out_of_range_score_raises() -> None:
    with pytest.raises(JudgeParseError, match="score"):
        parse_plan_quality_verdict("SCORE: 1.5\nLABEL: pass\nRATIONALE: x")


def test_parse_bad_label_raises() -> None:
    with pytest.raises(JudgeParseError, match="label"):
        parse_plan_quality_verdict("SCORE: 0.5\nLABEL: maybe\nRATIONALE: x")
