"""Tests for the Phase 6.1 public-API promotion."""

from __future__ import annotations

from pathlib import Path

import pytest

import ariadne_eval

pytestmark = pytest.mark.fast


_REPO = Path(__file__).resolve().parents[3]


def test_six_judge_symbols_now_top_level_public() -> None:
    for name in (
        "Judge",
        "JudgeParseError",
        "JudgeVerdict",
        "PlanQuality",
        "StubJudge",
        "TrajectoryJudge",
    ):
        assert name in ariadne_eval.__all__, f"{name!r} missing from __all__"
        assert getattr(ariadne_eval, name) is not None


def test_top_level_imports_work() -> None:
    # Doc-friendly form: ``from ariadne_eval import TrajectoryJudge``
    from ariadne_eval import (
        Judge,
        JudgeParseError,
        JudgeVerdict,
        PlanQuality,
        StubJudge,
        TrajectoryJudge,
    )

    assert Judge is not None
    assert JudgeParseError is not None
    assert JudgeVerdict is not None
    assert PlanQuality is not None
    assert StubJudge is not None
    assert TrajectoryJudge is not None


def test_calibration_report_committed() -> None:
    """Hard Rule #5: the judge promotion ships only with calibration evidence."""
    report = _REPO / "docs" / "calibration" / "v0.0.8-alpha-report.jsonl"
    assert report.exists(), (
        "Hard Rule #5 violation: judge symbols are top-level but "
        f"the calibration report is missing at {report}"
    )
    # Smoke-check the report has the expected trailing blocks
    text = report.read_text(encoding="utf-8")
    assert '"_kind": "summary"' in text or '"_kind":"summary"' in text
    assert '"_kind": "confusion"' in text or '"_kind":"confusion"' in text
    assert '"_kind": "meta"' in text or '"_kind":"meta"' in text


def test_calibration_page_committed() -> None:
    """The human-readable page exists and references the report."""
    page = _REPO / "docs" / "concepts" / "calibration.md"
    assert page.exists()
    body = page.read_text(encoding="utf-8")
    assert "Confusion matrix" in body
    assert "Per-label precision" in body
    assert "v0.0.8-alpha-report.jsonl" in body
