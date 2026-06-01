"""Tests for the Phase 6 public-API surface."""

from __future__ import annotations

import pytest

import ariadne_eval
from ariadne_eval import AsyncMetric, KappaResult, cohens_kappa

pytestmark = pytest.mark.fast


def test_top_level_adds_only_async_metric_and_kappa() -> None:
    for name in ("AsyncMetric", "cohens_kappa", "KappaResult"):
        assert name in ariadne_eval.__all__
        assert getattr(ariadne_eval, name) is not None


def test_judge_symbols_promoted_in_phase6_1() -> None:
    # Phase 6 kept these namespace-private pending calibration (Hard Rule #5).
    # Phase 6.1 shipped calibration data (κ = 0.32) and promoted them.
    # This test now confirms the promotion succeeded rather than blocking it.
    for name in (
        "Judge",
        "JudgeVerdict",
        "JudgeParseError",
        "TrajectoryJudge",
        "StubJudge",
        "PlanQuality",
    ):
        assert name in ariadne_eval.__all__, (
            f"{name} must be top-level public as of Phase 6.1 (calibration shipped)"
        )


def test_judges_namespace_exports_judge_symbols() -> None:
    from ariadne_eval.eval import judges

    for name in (
        "Judge",
        "JudgeVerdict",
        "JudgeParseError",
        "TrajectoryJudge",
        "StubJudge",
    ):
        assert hasattr(judges, name), f"missing from eval.judges: {name}"


def test_eval_namespace_exposes_plan_quality_and_kappa() -> None:
    from ariadne_eval import eval as ev

    assert ev.PlanQuality is not None
    assert ev.cohens_kappa is cohens_kappa
    assert ev.KappaResult is KappaResult
    assert ev.AsyncMetric is AsyncMetric
