"""Tests for the public re-export surface of ``ariadne_eval``."""

from __future__ import annotations

import pytest

import ariadne_eval
from ariadne_eval import (  # noqa: F401
    BootstrapCI,
    Case,
    EvalReport,
    ExpectedTool,
    FinalAnswerMatch,
    Metric,
    MetricResult,
    MissingReferenceError,
    Runner,
    StepEfficiency,
    ToolAccuracy,
    bootstrap_mean_ci,
)

pytestmark = pytest.mark.fast


def test_top_level_exports_resolve() -> None:
    for name in [
        "BootstrapCI",
        "Case",
        "EvalReport",
        "ExpectedTool",
        "FinalAnswerMatch",
        "Metric",
        "MetricResult",
        "MissingReferenceError",
        "Runner",
        "StepEfficiency",
        "ToolAccuracy",
        "bootstrap_mean_ci",
    ]:
        assert name in ariadne_eval.__all__, f"missing from __all__: {name}"
        assert getattr(ariadne_eval, name) is not None


def test_namespaced_eval_module_also_exposes_them() -> None:
    from ariadne_eval import eval as ev

    assert ev.Case is Case
    assert ev.Runner is Runner
    assert ev.bootstrap_mean_ci is bootstrap_mean_ci
