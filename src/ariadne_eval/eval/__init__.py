"""Evaluation: metrics, judges, statistical aggregation."""

from __future__ import annotations

from ariadne_eval.eval.case import Case, ExpectedTool
from ariadne_eval.eval.errors import (
    BootstrapInsufficientDataWarning,
    KappaInsufficientDataWarning,
    MissingReferenceError,
)
from ariadne_eval.eval.metrics.base import AsyncMetric, Metric, MetricResult
from ariadne_eval.eval.metrics.efficiency import StepEfficiency
from ariadne_eval.eval.metrics.final_answer import FinalAnswerMatch
from ariadne_eval.eval.metrics.plan_quality import PlanQuality
from ariadne_eval.eval.metrics.tool_accuracy import ToolAccuracy
from ariadne_eval.eval.runner import EvalReport, Runner
from ariadne_eval.eval.stats.agreement import KappaResult, cohens_kappa
from ariadne_eval.eval.stats.bootstrap import BootstrapCI, bootstrap_mean_ci

__all__ = [
    "AsyncMetric",
    "BootstrapCI",
    "BootstrapInsufficientDataWarning",
    "Case",
    "EvalReport",
    "ExpectedTool",
    "FinalAnswerMatch",
    "KappaInsufficientDataWarning",
    "KappaResult",
    "Metric",
    "MetricResult",
    "MissingReferenceError",
    "PlanQuality",
    "Runner",
    "StepEfficiency",
    "ToolAccuracy",
    "bootstrap_mean_ci",
    "cohens_kappa",
]
