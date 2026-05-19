"""Evaluation: metrics, judges, statistical aggregation."""

from __future__ import annotations

from ariadne_eval.eval.case import Case, ExpectedTool
from ariadne_eval.eval.errors import (
    BootstrapInsufficientDataWarning,
    MissingReferenceError,
)
from ariadne_eval.eval.metrics.base import Metric, MetricResult
from ariadne_eval.eval.metrics.efficiency import StepEfficiency
from ariadne_eval.eval.metrics.final_answer import FinalAnswerMatch
from ariadne_eval.eval.metrics.tool_accuracy import ToolAccuracy
from ariadne_eval.eval.runner import EvalReport, Runner
from ariadne_eval.eval.stats.bootstrap import BootstrapCI, bootstrap_mean_ci

__all__ = [
    "BootstrapCI",
    "BootstrapInsufficientDataWarning",
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
]
