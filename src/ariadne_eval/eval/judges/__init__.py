"""LLM-as-judge implementations and the Judge Protocol.

These symbols are top-level public as of v0.0.8-alpha: see
``docs/concepts/calibration.md`` for the maintainer-vs-judge kappa
(κ = 0.32, fair), the 3x3 confusion matrix, and per-label
precision/recall against the 51-fixture synthetic gold set committed
at ``docs/calibration/v0.0.8-alpha-report.jsonl``.
"""

from __future__ import annotations

from ariadne_eval.eval.judges.base import Judge, JudgeParseError, JudgeVerdict
from ariadne_eval.eval.judges.stub import StubJudge
from ariadne_eval.eval.judges.trajectory_judge import TrajectoryJudge

__all__ = [
    "Judge",
    "JudgeParseError",
    "JudgeVerdict",
    "StubJudge",
    "TrajectoryJudge",
]
