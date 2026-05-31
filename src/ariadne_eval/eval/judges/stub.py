"""Deterministic in-process judge for tests."""

from __future__ import annotations

from collections.abc import Callable

from ariadne_eval.core.trajectory import Step, Trajectory
from ariadne_eval.eval.case import Case
from ariadne_eval.eval.judges.base import JudgeVerdict

__all__ = ["StubJudge"]

_VerdictFn = Callable[[Trajectory, list[Step], Case | None], JudgeVerdict]


class StubJudge:
    """Test-only deterministic judge.

    Construct with a fixed ``JudgeVerdict`` or a callable that returns one
    per call. Used by unit tests to exercise metric/runner plumbing
    without invoking an LLM.
    """

    name: str

    def __init__(
        self,
        verdict: JudgeVerdict | _VerdictFn,
        *,
        name: str = "stub",
    ) -> None:
        """Initialise with either a fixed verdict or a verdict-producing callable."""
        self._verdict = verdict
        self.name = name

    async def judge(
        self,
        trajectory: Trajectory,
        steps: list[Step],
        case: Case | None,
    ) -> JudgeVerdict:
        """Return the configured verdict (called or returned-as-is)."""
        if callable(self._verdict):
            return self._verdict(trajectory, steps, case)
        return self._verdict
