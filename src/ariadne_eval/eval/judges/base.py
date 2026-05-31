"""Judge Protocol, verdict shape, and parser-error type."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ariadne_eval.core.trajectory import JsonValue, Step, Trajectory
from ariadne_eval.eval.case import Case

__all__ = ["Judge", "JudgeParseError", "JudgeVerdict"]


class JudgeVerdict(BaseModel):
    """Output of a Judge for a single trajectory.

    ``score`` is a float in ``[0, 1]``. ``label`` is the categorical
    judgment used for kappa computation. ``rationale`` is the judge's
    short explanation. ``raw`` carries optional structured sub-fields
    (e.g. per-dimension rubric scores).
    """

    model_config = {"frozen": True}

    score: float
    label: Literal["pass", "partial", "fail"]
    rationale: str
    raw: dict[str, JsonValue] = Field(default_factory=dict)


class JudgeParseError(ValueError):
    """Raised when a judge's textual response cannot be parsed into a verdict.

    Not caught by the Runner — judge bugs fail loud. The calibration
    script (``scripts/build_calibration_set.py``) is the one place this
    is caught and logged per-trajectory.
    """


@runtime_checkable
class Judge(Protocol):
    """Async judging contract over a full trajectory."""

    name: str

    async def judge(
        self,
        trajectory: Trajectory,
        steps: list[Step],
        case: Case | None,
    ) -> JudgeVerdict:
        """Produce a verdict for the trajectory."""
        ...
