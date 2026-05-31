"""Plan-quality prompts and response parser for the trajectory judge.

Constants are pinned: changes to ``PLAN_QUALITY_SYSTEM`` /
``PLAN_QUALITY_USER_TEMPLATE`` invalidate the VCR cassette in
``tests/integration/test_trajectory_judge.py`` and require re-recording.
"""

from __future__ import annotations

import re
from typing import Final

from ariadne_eval.eval.judges.base import JudgeParseError, JudgeVerdict

__all__ = [
    "PLAN_QUALITY_SYSTEM",
    "PLAN_QUALITY_USER_TEMPLATE",
    "parse_plan_quality_verdict",
]


PLAN_QUALITY_SYSTEM: Final[str] = (
    "You are a strict but fair judge of agent plans. You score the agent's "
    "first articulated plan against three dimensions: clarity (does it state "
    "what it will do?), decomposition (is the task broken into actionable "
    "steps?), and executability (do the steps correspond to tools the agent "
    "can actually invoke?). Respond in the exact format requested."
)


PLAN_QUALITY_USER_TEMPLATE: Final[str] = """Task: {task}

Agent's plan:
{plan}

Respond using exactly this format (one field per line, in this order):

SCORE: <a float between 0.0 and 1.0>
LABEL: <one of: pass, partial, fail>
RATIONALE: <one to three sentences explaining the score>
CLARITY: <integer 1-5>
DECOMPOSITION: <integer 1-5>
EXECUTABILITY: <integer 1-5>
"""


_FIELD = re.compile(
    r"^\s*(?P<key>SCORE|LABEL|RATIONALE|CLARITY|DECOMPOSITION|EXECUTABILITY)\s*:\s*(?P<val>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_plan_quality_verdict(text: str) -> JudgeVerdict:
    """Parse a plan-quality judge response into a JudgeVerdict.

    Required fields: SCORE, LABEL, RATIONALE. Optional fields:
    CLARITY, DECOMPOSITION, EXECUTABILITY (placed into ``raw``).
    Raises ``JudgeParseError`` on missing or invalid fields.
    """
    fields: dict[str, str] = {}
    for m in _FIELD.finditer(text):
        key = m.group("key").upper()
        fields.setdefault(key, m.group("val"))

    for required in ("SCORE", "LABEL", "RATIONALE"):
        if required not in fields:
            raise JudgeParseError(f"Missing required field: {required}")

    try:
        score = float(fields["SCORE"])
    except ValueError as exc:
        raise JudgeParseError(f"SCORE is not a float: {fields['SCORE']!r}") from exc
    if not 0.0 <= score <= 1.0:
        raise JudgeParseError(f"score {score!r} out of range [0, 1]")

    label_raw = fields["LABEL"].strip().lower()
    if label_raw not in ("pass", "partial", "fail"):
        raise JudgeParseError(f"label {label_raw!r} not in {{pass, partial, fail}}")

    raw: dict[str, object] = {}
    for opt in ("CLARITY", "DECOMPOSITION", "EXECUTABILITY"):
        if opt in fields:
            try:
                raw[opt.lower()] = int(fields[opt])
            except ValueError as exc:
                raise JudgeParseError(f"{opt} is not an integer: {fields[opt]!r}") from exc

    return JudgeVerdict(
        score=score,
        label=label_raw,  # type: ignore[arg-type]
        rationale=fields["RATIONALE"].strip(),
        raw=raw,  # type: ignore[arg-type]
    )
