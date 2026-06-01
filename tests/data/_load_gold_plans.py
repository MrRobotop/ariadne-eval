"""Loader for the synthetic plan-quality gold set.

Test-private: this module is intended to be imported by the calibration
CLI via a ``sys.path`` extension in the ``--source synth`` code path
(wired up in a subsequent task). It is NOT part of the library's public
API and lives under ``tests/`` so it ships only with tests.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import IO, Literal

from pydantic import TypeAdapter

from ariadne_eval.core.trajectory import Step, Trajectory

__all__ = ["GoldEntry", "GoldLabel", "iter_gold_plans"]


GoldLabel = Literal["pass", "partial", "fail"]


@dataclass(frozen=True)
class GoldEntry:
    """One synthetic gold-plan entry: trajectory + steps + human-assigned label."""

    trajectory: Trajectory
    steps: list[Step]
    gold_label: GoldLabel


_ADAPTER: TypeAdapter[GoldEntry] = TypeAdapter(GoldEntry)


def iter_gold_plans(stream: IO[str]) -> Iterator[GoldEntry]:
    """Yield ``GoldEntry`` per non-blank line of ``stream``.

    Each line must be a JSON object with ``trajectory``, ``steps``, and
    ``gold_label`` (one of ``"pass"``, ``"partial"``, ``"fail"``).
    Invalid shapes raise ``ValueError`` (via Pydantic validation).
    """
    for raw_line in stream:
        line = raw_line.strip()
        if not line:
            continue
        yield _ADAPTER.validate_python(json.loads(line))
