"""Golden-file test for scripts/render_calibration_md.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast


_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "scripts"))


def test_render_calibration_md_matches_golden(tmp_path: Path) -> None:
    from render_calibration_md import render

    report = _REPO / "tests" / "data" / "_golden_calibration_report.jsonl"
    expected = (_REPO / "tests" / "data" / "_golden_calibration_expected.md").read_text(
        encoding="utf-8"
    )

    out = tmp_path / "calibration.md"
    render(report_path=report, out_path=out, report_basename="v0.0.8-alpha-report.jsonl")

    actual = out.read_text(encoding="utf-8")
    assert actual == expected
