"""Tests for MetricResult and Metric Protocol."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ariadne_eval.eval.metrics.base import MetricResult

pytestmark = pytest.mark.fast


def test_metric_result_minimal() -> None:
    r = MetricResult(
        metric="m",
        case_id="c",
        trajectory_id="01J0000000000000000000000A",
        score=0.75,
    )
    assert r.label is None
    assert r.details == {}


def test_metric_result_full() -> None:
    r = MetricResult(
        metric="m",
        case_id="c",
        trajectory_id="01J0000000000000000000000A",
        score=1.0,
        label="pass",
        details={"reason": "ok"},
    )
    assert r.label == "pass"
    assert r.details["reason"] == "ok"


def test_metric_result_is_frozen() -> None:
    r = MetricResult(
        metric="m",
        case_id="c",
        trajectory_id="01J0000000000000000000000A",
        score=0.0,
    )
    with pytest.raises(ValidationError):
        r.score = 0.5  # type: ignore[misc]


def test_metric_result_label_literal_validated() -> None:
    with pytest.raises(ValidationError):
        MetricResult(
            metric="m",
            case_id="c",
            trajectory_id="01J0000000000000000000000A",
            score=0.0,
            label="bogus",  # type: ignore[arg-type]
        )
