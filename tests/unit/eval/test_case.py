"""Tests for the Case sidecar model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ariadne_eval.eval.case import Case, ExpectedTool

pytestmark = pytest.mark.fast


def test_case_minimal() -> None:
    c = Case(case_id="c1", task="add 2+2")
    assert c.case_id == "c1"
    assert c.task == "add 2+2"
    assert c.expected_answer is None
    assert c.expected_tools == ()
    assert c.expected_max_steps is None
    assert c.metadata == {}


def test_case_full() -> None:
    c = Case(
        case_id="c2",
        task="search and add",
        expected_answer="4",
        expected_tools=(
            ExpectedTool(name="search", args={"q": "x"}),
            ExpectedTool(name="calculator"),
        ),
        expected_max_steps=5,
        metadata={"benchmark": "demo"},
    )
    assert c.expected_tools[1].args is None
    assert c.metadata["benchmark"] == "demo"


def test_case_is_frozen() -> None:
    c = Case(case_id="c3", task="t")
    with pytest.raises(ValidationError):
        c.task = "different"  # type: ignore[misc]


def test_expected_tool_is_frozen() -> None:
    t = ExpectedTool(name="x")
    with pytest.raises(ValidationError):
        t.name = "y"  # type: ignore[misc]
