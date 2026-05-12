"""Stub tools used by the reference ReAct agent."""

from __future__ import annotations

import pytest

from ariadne_eval.examples.tools import TOOLS, Tool, calculator, search


@pytest.mark.fast
def test_calculator_basic_arithmetic():
    assert calculator("17*23") == 391
    assert calculator("391/6") == pytest.approx(65.166666, abs=1e-4)
    assert calculator("2+3-1") == 4
    assert calculator("(2+3)*4") == 20
    assert calculator("2**10") == 1024


@pytest.mark.fast
def test_calculator_handles_unary_minus():
    assert calculator("-5+3") == -2


@pytest.mark.fast
@pytest.mark.parametrize(
    "unsafe",
    [
        "__import__('os')",
        "open('/etc/passwd')",
        "x + 1",
        "[1, 2, 3]",
        "1 if True else 0",
        "lambda: 1",
    ],
)
def test_calculator_rejects_non_arithmetic(unsafe):
    with pytest.raises(ValueError):
        calculator(unsafe)


@pytest.mark.fast
def test_calculator_rejects_syntax_errors():
    with pytest.raises(ValueError):
        calculator("17 *")


@pytest.mark.fast
def test_search_known_query():
    out = search("banana")
    assert "6 letters" in out


@pytest.mark.fast
def test_search_unknown_query_returns_no_results():
    assert search("zzz_nonexistent") == "No results."


@pytest.mark.fast
def test_search_strips_and_lowercases():
    assert search("  Banana  ") == search("banana")


@pytest.mark.fast
def test_tools_registry_has_both_entries():
    assert set(TOOLS.keys()) == {"calculator", "search"}
    assert all(isinstance(t, Tool) for t in TOOLS.values())
    assert TOOLS["calculator"].name == "calculator"
    assert callable(TOOLS["calculator"].fn)
