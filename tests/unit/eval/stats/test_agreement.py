"""Tests for Cohen's kappa (judge-human agreement)."""

from __future__ import annotations

import math

import pytest

from ariadne_eval.eval.errors import KappaInsufficientDataWarning
from ariadne_eval.eval.stats.agreement import KappaResult, _interpret, cohens_kappa

pytestmark = pytest.mark.fast


def test_perfect_agreement_kappa_one() -> None:
    a = ["pass", "fail", "partial", "pass"]
    r = cohens_kappa(a, a)
    assert isinstance(r, KappaResult)
    assert r.kappa == 1.0
    assert r.n == 4
    assert r.interpretation == "almost_perfect"


def test_known_kappa_value() -> None:
    """Sanity check against a hand-computed example.

    a = [pass, pass, fail, fail, pass]
    b = [pass, fail, fail, pass, pass]
    p_o = 0.6; p_e = 0.52; kappa = 0.08 / 0.48 ≈ 0.1667
    """
    a = ["pass", "pass", "fail", "fail", "pass"]
    b = ["pass", "fail", "fail", "pass", "pass"]
    r = cohens_kappa(a, b)
    assert abs(r.kappa - (0.08 / 0.48)) < 1e-9
    assert r.interpretation == "slight"
    assert r.n == 5
    assert set(r.label_set) == {"pass", "fail"}


def test_complete_disagreement_two_labels() -> None:
    a = ["pass", "fail", "pass", "fail"]
    b = ["fail", "pass", "fail", "pass"]
    r = cohens_kappa(a, b)
    assert abs(r.kappa - (-1.0)) < 1e-9
    assert r.interpretation == "poor"


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        cohens_kappa(["pass"], ["pass", "fail"])


def test_empty_inputs_warn_and_nan() -> None:
    with pytest.warns(KappaInsufficientDataWarning):
        r = cohens_kappa([], [])
    assert r.n == 0
    assert math.isnan(r.kappa)
    assert r.interpretation == "poor"
    assert r.label_set == ()


def test_single_label_degenerate_all_agree() -> None:
    r = cohens_kappa(["pass", "pass", "pass"], ["pass", "pass", "pass"])
    assert r.kappa == 1.0
    assert r.interpretation == "almost_perfect"


def test_interpret_covers_all_bands() -> None:
    """Landis-Koch boundaries hit each band of the interpretation function."""
    assert _interpret(-0.1) == "poor"
    assert _interpret(math.nan) == "poor"
    assert _interpret(0.0) == "slight"
    assert _interpret(0.19) == "slight"
    assert _interpret(0.2) == "fair"
    assert _interpret(0.39) == "fair"
    assert _interpret(0.4) == "moderate"
    assert _interpret(0.59) == "moderate"
    assert _interpret(0.6) == "substantial"
    assert _interpret(0.79) == "substantial"
    assert _interpret(0.8) == "almost_perfect"
    assert _interpret(1.0) == "almost_perfect"


def test_explicit_label_set_constrains_distribution() -> None:
    """Passing labels=(...) widens the chance-agreement denominator."""
    a = ["pass", "pass"]
    b = ["pass", "pass"]
    r = cohens_kappa(a, b, labels=("pass", "fail", "partial"))
    assert r.kappa == 1.0
    assert r.label_set == ("pass", "fail", "partial")
