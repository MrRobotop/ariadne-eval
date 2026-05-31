from __future__ import annotations

import warnings

import pytest

from ariadne_eval.eval.errors import (
    BootstrapInsufficientDataWarning,
    KappaInsufficientDataWarning,
    MissingReferenceError,
)

pytestmark = pytest.mark.fast


def test_missing_reference_error_message() -> None:
    err = MissingReferenceError("expected_answer", case_id="c1")
    assert "expected_answer" in str(err)
    assert "c1" in str(err)


def test_missing_reference_error_is_value_error() -> None:
    assert issubclass(MissingReferenceError, ValueError)


def test_bootstrap_warning_is_user_warning() -> None:
    assert issubclass(BootstrapInsufficientDataWarning, UserWarning)


def test_bootstrap_warning_round_trip() -> None:
    with pytest.warns(BootstrapInsufficientDataWarning, match="n=0"):
        warnings.warn("n=0 not enough", BootstrapInsufficientDataWarning, stacklevel=1)


def test_kappa_warning_is_user_warning() -> None:
    assert issubclass(KappaInsufficientDataWarning, UserWarning)
