"""Tests for the transient-error retry helpers."""

from __future__ import annotations

import pytest

from ariadne_eval._transient import (
    MAX_TRANSIENT_RETRIES,
    TRANSIENT_BACKOFF_BASE,
    TRANSIENT_EXC_NAMES,
    is_transient,
)

pytestmark = pytest.mark.fast


def test_constants_have_expected_values() -> None:
    assert MAX_TRANSIENT_RETRIES == 4
    assert TRANSIENT_BACKOFF_BASE == 2.0
    assert set(TRANSIENT_EXC_NAMES) == {
        "InternalServerError",
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "ServiceUnavailableError",
    }


def test_is_transient_recognizes_known_classes() -> None:
    class InternalServerError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    assert is_transient(InternalServerError("oops"))
    assert is_transient(RateLimitError("slow down"))


def test_is_transient_rejects_unrelated_exceptions() -> None:
    assert not is_transient(ValueError("nope"))
    assert not is_transient(RuntimeError("nope"))
    assert not is_transient(KeyError("nope"))


def test_is_transient_only_looks_at_class_name_not_inheritance() -> None:
    """Provider-portable identification — by class name string."""

    class InternalServerError(Exception):
        pass

    class WeirdSubclassError(InternalServerError):
        pass

    # The subclass has a different name — it's NOT recognized as transient
    # unless we explicitly add it. This is intentional: provider class names
    # are stable contracts; subclasses are not.
    assert is_transient(InternalServerError("a"))
    assert not is_transient(WeirdSubclassError("b"))
