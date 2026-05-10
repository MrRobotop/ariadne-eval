"""ULID-based IDs: time-sortable, unique, validatable."""

import time

import pytest

from ariadne_eval.core.ids import is_valid_id, new_id


@pytest.mark.fast
def test_new_id_returns_26_char_string():
    out = new_id()
    assert isinstance(out, str)
    assert len(out) == 26


@pytest.mark.fast
def test_new_id_is_unique_over_10k_samples():
    seen = {new_id() for _ in range(10_000)}
    assert len(seen) == 10_000


@pytest.mark.fast
def test_new_id_is_time_sortable_across_milliseconds():
    """ULIDs encode the timestamp in their first 10 chars; IDs minted in
    monotonically later milliseconds must sort lexicographically later."""
    earlier = new_id()
    time.sleep(0.005)  # 5ms — guaranteed boundary across ms
    later = new_id()
    assert earlier < later


@pytest.mark.fast
def test_is_valid_id_accepts_freshly_minted_id():
    assert is_valid_id(new_id()) is True


@pytest.mark.fast
@pytest.mark.parametrize(
    "bad",
    [
        "",
        "too-short",
        "x" * 25,  # 25 chars
        "x" * 27,  # 27 chars
        "01ARZ3NDEKTSV4RRFFQ69G5FA!",  # invalid char
        "01ARZ3NDEKTSV4RRFFQ69G5FAU",  # contains 'U' — Crockford excludes I,L,O,U
    ],
)
def test_is_valid_id_rejects_malformed(bad):
    assert is_valid_id(bad) is False


@pytest.mark.fast
def test_is_valid_id_handles_non_string():
    """Defensive: callers pass us values from JSON; non-strings are False, not raise."""
    assert is_valid_id(None) is False
    assert is_valid_id(123) is False
