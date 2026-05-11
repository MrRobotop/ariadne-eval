"""The Store Protocol and the storage error hierarchy."""

from __future__ import annotations

import pytest

from ariadne_eval.storage.base import (
    MetadataTooLargeError,
    Store,
    StoreError,
    TrajectoryNotFoundError,
)


@pytest.mark.fast
def test_store_is_a_protocol():
    """Protocol classes are subclassable but not instantiable as-is."""
    for name in (
        "save_trajectory",
        "get_trajectory",
        "list_trajectories",
        "delete_trajectory",
        "count",
    ):
        assert hasattr(Store, name), f"Store.{name} missing"


@pytest.mark.fast
def test_error_hierarchy():
    assert issubclass(TrajectoryNotFoundError, StoreError)
    assert issubclass(MetadataTooLargeError, StoreError)
    assert issubclass(StoreError, Exception)


@pytest.mark.fast
def test_trajectory_not_found_carries_id():
    err = TrajectoryNotFoundError("01J...")
    assert "01J..." in str(err)


@pytest.mark.fast
def test_metadata_too_large_carries_size():
    err = MetadataTooLargeError(actual=2_000_000, max=1_048_576)
    assert "2000000" in str(err)
    assert "1048576" in str(err)
