"""Status enums are part of the public API; their string values must not
change without a major version bump. These tests pin the values."""

import pytest

from ariadne_eval.core.status import StepStatus, TrajectoryStatus


@pytest.mark.fast
def test_step_status_string_values():
    assert StepStatus.PENDING.value == "pending"
    assert StepStatus.RUNNING.value == "running"
    assert StepStatus.SUCCEEDED.value == "succeeded"
    assert StepStatus.FAILED.value == "failed"
    assert StepStatus.SKIPPED.value == "skipped"


@pytest.mark.fast
def test_trajectory_status_string_values():
    assert TrajectoryStatus.RUNNING.value == "running"
    assert TrajectoryStatus.SUCCEEDED.value == "succeeded"
    assert TrajectoryStatus.FAILED.value == "failed"
    assert TrajectoryStatus.ABORTED.value == "aborted"


@pytest.mark.fast
def test_status_enums_are_str_enums():
    """StrEnum members must compare equal to their string value."""
    assert StepStatus.SUCCEEDED == "succeeded"
    assert TrajectoryStatus.ABORTED == "aborted"


@pytest.mark.fast
def test_step_status_full_membership():
    assert {s.value for s in StepStatus} == {
        "pending", "running", "succeeded", "failed", "skipped"
    }


@pytest.mark.fast
def test_trajectory_status_full_membership():
    assert {s.value for s in TrajectoryStatus} == {
        "running", "succeeded", "failed", "aborted"
    }
