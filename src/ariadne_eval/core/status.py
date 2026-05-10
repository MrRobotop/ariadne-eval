"""Status enums for trajectories and steps.

These are part of the public API. Renaming a member or changing a string
value is a major version bump.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["StepStatus", "TrajectoryStatus"]


class StepStatus(StrEnum):
    """Lifecycle status of a single ``Step`` in a trajectory."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class TrajectoryStatus(StrEnum):
    """Lifecycle status of an end-to-end ``Trajectory``."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"
