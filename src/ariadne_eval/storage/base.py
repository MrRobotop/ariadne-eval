"""Storage protocol and error hierarchy.

The ``Store`` Protocol defines the abstract storage interface every concrete
backend (currently only ``DuckDBStore``) must satisfy. Tests that exercise
storage-agnostic code can mock against this Protocol directly.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ariadne_eval.core.status import TrajectoryStatus
    from ariadne_eval.core.trajectory import Step, Trajectory

__all__ = [
    "MetadataTooLargeError",
    "Store",
    "StoreError",
    "TrajectoryNotFoundError",
]


class StoreError(Exception):
    """Base class for all storage-layer errors."""


class TrajectoryNotFoundError(StoreError):
    """Raised by ``get_trajectory`` when no row matches the given id."""

    def __init__(self, traj_id: str) -> None:
        """Build the error with the missing trajectory id."""
        super().__init__(f"trajectory not found: {traj_id!r}")
        self.traj_id = traj_id


class MetadataTooLargeError(StoreError):
    """Raised when a trajectory or step's serialized metadata exceeds the cap."""

    def __init__(self, *, actual: int, max: int) -> None:
        """Build the error with the offending and allowed byte sizes."""
        super().__init__(f"metadata is {actual} bytes, max {max} bytes")
        self.actual = actual
        self.max = max


class Store(Protocol):
    """Abstract storage backend for trajectories and their step trees."""

    async def save_trajectory(self, traj: Trajectory, steps: list[Step]) -> None:
        """Persist a trajectory and its steps. Upserts on the same id."""
        ...

    async def get_trajectory(self, traj_id: str) -> tuple[Trajectory, list[Step]]:
        """Load a trajectory + its steps. Raises TrajectoryNotFoundError on miss."""
        ...

    async def list_trajectories(
        self,
        *,
        agent_name: str | None = None,
        model_id: str | None = None,
        final_status: TrajectoryStatus | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Trajectory]:
        """List trajectory metadata (no steps), most-recent first."""
        ...

    async def delete_trajectory(self, traj_id: str) -> None:
        """Idempotently remove a trajectory and its steps."""
        ...

    async def count(
        self,
        *,
        agent_name: str | None = None,
        model_id: str | None = None,
        final_status: TrajectoryStatus | None = None,
    ) -> int:
        """Count trajectories matching the filters."""
        ...
