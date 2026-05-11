"""Sampling decisions for trajectories.

The sampler is consulted once at ``start_trajectory``. If it returns
``False``, the entire trajectory is a no-op - recorders inside it short
circuit without allocating Steps. This is what makes sampling cheap in
production: unsampled trajectories pay near-zero overhead.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ariadne_eval.core.trajectory import JsonValue

__all__ = [
    "AlwaysSampler",
    "RateSampler",
    "Sampler",
    "TaskFilterSampler",
]


class Sampler(Protocol):
    """Per-trajectory sampling decision."""

    def should_sample(
        self,
        *,
        task: str,
        agent_name: str,
        agent_version: str,
        model_id: str,
        metadata: dict[str, "JsonValue"],
    ) -> bool:
        """Return ``True`` to record the trajectory, ``False`` to no-op it."""
        ...


class AlwaysSampler:
    """Default: every trajectory is recorded."""

    def should_sample(self, **_kw: Any) -> bool:
        """Always return True."""
        return True


@dataclass
class RateSampler:
    """Sample a fraction ``rate`` of trajectories uniformly at random.

    ``rate=0.0`` skips everything, ``rate=1.0`` records everything.
    Pass ``seed`` for deterministic test runs.
    """

    rate: float
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate rate and build the local RNG."""
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError(f"rate must be in [0.0, 1.0]; got {self.rate}")
        self._rng = random.Random(self.seed)

    def should_sample(self, **_kw: Any) -> bool:
        """Return True with probability ``rate``."""
        if self.rate == 0.0:
            return False
        if self.rate == 1.0:
            return True
        return self._rng.random() < self.rate


@dataclass
class TaskFilterSampler:
    """Sample only trajectories whose ``task`` matches a predicate."""

    predicate: Callable[[str], bool]

    def should_sample(self, *, task: str, **_kw: Any) -> bool:
        """Delegate to the predicate."""
        return self.predicate(task)
