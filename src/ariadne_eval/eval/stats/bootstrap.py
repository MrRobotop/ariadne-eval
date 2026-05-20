"""Percentile bootstrap confidence interval for the mean."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel

from ariadne_eval.eval.errors import BootstrapInsufficientDataWarning

__all__ = ["BootstrapCI", "bootstrap_mean_ci"]


class BootstrapCI(BaseModel):
    """Result of a percentile-bootstrap CI on the mean."""

    model_config = {"frozen": True}

    mean: float
    lo: float
    hi: float
    n: int
    n_resamples: int
    confidence: float


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> BootstrapCI:
    """Percentile-bootstrap confidence interval for the mean of ``values``.

    For ``n == 0`` returns an all-NaN CI and emits
    ``BootstrapInsufficientDataWarning``.
    For ``n == 1`` returns a degenerate CI equal to the single value and
    emits ``BootstrapInsufficientDataWarning``.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples!r}")

    n = len(values)
    if n == 0:
        warnings.warn(
            "bootstrap_mean_ci called with n=0; returning NaN CI",
            BootstrapInsufficientDataWarning,
            stacklevel=2,
        )
        return BootstrapCI(
            mean=math.nan,
            lo=math.nan,
            hi=math.nan,
            n=0,
            n_resamples=n_resamples,
            confidence=confidence,
        )

    arr = np.asarray(values, dtype=float)
    if n == 1:
        warnings.warn(
            "bootstrap_mean_ci called with n=1; CI degenerates to the value",
            BootstrapInsufficientDataWarning,
            stacklevel=2,
        )
        v = float(arr[0])
        return BootstrapCI(
            mean=v,
            lo=v,
            hi=v,
            n=1,
            n_resamples=n_resamples,
            confidence=confidence,
        )

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = arr[indices].mean(axis=1)
    alpha = 1.0 - confidence
    lo = float(np.quantile(resampled_means, alpha / 2))
    hi = float(np.quantile(resampled_means, 1 - alpha / 2))

    return BootstrapCI(
        mean=float(arr.mean()),
        lo=lo,
        hi=hi,
        n=n,
        n_resamples=n_resamples,
        confidence=confidence,
    )
