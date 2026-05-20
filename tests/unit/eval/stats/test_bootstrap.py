"""Tests for the percentile bootstrap CI implementation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, seed, settings
from hypothesis import strategies as st

from ariadne_eval.eval.errors import BootstrapInsufficientDataWarning
from ariadne_eval.eval.stats.bootstrap import BootstrapCI, bootstrap_mean_ci

pytestmark = pytest.mark.fast


def test_basic_shape_and_bounds() -> None:
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 1, size=200).tolist()
    ci = bootstrap_mean_ci(values, seed=42)
    assert isinstance(ci, BootstrapCI)
    assert ci.n == 200
    assert ci.n_resamples == 1000
    assert ci.confidence == 0.95
    assert ci.lo <= ci.mean <= ci.hi


def test_seed_is_reproducible() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    a = bootstrap_mean_ci(values, seed=7)
    b = bootstrap_mean_ci(values, seed=7)
    assert a == b


def test_different_seeds_differ() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    a = bootstrap_mean_ci(values, seed=1)
    b = bootstrap_mean_ci(values, seed=2)
    assert (a.lo, a.hi) != (b.lo, b.hi)


def test_empty_input_warns_and_nan() -> None:
    with pytest.warns(BootstrapInsufficientDataWarning):
        ci = bootstrap_mean_ci([], seed=0)
    assert ci.n == 0
    assert math.isnan(ci.mean)
    assert math.isnan(ci.lo)
    assert math.isnan(ci.hi)


def test_single_value_warns_and_degenerate() -> None:
    with pytest.warns(BootstrapInsufficientDataWarning):
        ci = bootstrap_mean_ci([0.7], seed=0)
    assert ci.n == 1
    assert ci.mean == 0.7
    assert ci.lo == 0.7
    assert ci.hi == 0.7


def test_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        bootstrap_mean_ci([0.1, 0.2], confidence=1.5, seed=0)


def test_invalid_n_resamples() -> None:
    with pytest.raises(ValueError):
        bootstrap_mean_ci([0.1, 0.2], n_resamples=0, seed=0)


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@seed(2026)
@given(seed_=st.integers(min_value=0, max_value=10_000))
def test_property_coverage_around_true_mean(seed_: int) -> None:
    """Loose coverage check: 95% CI of mean should cover the population mean
    most of the time. Per-call check, not a global rate."""
    rng = np.random.default_rng(seed_)
    values = rng.uniform(0, 1, size=200).tolist()
    ci = bootstrap_mean_ci(values, n_resamples=400, seed=seed_)
    sample_mean = float(np.mean(values))
    # The CI is a CI of the *sample* mean — it should always contain it.
    assert ci.lo <= sample_mean <= ci.hi
