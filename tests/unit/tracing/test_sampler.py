"""Sampler Protocol and the three concrete sampler implementations."""

from __future__ import annotations

import pytest

from ariadne_eval.tracing.sampler import (
    AlwaysSampler,
    RateSampler,
    Sampler,
    TaskFilterSampler,
)


_KW = dict(
    task="t",
    agent_name="a",
    agent_version="0.1",
    model_id="m",
    metadata={},
)


@pytest.mark.fast
def test_sampler_is_protocol():
    assert hasattr(Sampler, "should_sample")


@pytest.mark.fast
def test_always_sampler_returns_true():
    assert AlwaysSampler().should_sample(**_KW) is True


@pytest.mark.fast
def test_rate_sampler_zero_never_samples():
    s = RateSampler(rate=0.0, seed=42)
    assert all(s.should_sample(**_KW) is False for _ in range(100))


@pytest.mark.fast
def test_rate_sampler_one_always_samples():
    s = RateSampler(rate=1.0, seed=42)
    assert all(s.should_sample(**_KW) is True for _ in range(100))


@pytest.mark.fast
def test_rate_sampler_seeded_is_deterministic():
    a = RateSampler(rate=0.5, seed=42)
    b = RateSampler(rate=0.5, seed=42)
    seq_a = [a.should_sample(**_KW) for _ in range(20)]
    seq_b = [b.should_sample(**_KW) for _ in range(20)]
    assert seq_a == seq_b


@pytest.mark.fast
def test_rate_sampler_validates_rate():
    with pytest.raises(ValueError):
        RateSampler(rate=-0.1)
    with pytest.raises(ValueError):
        RateSampler(rate=1.5)


@pytest.mark.fast
def test_task_filter_sampler_uses_predicate():
    s = TaskFilterSampler(predicate=lambda task: "math" in task)
    assert s.should_sample(**{**_KW, "task": "math problem"}) is True
    assert s.should_sample(**{**_KW, "task": "writing"}) is False
