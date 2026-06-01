"""Tests for BenchmarkConfig + YAML loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from ariadne_eval.benchmarks.config import (
    BenchmarkConfig,
    ModelSpec,
    load_benchmark_config,
)

pytestmark = pytest.mark.fast


_VALID_YAML = """
benchmark:
  kind: tau-bench
  env: retail
  task_split: test
  user_model: groq/llama-3.3-70b-versatile
  user_strategy: llm
  agent_kind: tool-calling

models:
  - model: anthropic/claude-haiku-4-5-20251001
    provider: anthropic
  - model: groq/llama-3.3-70b-versatile
    provider: groq

tasks:
  limit: 50
  seed: 42

concurrency: 4

bootstrap:
  n_resamples: 1000
  confidence: 0.95

metrics:
  - step_efficiency
  - plan_quality

judge:
  model: anthropic/claude-sonnet-4-6
  temperature: 0.0

output:
  bundle_dir: docs/benchmarks/v0.0.9-alpha-tau-retail-50
  store_path: ~/.ariadne/bench-store.duckdb
"""


def test_load_valid_yaml(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(_VALID_YAML, encoding="utf-8")
    cfg = load_benchmark_config(p)
    assert isinstance(cfg, BenchmarkConfig)
    assert cfg.benchmark.kind == "tau-bench"
    assert cfg.benchmark.env == "retail"
    assert len(cfg.models) == 2
    assert cfg.models[0] == ModelSpec(
        model="anthropic/claude-haiku-4-5-20251001", provider="anthropic"
    )
    assert cfg.tasks.limit == 50
    assert cfg.tasks.seed == 42
    assert cfg.concurrency == 4
    assert cfg.judge.model == "anthropic/claude-sonnet-4-6"
    assert cfg.judge.temperature == 0.0
    assert "step_efficiency" in cfg.metrics
    assert "plan_quality" in cfg.metrics


def test_config_is_frozen(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(_VALID_YAML, encoding="utf-8")
    cfg = load_benchmark_config(p)
    with pytest.raises(Exception):  # ValidationError on frozen  # noqa: B017
        cfg.concurrency = 8  # type: ignore[misc]


def test_load_rejects_missing_benchmark_block(tmp_path: Path) -> None:
    bad = _VALID_YAML.replace("benchmark:", "wrong_key:")
    p = tmp_path / "cfg.yaml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark"):
        load_benchmark_config(p)


def test_load_rejects_invalid_env_name(tmp_path: Path) -> None:
    bad = _VALID_YAML.replace("env: retail", "env: groceries")
    p = tmp_path / "cfg.yaml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="env"):
        load_benchmark_config(p)


def test_load_rejects_invalid_metric_name(tmp_path: Path) -> None:
    bad = _VALID_YAML.replace("step_efficiency", "made_up_metric")
    p = tmp_path / "cfg.yaml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError):
        load_benchmark_config(p)


def test_load_rejects_negative_concurrency(tmp_path: Path) -> None:
    bad = _VALID_YAML.replace("concurrency: 4", "concurrency: -1")
    p = tmp_path / "cfg.yaml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError):
        load_benchmark_config(p)


def test_load_rejects_malformed_yaml(tmp_path: Path) -> None:
    """Unclosed bracket / bad indentation → ValueError (not yaml.ParserError)."""
    p = tmp_path / "cfg.yaml"
    # Unclosed bracket — a definitively malformed YAML
    p.write_text("benchmark:\n  kind: tau-bench\nmodels: [\n", encoding="utf-8")
    with pytest.raises(ValueError, match="failed to parse YAML"):
        load_benchmark_config(p)
