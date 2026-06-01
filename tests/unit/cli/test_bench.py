"""Tests for the `ariadne bench run` CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from ariadne_eval.cli.main import cli

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

tasks:
  limit: 50
  seed: 42

concurrency: 4

bootstrap:
  n_resamples: 1000
  confidence: 0.95

metrics:
  - step_efficiency

judge:
  model: anthropic/claude-sonnet-4-6
  temperature: 0.0

output:
  bundle_dir: /tmp/bundle
  store_path: /tmp/bench.duckdb
"""


def test_bench_run_dry_run_validates_config(tmp_path: Path) -> None:
    """--dry-run loads + validates the config; no LLM calls."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_VALID_YAML, encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["bench", "run", str(cfg), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "config valid" in result.output.lower() or "ok" in result.output.lower()


def test_bench_run_dry_run_invalid_config_fails(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_VALID_YAML.replace("env: retail", "env: nonsense"), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["bench", "run", str(cfg), "--dry-run"])
    assert result.exit_code != 0


def test_bench_run_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["bench", "run", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.output
    assert "--limit" in result.output
    assert "--models" in result.output
    assert "--resume" in result.output
