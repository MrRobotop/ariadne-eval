"""Benchmark run configuration: Pydantic model + YAML loader."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

__all__ = [
    "BenchmarkConfig",
    "BootstrapSpec",
    "JudgeSpec",
    "ModelSpec",
    "OutputSpec",
    "TasksSpec",
    "TauBenchSpec",
    "load_benchmark_config",
]


class ModelSpec(BaseModel):
    """One agent model under test."""

    model_config = {"frozen": True}

    model: str
    provider: str


class JudgeSpec(BaseModel):
    """Judge configuration for ``PlanQuality``."""

    model_config = {"frozen": True}

    model: str
    temperature: float = 0.0


class BootstrapSpec(BaseModel):
    """Bootstrap CI parameters."""

    model_config = {"frozen": True}

    n_resamples: int = Field(default=1000, gt=0)
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)


class TasksSpec(BaseModel):
    """Task sampling configuration."""

    model_config = {"frozen": True}

    limit: int | None = Field(default=None, ge=1)
    seed: int = 42


class TauBenchSpec(BaseModel):
    """tau-bench-specific benchmark configuration."""

    model_config = {"frozen": True}

    kind: Literal["tau-bench"]
    env: Literal["retail", "airline"]
    task_split: str = "test"
    user_model: str = "groq/llama-3.3-70b-versatile"
    user_strategy: str = "llm"
    agent_kind: Literal["tool-calling", "react", "few-shot-tool-calling"] = "tool-calling"


class OutputSpec(BaseModel):
    """Result bundle output paths."""

    model_config = {"frozen": True}

    bundle_dir: Path
    store_path: Path


class BenchmarkConfig(BaseModel):
    """Root benchmark run configuration; loaded from YAML."""

    model_config = {"frozen": True}

    benchmark: TauBenchSpec
    models: list[ModelSpec]
    tasks: TasksSpec
    concurrency: int = Field(default=4, gt=0)
    bootstrap: BootstrapSpec
    metrics: list[Literal["step_efficiency", "plan_quality"]]
    judge: JudgeSpec
    output: OutputSpec

    @field_validator("models")
    @classmethod
    def _at_least_one_model(cls, v: list[ModelSpec]) -> list[ModelSpec]:
        if not v:
            raise ValueError("models: at least one model must be specified")
        return v


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    """Load + validate a benchmark config YAML.

    Raises ``ValueError`` if either the YAML parser fails or the
    structure doesn't match :class:`BenchmarkConfig`. The error message
    names the offending field path (Pydantic) or the parser fault
    (PyYAML).
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"failed to parse YAML at {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"benchmark config root must be a mapping, got {type(raw).__name__}")
    return BenchmarkConfig.model_validate(raw)
