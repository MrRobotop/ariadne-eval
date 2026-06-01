"""Benchmark adapters and runners.

Concrete adapters (e.g. ``TauBenchAdapter``) live in submodules and are
gated behind optional extras (e.g. ``pip install 'ariadne-eval[tau-bench]'``).
The Protocol and runner types live here so users can compose them
without paying for adapter dependencies they don't use.
"""

from __future__ import annotations

from ariadne_eval.benchmarks.base import (
    Benchmark,
    BenchmarkRunResult,
    BenchmarkTask,
)

__all__ = ["Benchmark", "BenchmarkRunResult", "BenchmarkTask"]
