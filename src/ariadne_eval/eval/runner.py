"""Runner that evaluates (Trajectory, Steps, Case) triples through metrics."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ariadne_eval.core.trajectory import Step, Trajectory
from ariadne_eval.eval.case import Case
from ariadne_eval.eval.errors import MissingReferenceError
from ariadne_eval.eval.metrics.base import Metric, MetricResult
from ariadne_eval.eval.stats.bootstrap import BootstrapCI, bootstrap_mean_ci

__all__ = ["EvalReport", "Runner"]


class EvalReport(BaseModel):
    """Per-(case, metric) results plus bootstrap aggregates."""

    model_config = {"frozen": True}

    results: tuple[MetricResult, ...] = Field(default_factory=tuple)
    aggregates: dict[str, BootstrapCI] = Field(default_factory=dict)
    n_cases: int = 0
    seed: int = 0

    def to_jsonl(self, path: str | Path) -> None:
        """Write a JSONL file: one header line then one MetricResult per line."""
        p = Path(path)
        with p.open("w", encoding="utf-8") as f:
            header = {
                "_kind": "header",
                "n_cases": self.n_cases,
                "seed": self.seed,
                "aggregates": {k: v.model_dump() for k, v in self.aggregates.items()},
            }
            f.write(json.dumps(header, sort_keys=True))
            f.write("\n")
            for r in self.results:
                line = {"_kind": "result", **r.model_dump()}
                f.write(json.dumps(line, sort_keys=True, default=str))
                f.write("\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> EvalReport:
        """Read a JSONL file written by ``to_jsonl`` and reconstruct an EvalReport."""
        p = Path(path)
        results: list[MetricResult] = []
        n_cases = 0
        seed = 0
        aggregates: dict[str, BootstrapCI] = {}
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                kind = obj.pop("_kind", None)
                if kind == "header":
                    n_cases = int(obj["n_cases"])
                    seed = int(obj["seed"])
                    aggregates = {
                        k: BootstrapCI.model_validate(v) for k, v in obj["aggregates"].items()
                    }
                elif kind == "result":
                    results.append(MetricResult.model_validate(obj))
                else:  # pragma: no cover - defensive
                    raise ValueError(f"Unknown JSONL line kind: {kind!r}")
        return cls(
            results=tuple(results),
            aggregates=aggregates,
            n_cases=n_cases,
            seed=seed,
        )


class Runner:
    """Evaluate a stream of (Trajectory, Steps, Case) triples."""

    def __init__(
        self,
        metrics: Sequence[Metric],
        *,
        seed: int = 0,
        n_resamples: int = 1000,
        confidence: float = 0.95,
        on_missing_reference: Literal["skip", "error"] = "skip",
    ) -> None:
        """Initialise with a list of metrics and evaluation options."""
        self._metrics = list(metrics)
        self._seed = seed
        self._n_resamples = n_resamples
        self._confidence = confidence
        self._on_missing = on_missing_reference

    def evaluate(
        self,
        items: Iterable[tuple[Trajectory, list[Step], Case]],
    ) -> EvalReport:
        """Score every item through all metrics and return an EvalReport."""
        per_metric: dict[str, list[float]] = {m.name: [] for m in self._metrics}
        results: list[MetricResult] = []
        n_cases = 0

        for traj, steps, case in items:
            n_cases += 1
            for metric in self._metrics:
                try:
                    res = metric.score(traj, steps, case)
                except MissingReferenceError:
                    if self._on_missing == "error":
                        raise
                    continue
                results.append(res)
                per_metric[metric.name].append(res.score)

        aggregates = {
            name: bootstrap_mean_ci(
                values,
                n_resamples=self._n_resamples,
                confidence=self._confidence,
                seed=self._seed,
            )
            for name, values in per_metric.items()
        }
        return EvalReport(
            results=tuple(results),
            aggregates=aggregates,
            n_cases=n_cases,
            seed=self._seed,
        )
