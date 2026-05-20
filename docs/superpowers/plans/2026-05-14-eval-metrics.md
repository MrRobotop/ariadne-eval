# Phase 5 — Eval Metrics + Bootstrap CIs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first usable evaluation loop: deterministic per-trajectory metrics (`FinalAnswerMatch`, `ToolAccuracy`, `StepEfficiency`), a percentile bootstrap CI, and a `Runner` that turns `(Trajectory, list[Step], Case)` triples into an aggregated `EvalReport`.

**Architecture:** Pydantic-frozen value types for `Case`, `MetricResult`, `EvalReport`, `BootstrapCI`. Sync `Metric` Protocol — pure compute, no IO. Pure-NumPy seeded percentile bootstrap. JSONL round-trip on `EvalReport` (DuckDB persistence deferred). Branch: `phase-5-metrics` (already created and checked out).

**Tech Stack:** Python 3.11+, Pydantic v2, NumPy, pytest, hypothesis, ruff, mypy --strict.

**Spec:** `docs/superpowers/specs/2026-05-14-eval-metrics-design.md`

**Conventions to honor:**
- `mypy --strict src/` must stay clean.
- `ruff format` + `ruff check` clean.
- `pytest -m "fast and not integration"` clean (the `addopts` default).
- Use `Field(default_factory=dict)` for mutable defaults; frozen models can't have mutable defaults in field signatures.
- `pyproject.toml` has `filterwarnings = ["error", ...]` — any test that expects a warning MUST use `pytest.warns(...)` to consume it.
- Public API additions go in `src/ariadne_eval/__init__.py` AND `src/ariadne_eval/eval/__init__.py`.
- Conventional commits: `feat:`, `test:`, `docs:`, `chore:`.

---

## Task 1: Test scaffolding for `eval/`

**Files:**
- Create: `tests/unit/eval/__init__.py`
- Create: `tests/unit/eval/metrics/__init__.py`
- Create: `tests/unit/eval/stats/__init__.py`
- Create: `tests/unit/eval/_factories.py`

`tests/unit/eval/` and `tests/unit/eval/metrics/` exist as directories already but lack `__init__.py` — pytest collects fine without them, but the rest of `tests/unit/` uses package-marker `__init__.py` files, so we follow the convention. We also drop a small factories module so later tasks can build `Trajectory` / `Step` / `ToolCallPayload` instances without rewriting boilerplate every time.

- [ ] **Step 1: Create the three `__init__.py` files (empty)**

```bash
: > tests/unit/eval/__init__.py
: > tests/unit/eval/metrics/__init__.py
mkdir -p tests/unit/eval/stats && : > tests/unit/eval/stats/__init__.py
```

- [ ] **Step 2: Create `tests/unit/eval/_factories.py`**

```python
"""Test-only factories for Trajectory / Step instances.

Keep these tiny and dumb. If a test needs a wildly different shape, build
the model directly in the test rather than expanding these helpers.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import StepStatus, TrajectoryStatus
from ariadne_eval.core.trajectory import (
    JsonValue,
    Step,
    ToolCallPayload,
    Trajectory,
)


def make_trajectory(
    *,
    final_answer: JsonValue = "ok",
    final_status: TrajectoryStatus = TrajectoryStatus.COMPLETED,
    task: str = "demo",
    traj_id: str | None = None,
) -> Trajectory:
    started = datetime(2026, 5, 14, 12, 0, 0, tzinfo=UTC)
    return Trajectory(
        id=traj_id or new_id(),
        task=task,
        agent_name="test",
        agent_version="0.0.0",
        model_id="test/model",
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        final_status=final_status,
        final_answer=final_answer,
    )


def make_tool_step(
    *,
    trajectory_id: str,
    tool_name: str,
    arguments: dict[str, JsonValue] | None = None,
    result: JsonValue = None,
    started_at: datetime | None = None,
    parent_step_id: str | None = None,
) -> Step:
    started = started_at or datetime(2026, 5, 14, 12, 0, 0, tzinfo=UTC)
    return Step(
        id=new_id(),
        trajectory_id=trajectory_id,
        parent_step_id=parent_step_id,
        name=f"tool:{tool_name}",
        started_at=started,
        finished_at=started + timedelta(milliseconds=10),
        status=StepStatus.OK,
        payload=ToolCallPayload(
            tool_name=tool_name,
            arguments=arguments or {},
            result=result,
            latency_ms=10.0,
        ),
    )
```

- [ ] **Step 3: Verify nothing broke**

Run: `uv run pytest -m "fast and not integration" -q`
Expected: existing test suite still green (count unchanged).

- [ ] **Step 4: Commit**

```bash
git add tests/unit/eval/__init__.py tests/unit/eval/metrics/__init__.py tests/unit/eval/stats/__init__.py tests/unit/eval/_factories.py
git commit -m "test(eval): add package markers and shared factories for eval tests"
```

---

## Task 2: `Case` and `ExpectedTool`

**Files:**
- Create: `src/ariadne_eval/eval/case.py`
- Create: `tests/unit/eval/test_case.py`

- [ ] **Step 1: Write failing test**

`tests/unit/eval/test_case.py`:

```python
"""Tests for the Case sidecar model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ariadne_eval.eval.case import Case, ExpectedTool


def test_case_minimal() -> None:
    c = Case(case_id="c1", task="add 2+2")
    assert c.case_id == "c1"
    assert c.task == "add 2+2"
    assert c.expected_answer is None
    assert c.expected_tools == ()
    assert c.expected_max_steps is None
    assert c.metadata == {}


def test_case_full() -> None:
    c = Case(
        case_id="c2",
        task="search and add",
        expected_answer="4",
        expected_tools=(
            ExpectedTool(name="search", args={"q": "x"}),
            ExpectedTool(name="calculator"),
        ),
        expected_max_steps=5,
        metadata={"benchmark": "demo"},
    )
    assert c.expected_tools[1].args is None
    assert c.metadata["benchmark"] == "demo"


def test_case_is_frozen() -> None:
    c = Case(case_id="c3", task="t")
    with pytest.raises(ValidationError):
        c.task = "different"  # type: ignore[misc]


def test_expected_tool_is_frozen() -> None:
    t = ExpectedTool(name="x")
    with pytest.raises(ValidationError):
        t.name = "y"  # type: ignore[misc]
```

- [ ] **Step 2: Run test to confirm RED**

Run: `uv run pytest tests/unit/eval/test_case.py -v`
Expected: FAIL — `ImportError: cannot import name 'Case' from 'ariadne_eval.eval.case'`.

- [ ] **Step 3: Implement `Case`**

`src/ariadne_eval/eval/case.py`:

```python
"""Sidecar ground-truth model used by metrics during evaluation.

A ``Case`` is the "what should have happened" complement to a
``Trajectory`` ("what did happen"). Cases are intentionally a separate
type: they live next to benchmarks, not next to traced runs, and the
``Trajectory`` schema stays untouched.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ariadne_eval.core.trajectory import JsonValue

__all__ = ["Case", "ExpectedTool"]


class ExpectedTool(BaseModel):
    """A single expected tool invocation in a Case.

    ``args=None`` means "match this expected tool against any actual call
    of the same name, regardless of arguments". When ``ToolAccuracy`` is
    constructed with ``match_args=True``, this becomes a per-tool wildcard.
    """

    model_config = {"frozen": True}

    name: str
    args: dict[str, JsonValue] | None = None


class Case(BaseModel):
    """Ground-truth reference for a single evaluation example."""

    model_config = {"frozen": True}

    case_id: str
    task: str
    expected_answer: str | None = None
    expected_tools: tuple[ExpectedTool, ...] = ()
    expected_max_steps: int | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
```

- [ ] **Step 4: Run test, confirm GREEN**

Run: `uv run pytest tests/unit/eval/test_case.py -v`
Expected: 4 passed.

- [ ] **Step 5: Type-check and lint**

Run: `uv run mypy --strict src/ariadne_eval/eval/case.py && uv run ruff check src/ariadne_eval/eval/case.py && uv run ruff format --check src/ariadne_eval/eval/case.py`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/ariadne_eval/eval/case.py tests/unit/eval/test_case.py
git commit -m "feat(eval): add Case and ExpectedTool sidecar models"
```

---

## Task 3: Errors and warnings module

**Files:**
- Create: `src/ariadne_eval/eval/errors.py`
- Create: `tests/unit/eval/test_errors.py`

- [ ] **Step 1: Write failing test**

`tests/unit/eval/test_errors.py`:

```python
from __future__ import annotations

import warnings

import pytest

from ariadne_eval.eval.errors import (
    BootstrapInsufficientDataWarning,
    MissingReferenceError,
)


def test_missing_reference_error_message() -> None:
    err = MissingReferenceError("expected_answer", case_id="c1")
    assert "expected_answer" in str(err)
    assert "c1" in str(err)


def test_missing_reference_error_is_value_error() -> None:
    assert issubclass(MissingReferenceError, ValueError)


def test_bootstrap_warning_is_user_warning() -> None:
    assert issubclass(BootstrapInsufficientDataWarning, UserWarning)


def test_bootstrap_warning_round_trip() -> None:
    with pytest.warns(BootstrapInsufficientDataWarning, match="n=0"):
        warnings.warn("n=0 not enough", BootstrapInsufficientDataWarning, stacklevel=1)
```

- [ ] **Step 2: Run test, confirm RED**

Run: `uv run pytest tests/unit/eval/test_errors.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement errors module**

`src/ariadne_eval/eval/errors.py`:

```python
"""Errors and warnings raised by the evaluation layer."""

from __future__ import annotations

__all__ = ["BootstrapInsufficientDataWarning", "MissingReferenceError"]


class MissingReferenceError(ValueError):
    """A metric required a Case field that was not provided.

    Subclass of ``ValueError`` so callers who broadly catch validation
    errors still see it; ``Runner`` catches it explicitly to honor
    ``on_missing_reference``.
    """

    def __init__(self, field: str, *, case_id: str) -> None:
        super().__init__(
            f"Case {case_id!r} is missing required reference field {field!r}"
        )
        self.field = field
        self.case_id = case_id


class BootstrapInsufficientDataWarning(UserWarning):
    """Emitted when ``bootstrap_mean_ci`` cannot produce a meaningful CI.

    Raised for ``n == 0`` (NaN result) and ``n == 1`` (degenerate CI equal
    to the single value).
    """
```

- [ ] **Step 4: Run test, confirm GREEN**

Run: `uv run pytest tests/unit/eval/test_errors.py -v`
Expected: 4 passed.

- [ ] **Step 5: Type-check and lint**

Run: `uv run mypy --strict src/ariadne_eval/eval/errors.py && uv run ruff check src/ariadne_eval/eval/errors.py && uv run ruff format --check src/ariadne_eval/eval/errors.py`

- [ ] **Step 6: Commit**

```bash
git add src/ariadne_eval/eval/errors.py tests/unit/eval/test_errors.py
git commit -m "feat(eval): add MissingReferenceError and BootstrapInsufficientDataWarning"
```

---

## Task 4: `MetricResult` and `Metric` Protocol

**Files:**
- Create: `src/ariadne_eval/eval/metrics/base.py`
- Create: `tests/unit/eval/metrics/test_base.py`

- [ ] **Step 1: Write failing test**

`tests/unit/eval/metrics/test_base.py`:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from ariadne_eval.eval.metrics.base import MetricResult


def test_metric_result_minimal() -> None:
    r = MetricResult(
        metric="m",
        case_id="c",
        trajectory_id="01J0000000000000000000000A",
        score=0.75,
    )
    assert r.label is None
    assert r.details == {}


def test_metric_result_full() -> None:
    r = MetricResult(
        metric="m",
        case_id="c",
        trajectory_id="01J0000000000000000000000A",
        score=1.0,
        label="pass",
        details={"reason": "ok"},
    )
    assert r.label == "pass"
    assert r.details["reason"] == "ok"


def test_metric_result_is_frozen() -> None:
    r = MetricResult(
        metric="m",
        case_id="c",
        trajectory_id="01J0000000000000000000000A",
        score=0.0,
    )
    with pytest.raises(ValidationError):
        r.score = 0.5  # type: ignore[misc]


def test_metric_result_label_literal_validated() -> None:
    with pytest.raises(ValidationError):
        MetricResult(
            metric="m",
            case_id="c",
            trajectory_id="01J0000000000000000000000A",
            score=0.0,
            label="bogus",  # type: ignore[arg-type]
        )
```

- [ ] **Step 2: Run test, confirm RED**

Run: `uv run pytest tests/unit/eval/metrics/test_base.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `MetricResult` and `Metric` protocol**

`src/ariadne_eval/eval/metrics/base.py`:

```python
"""Base types for evaluation metrics."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ariadne_eval.core.trajectory import JsonValue, Step, Trajectory
from ariadne_eval.eval.case import Case

__all__ = ["Metric", "MetricResult"]


class MetricResult(BaseModel):
    """Per-(trajectory, case) output from a single Metric.

    ``score`` is always populated. For Phase-5 metrics it is in ``[0, 1]``,
    but the type does not enforce a range — future metrics may produce
    negative or unbounded scores.
    """

    model_config = {"frozen": True}

    metric: str
    case_id: str
    trajectory_id: str
    score: float
    label: Literal["pass", "fail", "partial"] | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


@runtime_checkable
class Metric(Protocol):
    """Pure-compute, sync per-trajectory scoring contract.

    Implementations are expected to be deterministic. Async metrics
    (judges) arrive in Phase 6 behind a separate ``AsyncMetric`` Protocol.
    """

    name: str

    def score(
        self, trajectory: Trajectory, steps: list[Step], case: Case
    ) -> MetricResult: ...
```

- [ ] **Step 4: Run test, confirm GREEN**

Run: `uv run pytest tests/unit/eval/metrics/test_base.py -v`
Expected: 4 passed.

- [ ] **Step 5: Type-check and lint**

Run: `uv run mypy --strict src/ariadne_eval/eval/metrics/base.py && uv run ruff check src/ariadne_eval/eval/metrics/base.py && uv run ruff format --check src/ariadne_eval/eval/metrics/base.py`

- [ ] **Step 6: Commit**

```bash
git add src/ariadne_eval/eval/metrics/base.py tests/unit/eval/metrics/test_base.py
git commit -m "feat(eval): add MetricResult and Metric Protocol"
```

---

## Task 5: `FinalAnswerMatch`

**Files:**
- Create: `src/ariadne_eval/eval/metrics/final_answer.py`
- Create: `tests/unit/eval/metrics/test_final_answer.py`

- [ ] **Step 1: Write failing test**

`tests/unit/eval/metrics/test_final_answer.py`:

```python
from __future__ import annotations

import pytest

from ariadne_eval.core.status import TrajectoryStatus
from ariadne_eval.eval.case import Case
from ariadne_eval.eval.errors import MissingReferenceError
from ariadne_eval.eval.metrics.final_answer import FinalAnswerMatch
from tests.unit.eval._factories import make_trajectory


def test_normalized_exact_pass() -> None:
    traj = make_trajectory(final_answer="  4  ")
    case = Case(case_id="c", task="2+2", expected_answer="4")
    r = FinalAnswerMatch().score(traj, [], case)
    assert r.score == 1.0
    assert r.label == "pass"
    assert r.metric == "final_answer_match"


def test_normalized_exact_collapses_internal_whitespace() -> None:
    traj = make_trajectory(final_answer="The   answer  is   four")
    case = Case(case_id="c", task="t", expected_answer="the answer is four")
    r = FinalAnswerMatch().score(traj, [], case)
    assert r.score == 1.0


def test_normalized_exact_fail() -> None:
    traj = make_trajectory(final_answer="five")
    case = Case(case_id="c", task="t", expected_answer="four")
    r = FinalAnswerMatch().score(traj, [], case)
    assert r.score == 0.0
    assert r.label == "fail"


def test_exact_mode_distinguishes_case() -> None:
    traj = make_trajectory(final_answer="Hello")
    case = Case(case_id="c", task="t", expected_answer="hello")
    r = FinalAnswerMatch(comparator="exact").score(traj, [], case)
    assert r.score == 0.0


def test_custom_comparator_partial() -> None:
    def half(a: str, b: str) -> float:
        return 0.5

    traj = make_trajectory(final_answer="x")
    case = Case(case_id="c", task="t", expected_answer="y")
    r = FinalAnswerMatch(comparator=half).score(traj, [], case)
    assert r.score == 0.5
    assert r.label == "partial"


def test_missing_reference_raises() -> None:
    traj = make_trajectory(final_answer="x")
    case = Case(case_id="c", task="t")
    with pytest.raises(MissingReferenceError):
        FinalAnswerMatch().score(traj, [], case)


def test_no_final_answer_is_fail() -> None:
    traj = make_trajectory(
        final_answer=None, final_status=TrajectoryStatus.FAILED
    )
    case = Case(case_id="c", task="t", expected_answer="x")
    r = FinalAnswerMatch().score(traj, [], case)
    assert r.score == 0.0
    assert r.label == "fail"
    assert r.details["reason"] == "no_final_answer"


def test_non_string_final_answer_is_json_serialized() -> None:
    traj = make_trajectory(final_answer={"value": 4})
    case = Case(
        case_id="c", task="t", expected_answer='{"value": 4}'
    )
    r = FinalAnswerMatch(comparator="exact").score(traj, [], case)
    assert r.score == 1.0
```

- [ ] **Step 2: Run test, confirm RED**

Run: `uv run pytest tests/unit/eval/metrics/test_final_answer.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `FinalAnswerMatch`**

`src/ariadne_eval/eval/metrics/final_answer.py`:

```python
"""Final-answer match metric."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Literal

from ariadne_eval.core.trajectory import Step, Trajectory
from ariadne_eval.eval.case import Case
from ariadne_eval.eval.errors import MissingReferenceError
from ariadne_eval.eval.metrics.base import MetricResult

__all__ = ["FinalAnswerMatch"]


_WHITESPACE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    return _WHITESPACE.sub(" ", s.strip().lower())


def _render(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _label_from_score(score: float) -> Literal["pass", "fail", "partial"]:
    if score >= 0.99:
        return "pass"
    if score <= 0.01:
        return "fail"
    return "partial"


class FinalAnswerMatch:
    """Compare ``trajectory.final_answer`` against ``case.expected_answer``."""

    name: str

    def __init__(
        self,
        comparator: Literal["normalized_exact", "exact"]
        | Callable[[str, str], float] = "normalized_exact",
        *,
        name: str = "final_answer_match",
    ) -> None:
        self._comparator = comparator
        self.name = name

    def score(
        self, trajectory: Trajectory, steps: list[Step], case: Case
    ) -> MetricResult:
        if case.expected_answer is None:
            raise MissingReferenceError("expected_answer", case_id=case.case_id)

        if trajectory.final_answer is None:
            return MetricResult(
                metric=self.name,
                case_id=case.case_id,
                trajectory_id=trajectory.id,
                score=0.0,
                label="fail",
                details={"reason": "no_final_answer"},
            )

        actual = _render(trajectory.final_answer)
        expected = case.expected_answer

        if self._comparator == "normalized_exact":
            score = 1.0 if _normalize(actual) == _normalize(expected) else 0.0
            label: Literal["pass", "fail", "partial"] = (
                "pass" if score == 1.0 else "fail"
            )
        elif self._comparator == "exact":
            score = 1.0 if actual == expected else 0.0
            label = "pass" if score == 1.0 else "fail"
        else:
            score = float(self._comparator(actual, expected))
            label = _label_from_score(score)

        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            trajectory_id=trajectory.id,
            score=score,
            label=label,
        )
```

- [ ] **Step 4: Run test, confirm GREEN**

Run: `uv run pytest tests/unit/eval/metrics/test_final_answer.py -v`
Expected: 8 passed.

- [ ] **Step 5: Type-check and lint**

Run: `uv run mypy --strict src/ariadne_eval/eval/metrics/final_answer.py && uv run ruff check src/ariadne_eval/eval/metrics/final_answer.py && uv run ruff format --check src/ariadne_eval/eval/metrics/final_answer.py`

- [ ] **Step 6: Commit**

```bash
git add src/ariadne_eval/eval/metrics/final_answer.py tests/unit/eval/metrics/test_final_answer.py
git commit -m "feat(eval): add FinalAnswerMatch metric"
```

---

## Task 6: `ToolAccuracy`

**Files:**
- Create: `src/ariadne_eval/eval/metrics/tool_accuracy.py`
- Create: `tests/unit/eval/metrics/test_tool_accuracy.py`

- [ ] **Step 1: Write failing test**

`tests/unit/eval/metrics/test_tool_accuracy.py`:

```python
from __future__ import annotations

from ariadne_eval.eval.case import Case, ExpectedTool
from ariadne_eval.eval.metrics.tool_accuracy import ToolAccuracy
from tests.unit.eval._factories import make_tool_step, make_trajectory


def _scenario(
    *, expected_names: list[str], actual_names: list[str]
) -> tuple[object, list[object], object]:
    traj = make_trajectory()
    steps = [
        make_tool_step(trajectory_id=traj.id, tool_name=n) for n in actual_names
    ]
    case = Case(
        case_id="c",
        task="t",
        expected_tools=tuple(ExpectedTool(name=n) for n in expected_names),
    )
    return traj, steps, case


def test_set_mode_perfect_match() -> None:
    traj, steps, case = _scenario(
        expected_names=["a", "b"], actual_names=["a", "b"]
    )
    r = ToolAccuracy().score(traj, steps, case)  # type: ignore[arg-type]
    assert r.score == 1.0
    assert r.label == "pass"
    assert r.details["precision"] == 1.0
    assert r.details["recall"] == 1.0


def test_set_mode_partial_f1() -> None:
    # expected = {a, b}, actual = {a, c} => tp=1, fp=1, fn=1 => P=R=0.5, F1=0.5
    traj, steps, case = _scenario(
        expected_names=["a", "b"], actual_names=["a", "c"]
    )
    r = ToolAccuracy().score(traj, steps, case)  # type: ignore[arg-type]
    assert r.score == 0.5
    assert r.label == "partial"
    assert r.details["matched"] == ["a"]
    assert r.details["missing"] == ["b"]
    assert r.details["extra"] == ["c"]


def test_set_mode_treats_duplicates_as_multiset() -> None:
    # expected = [a, a, b], actual = [a, b] => tp=2, fn=1, fp=0
    # P = 2/2 = 1.0, R = 2/3 ≈ 0.667, F1 = 0.8
    traj, steps, case = _scenario(
        expected_names=["a", "a", "b"], actual_names=["a", "b"]
    )
    r = ToolAccuracy().score(traj, steps, case)  # type: ignore[arg-type]
    assert r.score == pytest_approx(0.8)


def test_ordered_prefix_full_match() -> None:
    traj, steps, case = _scenario(
        expected_names=["a", "b", "c"], actual_names=["a", "b", "c", "d"]
    )
    r = ToolAccuracy(mode="ordered_prefix").score(traj, steps, case)  # type: ignore[arg-type]
    assert r.score == 1.0
    assert r.details["prefix_length"] == 3


def test_ordered_prefix_partial() -> None:
    traj, steps, case = _scenario(
        expected_names=["a", "b", "c"], actual_names=["a", "x", "c"]
    )
    r = ToolAccuracy(mode="ordered_prefix").score(traj, steps, case)  # type: ignore[arg-type]
    assert r.score == pytest_approx(1 / 3)
    assert r.details["first_divergence_index"] == 1


def test_match_args_true_strict() -> None:
    traj = make_trajectory()
    steps = [
        make_tool_step(trajectory_id=traj.id, tool_name="search", arguments={"q": "x"})
    ]
    case = Case(
        case_id="c",
        task="t",
        expected_tools=(ExpectedTool(name="search", args={"q": "x"}),),
    )
    r = ToolAccuracy(match_args=True).score(traj, steps, case)
    assert r.score == 1.0


def test_match_args_true_args_mismatch() -> None:
    traj = make_trajectory()
    steps = [
        make_tool_step(trajectory_id=traj.id, tool_name="search", arguments={"q": "x"})
    ]
    case = Case(
        case_id="c",
        task="t",
        expected_tools=(ExpectedTool(name="search", args={"q": "y"}),),
    )
    r = ToolAccuracy(match_args=True).score(traj, steps, case)
    assert r.score == 0.0


def test_match_args_true_with_args_none_is_per_tool_wildcard() -> None:
    traj = make_trajectory()
    steps = [
        make_tool_step(trajectory_id=traj.id, tool_name="search", arguments={"q": "x"})
    ]
    case = Case(
        case_id="c",
        task="t",
        expected_tools=(ExpectedTool(name="search", args=None),),
    )
    r = ToolAccuracy(match_args=True).score(traj, steps, case)
    assert r.score == 1.0


def test_empty_expected_set_mode_with_extras_fails() -> None:
    traj = make_trajectory()
    steps = [make_tool_step(trajectory_id=traj.id, tool_name="a")]
    case = Case(case_id="c", task="t", expected_tools=())
    r = ToolAccuracy().score(traj, steps, case)
    assert r.score == 0.0
    assert r.label == "fail"


def test_empty_expected_no_extras_passes() -> None:
    traj = make_trajectory()
    case = Case(case_id="c", task="t", expected_tools=())
    r = ToolAccuracy().score(traj, [], case)
    assert r.score == 1.0
    assert r.label == "pass"


def test_empty_expected_ordered_prefix_quirk_documented() -> None:
    # Documented quirk: empty prefix matches everything in ordered_prefix mode.
    traj = make_trajectory()
    steps = [make_tool_step(trajectory_id=traj.id, tool_name="a")]
    case = Case(case_id="c", task="t", expected_tools=())
    r = ToolAccuracy(mode="ordered_prefix").score(traj, steps, case)
    assert r.score == 1.0


# tiny local approx (avoid pytest.approx import noise)
def pytest_approx(value: float, tol: float = 1e-9) -> object:
    class _A:
        def __eq__(self, other: object) -> bool:
            return isinstance(other, float) and abs(other - value) <= tol
        def __repr__(self) -> str:
            return f"approx({value})"
    return _A()
```

- [ ] **Step 2: Run test, confirm RED**

Run: `uv run pytest tests/unit/eval/metrics/test_tool_accuracy.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `ToolAccuracy`**

`src/ariadne_eval/eval/metrics/tool_accuracy.py`:

```python
"""Tool-call accuracy metric."""

from __future__ import annotations

import json
from collections import Counter
from typing import Literal

from ariadne_eval.core.trajectory import (
    JsonValue,
    Step,
    ToolCallPayload,
    Trajectory,
)
from ariadne_eval.eval.case import Case, ExpectedTool
from ariadne_eval.eval.metrics.base import MetricResult

__all__ = ["ToolAccuracy"]


def _canon_args(args: dict[str, JsonValue]) -> str:
    return json.dumps(args, sort_keys=True, default=str)


def _label(score: float) -> Literal["pass", "fail", "partial"]:
    if score >= 0.99:
        return "pass"
    if score <= 0.01:
        return "fail"
    return "partial"


class ToolAccuracy:
    """Score how well the agent's tool calls match the expected ones."""

    name: str

    def __init__(
        self,
        mode: Literal["set", "ordered_prefix"] = "set",
        *,
        match_args: bool = False,
        name: str = "tool_accuracy",
    ) -> None:
        self._mode = mode
        self._match_args = match_args
        self.name = name

    def score(
        self, trajectory: Trajectory, steps: list[Step], case: Case
    ) -> MetricResult:
        actual = sorted(
            (s for s in steps if isinstance(s.payload, ToolCallPayload)),
            key=lambda s: s.started_at,
        )
        actual_payloads: list[ToolCallPayload] = [
            s.payload for s in actual if isinstance(s.payload, ToolCallPayload)
        ]

        if self._mode == "set":
            return self._score_set(trajectory, actual_payloads, case)
        return self._score_ordered_prefix(trajectory, actual_payloads, case)

    def _score_set(
        self,
        trajectory: Trajectory,
        actual: list[ToolCallPayload],
        case: Case,
    ) -> MetricResult:
        expected_keys = [self._expected_key(t) for t in case.expected_tools]
        actual_keys = [self._actual_key(p) for p in actual]

        # Per-tool wildcard handling for match_args=True with args=None:
        # remove a single matching actual call by name only.
        remaining_actual = list(actual_keys)
        matched: list[str] = []
        missing: list[str] = []
        for exp_key, exp_tool in zip(expected_keys, case.expected_tools, strict=True):
            if exp_key in remaining_actual:
                remaining_actual.remove(exp_key)
                matched.append(exp_tool.name)
            elif self._match_args and exp_tool.args is None:
                # name-only fallback
                wildcard_hit = next(
                    (k for k in remaining_actual if k.startswith(f"{exp_tool.name}::")),
                    None,
                )
                if wildcard_hit is not None:
                    remaining_actual.remove(wildcard_hit)
                    matched.append(exp_tool.name)
                else:
                    missing.append(exp_tool.name)
            else:
                missing.append(exp_tool.name)

        extra_names = [k.split("::", 1)[0] for k in remaining_actual]

        tp = len(matched)
        fp = len(extra_names)
        fn = len(missing)
        if tp == 0 and (fp > 0 or fn > 0):
            f1 = 0.0
            precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
            recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        elif tp == 0 and fp == 0 and fn == 0:
            f1 = 1.0
            precision = 1.0
            recall = 1.0
        else:
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        details: dict[str, JsonValue] = {
            "precision": precision,
            "recall": recall,
            "matched": matched,
            "missing": missing,
            "extra": extra_names,
        }
        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            trajectory_id=trajectory.id,
            score=f1,
            label=_label(f1),
            details=details,
        )

    def _score_ordered_prefix(
        self,
        trajectory: Trajectory,
        actual: list[ToolCallPayload],
        case: Case,
    ) -> MetricResult:
        expected = case.expected_tools
        prefix_len = 0
        first_divergence: int | None = None
        for i, exp in enumerate(expected):
            if i >= len(actual):
                first_divergence = i
                break
            if not self._matches(exp, actual[i]):
                first_divergence = i
                break
            prefix_len += 1

        score = 1.0 if not expected else prefix_len / len(expected)

        details: dict[str, JsonValue] = {
            "prefix_length": prefix_len,
            "expected_length": len(expected),
            "first_divergence_index": first_divergence,
        }
        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            trajectory_id=trajectory.id,
            score=score,
            label=_label(score),
            details=details,
        )

    def _expected_key(self, t: ExpectedTool) -> str:
        if not self._match_args or t.args is None:
            return f"{t.name}::*"
        return f"{t.name}::{_canon_args(t.args)}"

    def _actual_key(self, p: ToolCallPayload) -> str:
        if not self._match_args:
            return f"{p.tool_name}::*"
        return f"{p.tool_name}::{_canon_args(p.arguments)}"

    def _matches(self, exp: ExpectedTool, act: ToolCallPayload) -> bool:
        if exp.name != act.tool_name:
            return False
        if not self._match_args or exp.args is None:
            return True
        return _canon_args(exp.args) == _canon_args(act.arguments)
```

- [ ] **Step 4: Run test, confirm GREEN**

Run: `uv run pytest tests/unit/eval/metrics/test_tool_accuracy.py -v`
Expected: 11 passed.

- [ ] **Step 5: Type-check and lint**

Run: `uv run mypy --strict src/ariadne_eval/eval/metrics/tool_accuracy.py && uv run ruff check src/ariadne_eval/eval/metrics/tool_accuracy.py && uv run ruff format --check src/ariadne_eval/eval/metrics/tool_accuracy.py`

If `ruff format --check` fails, run `uv run ruff format src/ariadne_eval/eval/metrics/tool_accuracy.py` and re-check.

- [ ] **Step 6: Commit**

```bash
git add src/ariadne_eval/eval/metrics/tool_accuracy.py tests/unit/eval/metrics/test_tool_accuracy.py
git commit -m "feat(eval): add ToolAccuracy metric (set / ordered_prefix modes)"
```

---

## Task 7: `StepEfficiency`

**Files:**
- Create: `src/ariadne_eval/eval/metrics/efficiency.py`
- Create: `tests/unit/eval/metrics/test_efficiency.py`

- [ ] **Step 1: Write failing test**

`tests/unit/eval/metrics/test_efficiency.py`:

```python
from __future__ import annotations

import pytest

from ariadne_eval.eval.case import Case
from ariadne_eval.eval.errors import MissingReferenceError
from ariadne_eval.eval.metrics.efficiency import StepEfficiency
from tests.unit.eval._factories import make_tool_step, make_trajectory


def test_under_budget_pass() -> None:
    traj = make_trajectory()
    steps = [make_tool_step(trajectory_id=traj.id, tool_name="a")]
    case = Case(case_id="c", task="t", expected_max_steps=3)
    r = StepEfficiency().score(traj, steps, case)
    assert r.score == 1.0
    assert r.label == "pass"
    assert r.details == {"actual_steps": 1, "expected_max_steps": 3}


def test_at_budget_pass() -> None:
    traj = make_trajectory()
    steps = [
        make_tool_step(trajectory_id=traj.id, tool_name="a"),
        make_tool_step(trajectory_id=traj.id, tool_name="b"),
        make_tool_step(trajectory_id=traj.id, tool_name="c"),
    ]
    case = Case(case_id="c", task="t", expected_max_steps=3)
    r = StepEfficiency().score(traj, steps, case)
    assert r.score == 1.0
    assert r.label == "pass"


def test_over_budget_partial() -> None:
    traj = make_trajectory()
    steps = [make_tool_step(trajectory_id=traj.id, tool_name=n) for n in "abcd"]
    case = Case(case_id="c", task="t", expected_max_steps=2)
    r = StepEfficiency().score(traj, steps, case)
    assert r.score == 0.5
    assert r.label == "partial"


def test_zero_steps_with_budget() -> None:
    traj = make_trajectory()
    case = Case(case_id="c", task="t", expected_max_steps=2)
    r = StepEfficiency().score(traj, [], case)
    # 0 actual steps => max(actual,1)=1 => score = min(1, 2/1) = 1.0, pass
    assert r.score == 1.0
    assert r.label == "pass"


def test_missing_reference_raises() -> None:
    traj = make_trajectory()
    case = Case(case_id="c", task="t")
    with pytest.raises(MissingReferenceError):
        StepEfficiency().score(traj, [], case)
```

- [ ] **Step 2: Run test, confirm RED**

Run: `uv run pytest tests/unit/eval/metrics/test_efficiency.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `StepEfficiency`**

`src/ariadne_eval/eval/metrics/efficiency.py`:

```python
"""Step-efficiency metric."""

from __future__ import annotations

from ariadne_eval.core.trajectory import JsonValue, Step, Trajectory
from ariadne_eval.eval.case import Case
from ariadne_eval.eval.errors import MissingReferenceError
from ariadne_eval.eval.metrics.base import MetricResult

__all__ = ["StepEfficiency"]


class StepEfficiency:
    """Score = min(1, expected_max_steps / max(actual_steps, 1))."""

    name: str

    def __init__(self, *, name: str = "step_efficiency") -> None:
        self.name = name

    def score(
        self, trajectory: Trajectory, steps: list[Step], case: Case
    ) -> MetricResult:
        if case.expected_max_steps is None:
            raise MissingReferenceError("expected_max_steps", case_id=case.case_id)

        actual = len(steps)
        budget = case.expected_max_steps
        score = min(1.0, budget / max(actual, 1))
        label = "pass" if actual <= budget else "partial"
        details: dict[str, JsonValue] = {
            "actual_steps": actual,
            "expected_max_steps": budget,
        }
        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            trajectory_id=trajectory.id,
            score=score,
            label=label,
            details=details,
        )
```

- [ ] **Step 4: Run test, confirm GREEN**

Run: `uv run pytest tests/unit/eval/metrics/test_efficiency.py -v`
Expected: 5 passed.

- [ ] **Step 5: Type-check and lint**

Run: `uv run mypy --strict src/ariadne_eval/eval/metrics/efficiency.py && uv run ruff check src/ariadne_eval/eval/metrics/efficiency.py && uv run ruff format --check src/ariadne_eval/eval/metrics/efficiency.py`

- [ ] **Step 6: Commit**

```bash
git add src/ariadne_eval/eval/metrics/efficiency.py tests/unit/eval/metrics/test_efficiency.py
git commit -m "feat(eval): add StepEfficiency metric"
```

---

## Task 8: Bootstrap CI

**Files:**
- Create: `src/ariadne_eval/eval/stats/bootstrap.py`
- Create: `tests/unit/eval/stats/test_bootstrap.py`

- [ ] **Step 1: Write failing test**

`tests/unit/eval/stats/test_bootstrap.py`:

```python
from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, seed, settings
from hypothesis import strategies as st

from ariadne_eval.eval.errors import BootstrapInsufficientDataWarning
from ariadne_eval.eval.stats.bootstrap import BootstrapCI, bootstrap_mean_ci


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
```

- [ ] **Step 2: Run test, confirm RED**

Run: `uv run pytest tests/unit/eval/stats/test_bootstrap.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement bootstrap**

`src/ariadne_eval/eval/stats/bootstrap.py`:

```python
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
```

- [ ] **Step 4: Run test, confirm GREEN**

Run: `uv run pytest tests/unit/eval/stats/test_bootstrap.py -v`
Expected: 8 passed (the property test counts as one).

- [ ] **Step 5: Type-check and lint**

Run: `uv run mypy --strict src/ariadne_eval/eval/stats/bootstrap.py && uv run ruff check src/ariadne_eval/eval/stats/bootstrap.py && uv run ruff format --check src/ariadne_eval/eval/stats/bootstrap.py`

- [ ] **Step 6: Commit**

```bash
git add src/ariadne_eval/eval/stats/bootstrap.py tests/unit/eval/stats/test_bootstrap.py
git commit -m "feat(eval): add percentile bootstrap_mean_ci with seeded reproducibility"
```

---

## Task 9: `Runner` and `EvalReport`

**Files:**
- Create: `src/ariadne_eval/eval/runner.py`
- Create: `tests/unit/eval/test_runner.py`

- [ ] **Step 1: Write failing test**

`tests/unit/eval/test_runner.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from ariadne_eval.eval.case import Case, ExpectedTool
from ariadne_eval.eval.errors import MissingReferenceError
from ariadne_eval.eval.metrics.efficiency import StepEfficiency
from ariadne_eval.eval.metrics.final_answer import FinalAnswerMatch
from ariadne_eval.eval.metrics.tool_accuracy import ToolAccuracy
from ariadne_eval.eval.runner import EvalReport, Runner
from tests.unit.eval._factories import make_tool_step, make_trajectory


def _three_pairs() -> list[tuple[object, list[object], Case]]:
    pairs: list[tuple[object, list[object], Case]] = []
    for i, (ans, tools, budget) in enumerate(
        [("4", ["calc"], 2), ("5", ["calc"], 2), ("4", ["calc", "search"], 2)]
    ):
        traj = make_trajectory(final_answer=ans)
        steps = [make_tool_step(trajectory_id=traj.id, tool_name=t) for t in tools]
        case = Case(
            case_id=f"c{i}",
            task="t",
            expected_answer="4",
            expected_tools=(ExpectedTool(name="calc"),),
            expected_max_steps=budget,
        )
        pairs.append((traj, steps, case))
    return pairs


def test_runner_evaluates_three_metrics_with_aggregates() -> None:
    runner = Runner(
        metrics=[FinalAnswerMatch(), ToolAccuracy(), StepEfficiency()],
        seed=0,
        n_resamples=200,
    )
    report = runner.evaluate(_three_pairs())  # type: ignore[arg-type]
    assert isinstance(report, EvalReport)
    assert report.n_cases == 3
    assert report.seed == 0
    # 3 cases × 3 metrics
    assert len(report.results) == 9
    assert set(report.aggregates) == {
        "final_answer_match",
        "tool_accuracy",
        "step_efficiency",
    }
    # final_answer_match: 2/3 pass => mean ≈ 0.667
    fa = report.aggregates["final_answer_match"]
    assert abs(fa.mean - 2 / 3) < 1e-9
    assert fa.n == 3


def test_runner_skip_on_missing_reference() -> None:
    traj = make_trajectory(final_answer="x")
    case_with = Case(case_id="c1", task="t", expected_answer="x")
    case_without = Case(case_id="c2", task="t")  # no expected_answer
    runner = Runner(metrics=[FinalAnswerMatch()], seed=0, n_resamples=100)
    report = runner.evaluate(
        [(traj, [], case_with), (traj, [], case_without)],
    )
    # Only c1 produced a result
    assert len(report.results) == 1
    assert report.results[0].case_id == "c1"
    assert report.aggregates["final_answer_match"].n == 1


def test_runner_error_on_missing_reference() -> None:
    traj = make_trajectory(final_answer="x")
    case_without = Case(case_id="c", task="t")
    runner = Runner(
        metrics=[FinalAnswerMatch()], on_missing_reference="error"
    )
    with pytest.raises(MissingReferenceError):
        runner.evaluate([(traj, [], case_without)])


def test_eval_report_jsonl_round_trip(tmp_path: Path) -> None:
    runner = Runner(
        metrics=[FinalAnswerMatch(), StepEfficiency()],
        seed=3,
        n_resamples=200,
    )
    report = runner.evaluate(_three_pairs())  # type: ignore[arg-type]
    out = tmp_path / "report.jsonl"
    report.to_jsonl(out)
    loaded = EvalReport.from_jsonl(out)
    assert loaded == report
```

- [ ] **Step 2: Run test, confirm RED**

Run: `uv run pytest tests/unit/eval/test_runner.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `Runner` and `EvalReport`**

`src/ariadne_eval/eval/runner.py`:

```python
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
    def from_jsonl(cls, path: str | Path) -> "EvalReport":
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
                        k: BootstrapCI.model_validate(v)
                        for k, v in obj["aggregates"].items()
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
        self._metrics = list(metrics)
        self._seed = seed
        self._n_resamples = n_resamples
        self._confidence = confidence
        self._on_missing = on_missing_reference

    def evaluate(
        self,
        items: Iterable[tuple[Trajectory, list[Step], Case]],
    ) -> EvalReport:
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
```

- [ ] **Step 4: Run test, confirm GREEN**

Run: `uv run pytest tests/unit/eval/test_runner.py -v`
Expected: 4 passed.

- [ ] **Step 5: Type-check and lint**

Run: `uv run mypy --strict src/ariadne_eval/eval/runner.py && uv run ruff check src/ariadne_eval/eval/runner.py && uv run ruff format --check src/ariadne_eval/eval/runner.py`

- [ ] **Step 6: Commit**

```bash
git add src/ariadne_eval/eval/runner.py tests/unit/eval/test_runner.py
git commit -m "feat(eval): add Runner and EvalReport with JSONL round-trip"
```

---

## Task 10: Wire up public re-exports

**Files:**
- Modify: `src/ariadne_eval/eval/__init__.py`
- Modify: `src/ariadne_eval/__init__.py`
- Create: `tests/unit/eval/test_public_api.py`

- [ ] **Step 1: Write failing test**

`tests/unit/eval/test_public_api.py`:

```python
from __future__ import annotations

import ariadne_eval
from ariadne_eval import (
    BootstrapCI,
    Case,
    EvalReport,
    ExpectedTool,
    FinalAnswerMatch,
    Metric,
    MetricResult,
    MissingReferenceError,
    Runner,
    StepEfficiency,
    ToolAccuracy,
    bootstrap_mean_ci,
)


def test_top_level_exports_resolve() -> None:
    for name in [
        "BootstrapCI",
        "Case",
        "EvalReport",
        "ExpectedTool",
        "FinalAnswerMatch",
        "Metric",
        "MetricResult",
        "MissingReferenceError",
        "Runner",
        "StepEfficiency",
        "ToolAccuracy",
        "bootstrap_mean_ci",
    ]:
        assert name in ariadne_eval.__all__, f"missing from __all__: {name}"
        assert getattr(ariadne_eval, name) is not None


def test_namespaced_eval_module_also_exposes_them() -> None:
    from ariadne_eval import eval as ev

    assert ev.Case is Case
    assert ev.Runner is Runner
    assert ev.bootstrap_mean_ci is bootstrap_mean_ci
```

- [ ] **Step 2: Run test, confirm RED**

Run: `uv run pytest tests/unit/eval/test_public_api.py -v`
Expected: FAIL — symbols not exported.

- [ ] **Step 3: Update `src/ariadne_eval/eval/__init__.py`**

Replace contents:

```python
"""Evaluation: metrics, judges, statistical aggregation."""

from __future__ import annotations

from ariadne_eval.eval.case import Case, ExpectedTool
from ariadne_eval.eval.errors import (
    BootstrapInsufficientDataWarning,
    MissingReferenceError,
)
from ariadne_eval.eval.metrics.base import Metric, MetricResult
from ariadne_eval.eval.metrics.efficiency import StepEfficiency
from ariadne_eval.eval.metrics.final_answer import FinalAnswerMatch
from ariadne_eval.eval.metrics.tool_accuracy import ToolAccuracy
from ariadne_eval.eval.runner import EvalReport, Runner
from ariadne_eval.eval.stats.bootstrap import BootstrapCI, bootstrap_mean_ci

__all__ = [
    "BootstrapCI",
    "BootstrapInsufficientDataWarning",
    "Case",
    "EvalReport",
    "ExpectedTool",
    "FinalAnswerMatch",
    "Metric",
    "MetricResult",
    "MissingReferenceError",
    "Runner",
    "StepEfficiency",
    "ToolAccuracy",
    "bootstrap_mean_ci",
]
```

- [ ] **Step 4: Update `src/ariadne_eval/__init__.py`**

Add imports (alphabetized into the existing import block; preserve existing structure):

```python
from ariadne_eval.eval.case import Case, ExpectedTool
from ariadne_eval.eval.errors import (
    BootstrapInsufficientDataWarning,
    MissingReferenceError,
)
from ariadne_eval.eval.metrics.base import Metric, MetricResult
from ariadne_eval.eval.metrics.efficiency import StepEfficiency
from ariadne_eval.eval.metrics.final_answer import FinalAnswerMatch
from ariadne_eval.eval.metrics.tool_accuracy import ToolAccuracy
from ariadne_eval.eval.runner import EvalReport, Runner
from ariadne_eval.eval.stats.bootstrap import BootstrapCI, bootstrap_mean_ci
```

And merge into the existing `__all__` list (alphabetized) the following new names:

```
"BootstrapCI",
"BootstrapInsufficientDataWarning",
"Case",
"EvalReport",
"ExpectedTool",
"FinalAnswerMatch",
"Metric",
"MetricResult",
"MissingReferenceError",
"Runner",
"StepEfficiency",
"ToolAccuracy",
"bootstrap_mean_ci",
```

After editing, the file's `__all__` should remain alphabetized end-to-end.

- [ ] **Step 5: Run test, confirm GREEN**

Run: `uv run pytest tests/unit/eval/test_public_api.py -v`
Expected: 2 passed.

- [ ] **Step 6: Run full fast suite + mypy + ruff**

Run: `uv run pytest -m "fast and not integration" -q && uv run mypy --strict src/ariadne_eval && uv run ruff check src/ariadne_eval tests && uv run ruff format --check src/ariadne_eval tests`
Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add src/ariadne_eval/eval/__init__.py src/ariadne_eval/__init__.py tests/unit/eval/test_public_api.py
git commit -m "feat(eval): expose Case/Runner/metrics/bootstrap on the public API"
```

---

## Task 11: Concept docs page

**Files:**
- Create: `docs/concepts/metrics.md`
- Modify: `mkdocs.yml`

- [ ] **Step 1: Write `docs/concepts/metrics.md`**

```markdown
# Metrics

A *metric* in `ariadne-eval` is a pure-compute function from a
`(Trajectory, list[Step], Case)` triple to a `MetricResult` (a
score, an optional pass/fail/partial label, and a `details` dict).
The scoring is deterministic; aggregation across many cases is a
separate layer that wraps every reported number with a 95%
percentile-bootstrap CI.

## Built-in metrics

### `FinalAnswerMatch`

Compares `trajectory.final_answer` against `case.expected_answer`.

| Comparator | Behavior |
|---|---|
| `"normalized_exact"` (default) | Lowercase, strip, collapse whitespace, then equality. Returns `0.0` or `1.0`. |
| `"exact"` | Byte-for-byte equality. Returns `0.0` or `1.0`. |
| Callable `(actual, expected) -> float` | Caller-defined score in `[0, 1]`; label is `pass` (`>= 0.99`), `fail` (`<= 0.01`), or `partial`. |

If the trajectory has no final answer, the result is `score=0.0`,
`label="fail"`, `details={"reason": "no_final_answer"}`. If the case has
no `expected_answer`, the metric raises `MissingReferenceError` (the
`Runner` honors this — see "Missing references" below).

### `ToolAccuracy`

Walks the supplied steps for `ToolCallPayload`s in `started_at` order and
compares them against `case.expected_tools`.

- `mode="set"` (default): F1 score over the multisets of tool calls.
  Details include `precision`, `recall`, `matched`, `missing`, `extra`.
- `mode="ordered_prefix"`: longest matching prefix divided by the
  expected length. Details include `prefix_length`,
  `expected_length`, `first_divergence_index`.

`match_args=True` strictens equality to also compare the JSON-canonical
arguments dict. An `ExpectedTool` with `args=None` becomes a per-tool
wildcard in this mode (any actual call of the same name matches).

> **Documented quirk.** With `mode="ordered_prefix"` and an empty
> `expected_tools`, the score is always `1.0` — an empty prefix matches
> everything. If you want "no tools allowed," use `mode="set"`.

### `StepEfficiency`

`min(1.0, expected_max_steps / max(actual_steps, 1))`. Label is `pass`
when actual ≤ budget, otherwise `partial` (going over budget is a smell,
not a correctness failure).

## Confidence intervals

Every aggregate in an `EvalReport` is a `BootstrapCI` produced by
`bootstrap_mean_ci`. The implementation is a standard percentile
bootstrap on the sample mean: `n_resamples` resamples with replacement,
empirical α/2 and 1−α/2 percentiles. Defaults are 1000 resamples and
95% confidence.

Reproducibility is guaranteed: identical input + identical seed produces
identical CIs.

Edge cases:

- `n=0`: NaN CI plus a `BootstrapInsufficientDataWarning`.
- `n=1`: degenerate CI equal to the value plus the same warning.

We use the percentile bootstrap (not BCa) deliberately. For bounded
scores in `[0, 1]` the bias is small; the function signature is stable,
so we can move to `scipy.stats.bootstrap` later without breaking callers.

## Missing references

A metric like `FinalAnswerMatch` needs `case.expected_answer`. If a case
doesn't carry that field, the metric raises `MissingReferenceError`. The
`Runner` has two modes:

- `on_missing_reference="skip"` (default): silently omit that
  `(metric, case)` pair from the report. Aggregates cover only the cases
  the metric could score (`BootstrapCI.n` reflects this).
- `on_missing_reference="error"`: re-raise the exception. Use this in CI
  to catch malformed benchmarks early.

## Writing your own metric

A metric is anything that satisfies the `Metric` protocol — a `name`
attribute and a `score(trajectory, steps, case)` method that returns a
`MetricResult`. See `examples/03_custom_metric/` for a worked example.
```

- [ ] **Step 2: Add the new page to `mkdocs.yml` nav**

Edit `mkdocs.yml`. The current Concepts block is:

```yaml
  - Concepts:
      - concepts/index.md
      - Trajectory model: concepts/trajectory.md
      - Storage: concepts/storage.md
      - Tracing: concepts/tracing.md
```

Insert `      - Metrics: concepts/metrics.md` so the block reads (alphabetical by display name after the index page):

```yaml
  - Concepts:
      - concepts/index.md
      - Metrics: concepts/metrics.md
      - Storage: concepts/storage.md
      - Tracing: concepts/tracing.md
      - Trajectory model: concepts/trajectory.md
```

- [ ] **Step 3: Build the docs to verify**

Run: `uv run mkdocs build --strict`
Expected: build succeeds, no broken links.

- [ ] **Step 4: Commit**

```bash
git add docs/concepts/metrics.md mkdocs.yml
git commit -m "docs(concepts): add metrics page covering scoring, CIs, missing-reference policy"
```

---

## Task 12: API reference page

**Files:**
- Create: `docs/reference/eval.md`
- Modify: `mkdocs.yml` (only if reference nav needs the new entry)

- [ ] **Step 1: Write `docs/reference/eval.md`**

```markdown
# `ariadne_eval.eval`

::: ariadne_eval.eval.case
    options:
      show_root_heading: true

::: ariadne_eval.eval.metrics.base
    options:
      show_root_heading: true

::: ariadne_eval.eval.metrics.final_answer
    options:
      show_root_heading: true

::: ariadne_eval.eval.metrics.tool_accuracy
    options:
      show_root_heading: true

::: ariadne_eval.eval.metrics.efficiency
    options:
      show_root_heading: true

::: ariadne_eval.eval.stats.bootstrap
    options:
      show_root_heading: true

::: ariadne_eval.eval.runner
    options:
      show_root_heading: true

::: ariadne_eval.eval.errors
    options:
      show_root_heading: true
```

- [ ] **Step 2: Add to `mkdocs.yml` nav**

Edit `mkdocs.yml`. The current Reference block starts:

```yaml
  - Reference:
      - reference/index.md
      - Tracing: reference/tracing.md
```

Insert `      - Eval: reference/eval.md` so the block reads:

```yaml
  - Reference:
      - reference/index.md
      - Eval: reference/eval.md
      - Tracing: reference/tracing.md
```

- [ ] **Step 3: Build the docs**

Run: `uv run mkdocs build --strict`
Expected: build succeeds; the eval page renders symbols.

- [ ] **Step 4: Commit**

```bash
git add docs/reference/eval.md mkdocs.yml
git commit -m "docs(reference): add eval API reference page"
```

---

## Task 13: Custom-metric example

**Files:**
- Create: `examples/03_custom_metric/main.py`
- Create: `examples/03_custom_metric/README.md`

- [ ] **Step 1: Write `examples/03_custom_metric/main.py`**

```python
"""Worked example: write a custom Metric and run it through the Runner.

Run with:
    uv run python examples/03_custom_metric/main.py
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ariadne_eval import (
    Case,
    ExpectedTool,
    FinalAnswerMatch,
    Metric,
    MetricResult,
    Runner,
    StepEfficiency,
    ToolAccuracy,
    new_id,
)
from ariadne_eval.core.status import StepStatus, TrajectoryStatus
from ariadne_eval.core.trajectory import (
    Step,
    ToolCallPayload,
    Trajectory,
)


class FinalAnswerLength(Metric):
    """A toy custom metric: closer to expected length is better."""

    name = "final_answer_length"

    def score(
        self, trajectory: Trajectory, steps: list[Step], case: Case
    ) -> MetricResult:
        actual = trajectory.final_answer or ""
        actual_len = len(actual) if isinstance(actual, str) else 0
        target = case.metadata.get("target_length", 1)
        target_int = int(target) if isinstance(target, (int, float, str)) else 1
        score = max(0.0, 1.0 - abs(actual_len - target_int) / max(target_int, 1))
        return MetricResult(
            metric=self.name,
            case_id=case.case_id,
            trajectory_id=trajectory.id,
            score=score,
            details={"actual_length": actual_len, "target_length": target_int},
        )


def _make_traj(answer: str) -> Trajectory:
    started = datetime(2026, 5, 14, 12, 0, 0, tzinfo=UTC)
    return Trajectory(
        id=new_id(),
        task="demo",
        agent_name="example",
        agent_version="0.0.0",
        model_id="example/model",
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        final_status=TrajectoryStatus.COMPLETED,
        final_answer=answer,
    )


def _make_step(traj_id: str, tool: str) -> Step:
    started = datetime(2026, 5, 14, 12, 0, 0, tzinfo=UTC)
    return Step(
        id=new_id(),
        trajectory_id=traj_id,
        parent_step_id=None,
        name=f"tool:{tool}",
        started_at=started,
        finished_at=started + timedelta(milliseconds=5),
        status=StepStatus.OK,
        payload=ToolCallPayload(
            tool_name=tool, arguments={}, result=None, latency_ms=5.0
        ),
    )


def main() -> None:
    pairs = []
    for i, (answer, tools) in enumerate(
        [("4", ["calc"]), ("forty-two", ["calc", "search"]), ("4", ["calc"])]
    ):
        traj = _make_traj(answer)
        steps = [_make_step(traj.id, t) for t in tools]
        case = Case(
            case_id=f"c{i}",
            task="what is 2+2?",
            expected_answer="4",
            expected_tools=(ExpectedTool(name="calc"),),
            expected_max_steps=1,
            metadata={"target_length": 1},
        )
        pairs.append((traj, steps, case))

    runner = Runner(
        metrics=[
            FinalAnswerMatch(),
            ToolAccuracy(),
            StepEfficiency(),
            FinalAnswerLength(),
        ],
        seed=0,
        n_resamples=500,
    )
    report = runner.evaluate(pairs)

    print(f"n_cases = {report.n_cases}")
    for name, ci in report.aggregates.items():
        print(f"  {name:24s} mean={ci.mean:.3f}  95% CI=[{ci.lo:.3f}, {ci.hi:.3f}]  n={ci.n}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `examples/03_custom_metric/README.md`**

```markdown
# Example 03: Custom metric

Shows how to write a `Metric` (a `name` attribute + a `score()` method) and
feed it through `Runner` alongside the built-in metrics. The example
constructs three synthetic trajectories in-process — no network, no API
key.

## Run

```bash
uv run python examples/03_custom_metric/main.py
```

## Expected output (approximate)

```
n_cases = 3
  final_answer_match       mean=0.667  95% CI=[0.333, 1.000]  n=3
  tool_accuracy            mean=0.889  95% CI=[0.667, 1.000]  n=3
  step_efficiency          mean=0.778  95% CI=[0.500, 1.000]  n=3
  final_answer_length      mean=0.667  95% CI=[0.000, 1.000]  n=3
```

The CI bounds depend on the bootstrap seed; the means do not.
```

- [ ] **Step 3: Run the example**

Run: `uv run python examples/03_custom_metric/main.py`
Expected: prints `n_cases = 3` and four metric lines with CIs. No errors.

- [ ] **Step 4: Type-check and lint**

Run: `uv run mypy --strict examples/03_custom_metric/main.py && uv run ruff check examples/03_custom_metric && uv run ruff format --check examples/03_custom_metric`

If `ruff format --check` complains, run `uv run ruff format examples/03_custom_metric` and re-check.

- [ ] **Step 5: Commit**

```bash
git add examples/03_custom_metric/main.py examples/03_custom_metric/README.md
git commit -m "docs(examples): custom-metric walkthrough using Runner + bootstrap CIs"
```

---

## Task 14: CHANGELOG entry + final verification

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Append to `CHANGELOG.md` under `## [Unreleased]`**

Add (under the existing Unreleased section, in the `### Added` subsection — create the subsection if absent):

```markdown
### Added

- `ariadne_eval.eval` namespace: `Case`, `ExpectedTool`, `Metric`,
  `MetricResult`, `FinalAnswerMatch`, `ToolAccuracy`, `StepEfficiency`,
  `Runner`, `EvalReport`, `bootstrap_mean_ci`, `BootstrapCI`,
  `MissingReferenceError`, `BootstrapInsufficientDataWarning`. All
  re-exported from the top-level `ariadne_eval`.
- `docs/concepts/metrics.md` and `docs/reference/eval.md`.
- `examples/03_custom_metric/` walkthrough.
```

- [ ] **Step 2: Run the full fast suite, mypy, ruff, mkdocs**

Run all of these in order; each must succeed:

```bash
uv run pytest -m "fast and not integration" --cov=src/ariadne_eval/eval --cov-report=term-missing -q
uv run mypy --strict src/ariadne_eval
uv run ruff check src/ariadne_eval tests examples
uv run ruff format --check src/ariadne_eval tests examples
uv run mkdocs build --strict
```

Expected:
- pytest: all green; coverage on `src/ariadne_eval/eval` ≥ 90%.
- mypy: clean.
- ruff: clean.
- mkdocs: build succeeds with no warnings.

If coverage on a specific eval file is below 90%, add a targeted unit test for the missing branch and re-run.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): unreleased entry for the eval metrics module"
```

- [ ] **Step 4: Merge to main with a no-ff merge and tag**

(Per `phase_workflow` memory: feature branch → `--no-ff` merge → `v0.0.N-alpha` tag.)

```bash
git checkout main
git merge --no-ff phase-5-metrics -m "Merge branch 'phase-5-metrics' into main"
git tag -a v0.0.6-alpha -m "Phase 5: deterministic metrics + bootstrap CIs"
```

(Do NOT push without explicit user approval.)

- [ ] **Step 5: Update `current_phase_state.md` memory**

Update `/Users/rish/.claude/projects/-Users-rish-Desktop-AI-Projects-ariadne-eval/memory/current_phase_state.md` to reflect: on `main`, last tag `v0.0.6-alpha` (Phase 5), next up is Phase 6 judges.

---

## Self-review notes

- **Spec coverage:** Every spec section maps to a task. `Case`/`ExpectedTool` (T2), errors (T3), `MetricResult`/Protocol (T4), three metrics (T5–T7), bootstrap (T8), `Runner`/`EvalReport` + JSONL (T9), public re-exports (T10), concept docs (T11), API ref (T12), example (T13), CHANGELOG + verification (T14).
- **Type consistency:** `score(trajectory, steps, case)` signature appears identically in the Protocol (T4) and every metric (T5–T7), and the Runner calls it with the same arity (T9). All metric names (`final_answer_match`, `tool_accuracy`, `step_efficiency`) are consistent between metric defaults and the public-API test (T10) and the CHANGELOG entry (T14).
- **No placeholders:** every step has either complete code or an exact command. The mkdocs.yml edits in T11/T12 are the only "find this and add that" steps; they include the inspection command and the exact line to add.
