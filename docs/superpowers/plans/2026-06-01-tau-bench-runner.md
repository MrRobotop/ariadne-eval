# Phase 7 — tau-bench Benchmark Runner: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Sierra's τ-bench wrapped behind a tau-agnostic `Benchmark` Protocol, score captured trajectories with the existing eval Runner + `PlanQuality`, and write a reproducible result bundle for one canonical run: 2 agent models × 50 retail tasks at ≤$5 Anthropic budget.

**Architecture:** Add `tau-bench` as an optional `[tau-bench]` git-installable extra. Lazy-import inside `src/ariadne_eval/benchmarks/tau_bench.py`. Wrap `agent.solve()` in our tracing context, convert tau-bench's `EnvRunResult.traj` into the ariadne `Trajectory` schema, persist via the existing DuckDB store, run the existing `Runner.aevaluate` over the captured trajectories. Hard Rule #3 (no framework-specific deps in core) honored via the extra + lazy import + actionable ImportError.

**Tech Stack:** Python 3.11+, Pydantic v2, tau-bench (git, pinned to SHA `59a200c6d575d595120f1cb70fea53cef0632f6b`), litellm, asyncio.TaskGroup, Click, PyYAML, DuckDB (existing), pytest. No new dev dependencies.

---

## File map

| File | Action | Purpose |
|---|---|---|
| `src/ariadne_eval/_transient.py` | Create | Extracted transient-error retry primitives (constants + `is_transient`) |
| `scripts/build_calibration_set.py` | Modify | Import from `_transient` instead of defining locally |
| `tests/unit/test_transient.py` | Create | Unit tests for `is_transient` over the 5 known class names |
| `src/ariadne_eval/benchmarks/__init__.py` | Create | Public re-exports: `Benchmark`, `BenchmarkTask`, `BenchmarkRunResult`, `BenchmarkConfig`, `BenchmarkRunner`, `BenchmarkReport`. NOT `TauBenchAdapter` (extra-gated). |
| `src/ariadne_eval/benchmarks/base.py` | Create | `Benchmark` Protocol + `BenchmarkTask` + `BenchmarkRunResult` dataclasses |
| `src/ariadne_eval/benchmarks/tau_bench.py` | Create | `TauBenchAdapter` (lazy-imports tau_bench) + `_convert_tau_traj` helper |
| `src/ariadne_eval/benchmarks/config.py` | Create | `BenchmarkConfig` Pydantic model + `load_benchmark_config(path)` YAML loader |
| `src/ariadne_eval/benchmarks/runner.py` | Create | `BenchmarkRunner` orchestrator + `BenchmarkReport` |
| `src/ariadne_eval/cli/bench.py` | Create | `ariadne bench run` Click subcommand |
| `src/ariadne_eval/cli/main.py` | Modify | Register `bench` subcommand on the main CLI group |
| `tests/unit/benchmarks/__init__.py` | Create | Package marker |
| `tests/unit/benchmarks/test_base.py` | Create | Tests: dataclasses, Protocol runtime check |
| `tests/unit/benchmarks/test_tau_bench_convert.py` | Create | Tests for `_convert_tau_traj`: 3 hand-crafted fixtures |
| `tests/unit/benchmarks/test_tau_bench.py` | Create | Adapter tests: ImportError without extra, monkeypatched `get_env` |
| `tests/unit/benchmarks/test_config.py` | Create | YAML load + Pydantic validation |
| `tests/unit/benchmarks/test_runner.py` | Create | End-to-end with `StubBenchmark` + `StubJudge`; summary.json shape |
| `tests/unit/cli/test_bench.py` | Create | `--dry-run` config validation; `--limit` override |
| `pyproject.toml` | Modify | Add `[tau-bench]` and `[bench]` optional extras; bump version |
| `configs/benchmarks/tau_retail_baseline.yaml` | Create | Headline run config |
| `docs/benchmarks/v0.0.9-alpha-tau-retail-50/` | Create (manual) | Result bundle: `config.yaml` + `trajectories.jsonl` + `summary.json` |
| `docs/concepts/benchmarks.md` | Create | Methodology page rendered from bundle |
| `docs/concepts/judges.md` | Modify | Optional cross-link to benchmarks page |
| `README.md` | Modify | Add headline benchmark table (2 rows) + reflect Phase 7 row in shipped table |
| `mkdocs.yml` | Modify | Add `Benchmarks: concepts/benchmarks.md` to Concepts nav |
| `CHANGELOG.md` | Modify | `[Unreleased]` entry for Phase 7 |
| `src/ariadne_eval/_version.py` | Modify | Bump to `0.0.9-alpha` |
| `tests/unit/test_smoke.py` | Modify | Update version assertions |

---

## Task 1: Extract `_transient` retry primitives

**Goal:** Move retry constants + `is_transient` helper from `scripts/build_calibration_set.py` into `src/ariadne_eval/_transient.py`. Both the existing calibration script and the new benchmark runner will import from it.

**Files:**
- Create: `src/ariadne_eval/_transient.py`
- Modify: `scripts/build_calibration_set.py`
- Create: `tests/unit/test_transient.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_transient.py`:

```python
"""Tests for the transient-error retry helpers."""

from __future__ import annotations

import pytest

from ariadne_eval._transient import (
    MAX_TRANSIENT_RETRIES,
    TRANSIENT_BACKOFF_BASE,
    TRANSIENT_EXC_NAMES,
    is_transient,
)

pytestmark = pytest.mark.fast


def test_constants_have_expected_values() -> None:
    assert MAX_TRANSIENT_RETRIES == 4
    assert TRANSIENT_BACKOFF_BASE == 2.0
    assert set(TRANSIENT_EXC_NAMES) == {
        "InternalServerError",
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "ServiceUnavailableError",
    }


def test_is_transient_recognizes_known_classes() -> None:
    class InternalServerError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    assert is_transient(InternalServerError("oops"))
    assert is_transient(RateLimitError("slow down"))


def test_is_transient_rejects_unrelated_exceptions() -> None:
    assert not is_transient(ValueError("nope"))
    assert not is_transient(RuntimeError("nope"))
    assert not is_transient(KeyError("nope"))


def test_is_transient_only_looks_at_class_name_not_inheritance() -> None:
    """Provider-portable identification — by class name string."""

    class InternalServerError(Exception):
        pass

    class WeirdSubclass(InternalServerError):
        pass

    # The subclass has a different name — it's NOT recognized as transient
    # unless we explicitly add it. This is intentional: provider class names
    # are stable contracts; subclasses are not.
    assert is_transient(InternalServerError("a"))
    assert not is_transient(WeirdSubclass("b"))
```

- [ ] **Step 2: Run test to confirm RED**

Run: `uv run pytest tests/unit/test_transient.py -q`
Expected: FAIL — `ariadne_eval._transient` module missing.

- [ ] **Step 3: Implement `src/ariadne_eval/_transient.py`**

Create:

```python
"""Transient-error retry primitives shared across LLM-calling components.

Provider APIs (Anthropic, OpenAI, Groq, etc.) occasionally return HTTP
5xx / rate-limit / connection errors on otherwise-valid calls. Bounded
exponential backoff handles those without polluting reports or aborting
long runs. Identification is by class name (provider-portable across
litellm exception types).
"""

from __future__ import annotations

__all__ = [
    "MAX_TRANSIENT_RETRIES",
    "TRANSIENT_BACKOFF_BASE",
    "TRANSIENT_EXC_NAMES",
    "is_transient",
]


MAX_TRANSIENT_RETRIES: int = 4
TRANSIENT_BACKOFF_BASE: float = 2.0  # seconds; doubles per attempt: 2, 4, 8, 16
TRANSIENT_EXC_NAMES: tuple[str, ...] = (
    "InternalServerError",
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "ServiceUnavailableError",
)


def is_transient(exc: BaseException) -> bool:
    """Return True if ``exc`` is a known provider-side transient error.

    Identifies by exact class name (no inheritance walk) so the contract
    is stable across litellm / openai / anthropic exception hierarchies.
    """
    return type(exc).__name__ in TRANSIENT_EXC_NAMES
```

- [ ] **Step 4: Update `scripts/build_calibration_set.py` to import from `_transient`**

Open `scripts/build_calibration_set.py`. Find the block of constants and `_is_transient`:

```python
# Transient-error retry policy. Anthropic and other providers occasionally
# return HTTP 5xx / rate-limit / connection errors on otherwise-valid calls.
# Bounded exponential backoff handles those without polluting the report.
_MAX_TRANSIENT_RETRIES = 4
_TRANSIENT_BACKOFF_BASE = 2.0  # seconds; doubles each attempt (2, 4, 8, 16)
_TRANSIENT_EXC_NAMES = (
    "InternalServerError",
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "ServiceUnavailableError",
)


def _is_transient(exc: Exception) -> bool:
    """Identify provider-side transient errors by class name (provider-portable)."""
    return type(exc).__name__ in _TRANSIENT_EXC_NAMES
```

Delete that block. Add a single import near the top of the file (after the existing `from ariadne_eval.eval.stats.agreement import cohens_kappa` line):

```python
from ariadne_eval._transient import (
    MAX_TRANSIENT_RETRIES as _MAX_TRANSIENT_RETRIES,
    TRANSIENT_BACKOFF_BASE as _TRANSIENT_BACKOFF_BASE,
    is_transient as _is_transient,
)
```

The `as _XXX` rebinds preserve the existing in-file underscore-prefixed names so the rest of the script's retry loop (which uses `_MAX_TRANSIENT_RETRIES`, `_TRANSIENT_BACKOFF_BASE`, `_is_transient`) does not need to change.

- [ ] **Step 5: Run tests + verify the calibration script's existing tests still pass**

```bash
uv run pytest tests/unit/test_transient.py -q
uv run pytest tests/unit/scripts/test_build_calibration_set.py -q
uv run pytest -m "fast and not integration" -q
```

Expected: all green. The calibration script's 11 tests should pass with no behavior change.

- [ ] **Step 6: Verify gates**

```bash
uv run mypy --strict src/ariadne_eval/_transient.py scripts/build_calibration_set.py
uv run ruff check src/ariadne_eval/_transient.py scripts/build_calibration_set.py tests/unit/test_transient.py
uv run ruff format src/ariadne_eval/_transient.py scripts/build_calibration_set.py tests/unit/test_transient.py
uv run ruff format --check src/ariadne_eval/_transient.py scripts/build_calibration_set.py tests/unit/test_transient.py
```

Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add src/ariadne_eval/_transient.py scripts/build_calibration_set.py tests/unit/test_transient.py
git commit -m "refactor(_transient): extract retry primitives to shared module"
```

---

## Task 2: `Benchmark` Protocol + dataclasses

**Goal:** A trajectory-agnostic benchmark contract. `BenchmarkTask` carries an id, instruction, and opaque payload. `BenchmarkRunResult` carries trajectory_id, success, raw_score, optional error. The `Benchmark` Protocol declares `tasks()` and async `run_task()`.

**Files:**
- Create: `src/ariadne_eval/benchmarks/__init__.py`
- Create: `src/ariadne_eval/benchmarks/base.py`
- Create: `tests/unit/benchmarks/__init__.py`
- Create: `tests/unit/benchmarks/test_base.py`

- [ ] **Step 1: Make the test package**

```bash
mkdir -p tests/unit/benchmarks
touch tests/unit/benchmarks/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/unit/benchmarks/test_base.py`:

```python
"""Tests for the Benchmark Protocol + dataclasses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from ariadne_eval.benchmarks.base import (
    Benchmark,
    BenchmarkRunResult,
    BenchmarkTask,
)
from ariadne_eval.core.trajectory import Step, Trajectory  # noqa: F401  (used in type hints)
from ariadne_eval.storage.base import Store  # noqa: F401

pytestmark = pytest.mark.fast


def test_benchmark_task_minimal() -> None:
    t = BenchmarkTask(task_id="t1", task_index=0, instruction="do X", payload={"raw": 1})
    assert t.task_id == "t1"
    assert t.task_index == 0
    assert t.instruction == "do X"
    assert t.payload == {"raw": 1}


def test_benchmark_task_is_frozen() -> None:
    t = BenchmarkTask(task_id="t1", task_index=0, instruction="i", payload=None)
    with pytest.raises(Exception):  # FrozenInstanceError
        t.task_id = "t2"  # type: ignore[misc]


def test_benchmark_run_result_minimal() -> None:
    r = BenchmarkRunResult(trajectory_id="01J0", success=True, raw_score=1.0)
    assert r.trajectory_id == "01J0"
    assert r.success is True
    assert r.raw_score == 1.0
    assert r.error is None


def test_benchmark_run_result_with_error() -> None:
    r = BenchmarkRunResult(
        trajectory_id="01J0",
        success=False,
        raw_score=0.0,
        error="provider timeout after 4 retries",
    )
    assert r.error == "provider timeout after 4 retries"


def test_benchmark_protocol_runtime_checkable() -> None:
    class _OkBenchmark:
        name = "ok"

        def tasks(
            self, *, split: str = "test", limit: int | None = None
        ) -> Sequence[BenchmarkTask]:
            return []

        async def run_task(
            self,
            task: BenchmarkTask,
            model: str,
            provider: str,
            *,
            store: Any,
            seed: int = 42,
        ) -> BenchmarkRunResult:
            return BenchmarkRunResult(trajectory_id="x", success=True, raw_score=1.0)

    assert isinstance(_OkBenchmark(), Benchmark)

    class _NotBenchmark:
        name = "no"

    assert not isinstance(_NotBenchmark(), Benchmark)
```

- [ ] **Step 3: Run test to confirm RED**

Run: `uv run pytest tests/unit/benchmarks/test_base.py -q`
Expected: FAIL — `ariadne_eval.benchmarks` package missing.

- [ ] **Step 4: Implement the package + base module**

Create `src/ariadne_eval/benchmarks/__init__.py`:

```python
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
```

Create `src/ariadne_eval/benchmarks/base.py`:

```python
"""Benchmark Protocol + result dataclasses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ariadne_eval.core.trajectory import Step, Trajectory  # noqa: F401  (re-exported via Protocol)
from ariadne_eval.storage.base import Store

__all__ = ["Benchmark", "BenchmarkRunResult", "BenchmarkTask"]


@dataclass(frozen=True)
class BenchmarkTask:
    """One task in a benchmark.

    ``payload`` is benchmark-specific and not interpreted by the
    runner; concrete adapters (e.g. ``TauBenchAdapter``) carry the
    benchmark's native task object through here so ``run_task`` can
    reconstitute env state per task.
    """

    task_id: str
    task_index: int
    instruction: str
    payload: Any


@dataclass(frozen=True)
class BenchmarkRunResult:
    """What the benchmark hands back per task.

    ``trajectory_id`` is a foreign key into the store; the full trace
    lives there. ``success`` is the benchmark's boolean verdict;
    ``raw_score`` is its native numeric score (tau-bench: ≈1.0 → pass).
    """

    trajectory_id: str
    success: bool
    raw_score: float
    error: str | None = None


@runtime_checkable
class Benchmark(Protocol):
    """Trajectory-agnostic benchmark contract."""

    name: str

    def tasks(
        self,
        *,
        split: str = "test",
        limit: int | None = None,
    ) -> Sequence[BenchmarkTask]:
        """Return the benchmark's task list. May read disk / network on first call."""
        ...

    async def run_task(
        self,
        task: BenchmarkTask,
        model: str,
        provider: str,
        *,
        store: Store,
        seed: int = 42,
    ) -> BenchmarkRunResult:
        """Run ``task`` against ``model`` and persist the trajectory to ``store``."""
        ...
```

- [ ] **Step 5: Run test to confirm GREEN**

```bash
uv run pytest tests/unit/benchmarks/test_base.py -q
```

Expected: 5 passed.

- [ ] **Step 6: Verify gates**

```bash
uv run mypy --strict src/ariadne_eval/benchmarks
uv run ruff check src/ariadne_eval/benchmarks tests/unit/benchmarks
uv run ruff format src/ariadne_eval/benchmarks tests/unit/benchmarks
uv run ruff format --check src/ariadne_eval/benchmarks tests/unit/benchmarks
```

Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add src/ariadne_eval/benchmarks/__init__.py src/ariadne_eval/benchmarks/base.py tests/unit/benchmarks/__init__.py tests/unit/benchmarks/test_base.py
git commit -m "feat(benchmarks): Benchmark Protocol + BenchmarkTask + BenchmarkRunResult"
```

---

## Task 3: `_convert_tau_traj` helper (no tau-bench dependency yet)

**Goal:** A pure function that converts tau-bench's flat message-list `EnvRunResult` shape into ariadne's `Trajectory` + parent/child `Step` tree. Tested with hand-crafted fixtures — no real tau-bench install needed.

**Files:**
- Create: `src/ariadne_eval/benchmarks/tau_bench.py` (just the helper this task)
- Create: `tests/unit/benchmarks/test_tau_bench_convert.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/benchmarks/test_tau_bench_convert.py`:

```python
"""Tests for the tau-bench → ariadne trajectory converter."""

from __future__ import annotations

import pytest

from ariadne_eval.benchmarks.tau_bench import _convert_tau_traj
from ariadne_eval.core.status import TrajectoryStatus
from ariadne_eval.core.trajectory import (
    LLMCallPayload,
    ToolCallPayload,
    UserInputPayload,
)

pytestmark = pytest.mark.fast


# A minimal EnvRunResult-shaped dict. We don't import tau_bench's types;
# the converter accepts a plain dict with the same keys.
def _make_env_result(*, reward: float, traj: list[dict[str, object]], task_id: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "reward": reward,
        "info": {},
        "traj": traj,
    }


def test_convert_zero_tool_calls_final_answer() -> None:
    """An assistant turn with no tool_calls becomes the final answer."""
    env_result = _make_env_result(
        task_id="task-1",
        reward=1.0,
        traj=[
            {"role": "user", "content": "Add 1 and 2."},
            {"role": "assistant", "content": "The answer is 3."},
        ],
    )
    traj, steps = _convert_tau_traj(
        env_result,
        instruction="Add 1 and 2.",
        model_id="anthropic/claude-haiku-4-5",
        agent_name="tau-bench/tool-calling",
        agent_version="59a200c",
    )
    assert traj.task == "Add 1 and 2."
    assert traj.final_answer == "The answer is 3."
    assert traj.final_status == TrajectoryStatus.SUCCEEDED  # reward >= 1.0
    assert traj.metadata["tau_bench_reward"] == 1.0
    assert traj.metadata["tau_bench_task_id"] == "task-1"
    # 1 user step + 1 llm step = 2 steps total
    assert len(steps) == 2
    assert isinstance(steps[0].payload, UserInputPayload)
    assert isinstance(steps[1].payload, LLMCallPayload)


def test_convert_multi_step_tool_calls() -> None:
    """Assistant with tool_calls → parent LLM Step + child ToolCall Steps."""
    env_result = _make_env_result(
        task_id="task-2",
        reward=1.0,
        traj=[
            {"role": "user", "content": "What is 17 * 23?"},
            {
                "role": "assistant",
                "content": "I'll use the calculator.",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "calculator", "arguments": '{"expression":"17*23"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "391"},
            {"role": "assistant", "content": "The answer is 391."},
        ],
    )
    traj, steps = _convert_tau_traj(
        env_result,
        instruction="What is 17 * 23?",
        model_id="anthropic/claude-haiku-4-5",
        agent_name="tau-bench/tool-calling",
        agent_version="59a200c",
    )
    # Steps: user, llm_call (parent) → tool_call (child), final llm_call = 4 steps
    assert len(steps) == 4
    user, llm_parent, tool_child, llm_final = steps
    assert isinstance(user.payload, UserInputPayload)
    assert isinstance(llm_parent.payload, LLMCallPayload)
    assert isinstance(tool_child.payload, ToolCallPayload)
    assert isinstance(llm_final.payload, LLMCallPayload)
    assert tool_child.parent_step_id == llm_parent.id
    assert tool_child.payload.tool_name == "calculator"
    assert tool_child.payload.arguments == {"expression": "17*23"}
    assert tool_child.payload.result == "391"
    assert traj.final_answer == "The answer is 391."


def test_convert_failure_records_status() -> None:
    """reward < 1.0 → final_status=FAILED, traj.metadata records reward."""
    env_result = _make_env_result(
        task_id="task-3",
        reward=0.0,
        traj=[
            {"role": "user", "content": "Place an order."},
            {"role": "assistant", "content": "I cannot help with that."},
        ],
    )
    traj, _ = _convert_tau_traj(
        env_result,
        instruction="Place an order.",
        model_id="anthropic/claude-haiku-4-5",
        agent_name="tau-bench/tool-calling",
        agent_version="59a200c",
    )
    assert traj.final_status == TrajectoryStatus.FAILED
    assert traj.metadata["tau_bench_reward"] == 0.0
```

- [ ] **Step 2: Run test to confirm RED**

```bash
uv run pytest tests/unit/benchmarks/test_tau_bench_convert.py -q
```

Expected: FAIL — `ariadne_eval.benchmarks.tau_bench._convert_tau_traj` not found.

- [ ] **Step 3: Implement `_convert_tau_traj` in `src/ariadne_eval/benchmarks/tau_bench.py`**

Create `src/ariadne_eval/benchmarks/tau_bench.py`:

```python
"""tau-bench adapter and trajectory converter.

The adapter class lazy-imports the ``tau_bench`` package; users who
don't install the ``[tau-bench]`` extra never trigger the import. The
converter below is pure-Python and does not depend on the tau_bench
package — it accepts the EnvRunResult shape as a dict so it can be
exercised from unit tests without the extra installed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import StepStatus, TrajectoryStatus
from ariadne_eval.core.trajectory import (
    JsonValue,
    LLMCallPayload,
    Message,
    Step,
    StepError,  # noqa: F401  (re-exported for typing parity)
    ToolCallPayload,
    Trajectory,
    UserInputPayload,
)

__all__ = ["_convert_tau_traj"]


_SUCCESS_THRESHOLD = 1.0 - 1e-6


def _convert_tau_traj(
    env_result: dict[str, object],
    *,
    instruction: str,
    model_id: str,
    agent_name: str,
    agent_version: str,
) -> tuple[Trajectory, list[Step]]:
    """Convert a tau-bench ``EnvRunResult``-shaped dict to ``(Trajectory, list[Step])``.

    Accepts a plain dict with keys ``task_id``, ``reward``, ``info``,
    ``traj`` (a list of message dicts). This lets the converter be
    tested without installing the ``tau_bench`` package.
    """
    traj_id = new_id()
    started = datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC)

    reward = float(cast(float, env_result["reward"]))
    task_id = str(env_result["task_id"])
    raw_messages = cast(list[dict[str, Any]], env_result["traj"])

    steps: list[Step] = []
    final_answer: str | None = None
    pending_tool_parents: dict[str, str] = {}  # tool_call_id → parent step.id
    pending_tool_payloads: dict[str, Step] = {}  # tool_call_id → the child Step (so we can fill result)
    msg_index = 0

    for raw_msg in raw_messages:
        role = raw_msg.get("role")
        content = raw_msg.get("content", "") or ""
        step_started = started + timedelta(seconds=msg_index)
        step_finished = step_started + timedelta(milliseconds=10)

        if role == "user":
            step = Step(
                id=new_id(),
                trajectory_id=traj_id,
                parent_step_id=None,
                name="user_input",
                started_at=step_started,
                finished_at=step_finished,
                status=StepStatus.SUCCEEDED,
                payload=UserInputPayload(message=str(content)),
            )
            steps.append(step)

        elif role == "assistant":
            tool_calls = raw_msg.get("tool_calls") or []
            llm_step = Step(
                id=new_id(),
                trajectory_id=traj_id,
                parent_step_id=None,
                name="llm_call",
                started_at=step_started,
                finished_at=step_finished,
                status=StepStatus.SUCCEEDED,
                payload=LLMCallPayload(
                    model_id=model_id,
                    prompt_messages=[Message(role="user", content=instruction)],
                    completion=str(content),
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=10.0,
                    cost_usd=0.0,
                ),
            )
            steps.append(llm_step)

            if not tool_calls:
                # Final-answer assistant turn (no tool calls)
                if content:
                    final_answer = str(content)
            else:
                # Emit one child ToolCall step per tool_call
                for tool_call in tool_calls:
                    fn = tool_call.get("function", {})
                    tool_name = str(fn.get("name", "unknown"))
                    arguments_raw = fn.get("arguments", "{}")
                    try:
                        arguments = json.loads(arguments_raw) if isinstance(arguments_raw, str) else dict(arguments_raw)
                    except (json.JSONDecodeError, TypeError):
                        arguments = {"_raw": str(arguments_raw)}
                    tool_step = Step(
                        id=new_id(),
                        trajectory_id=traj_id,
                        parent_step_id=llm_step.id,
                        name=f"tool_{tool_name}",
                        started_at=step_started + timedelta(milliseconds=1),
                        finished_at=step_finished,
                        status=StepStatus.SUCCEEDED,
                        payload=ToolCallPayload(
                            tool_name=tool_name,
                            arguments=cast("dict[str, JsonValue]", arguments),
                            result=None,
                            latency_ms=10.0,
                        ),
                    )
                    steps.append(tool_step)
                    pending_tool_parents[str(tool_call.get("id", ""))] = llm_step.id
                    pending_tool_payloads[str(tool_call.get("id", ""))] = tool_step

        elif role == "tool":
            tool_call_id = str(raw_msg.get("tool_call_id", ""))
            if tool_call_id in pending_tool_payloads:
                child_step = pending_tool_payloads.pop(tool_call_id)
                # Pydantic frozen → use model_copy(update=...) to set result
                new_payload = cast(ToolCallPayload, child_step.payload).model_copy(
                    update={"result": str(content)}
                )
                # Replace the step in `steps` with one carrying the updated payload
                idx = steps.index(child_step)
                steps[idx] = child_step.model_copy(update={"payload": new_payload})

        # All other roles (system, etc.) are ignored
        msg_index += 1

    final_status = (
        TrajectoryStatus.SUCCEEDED if reward >= _SUCCESS_THRESHOLD else TrajectoryStatus.FAILED
    )

    trajectory = Trajectory(
        id=traj_id,
        task=instruction,
        agent_name=agent_name,
        agent_version=agent_version,
        model_id=model_id,
        started_at=started,
        finished_at=started + timedelta(seconds=max(msg_index, 1)),
        final_status=final_status,
        final_answer=final_answer,
        metadata={
            "tau_bench_reward": reward,
            "tau_bench_task_id": task_id,
        },
    )

    return trajectory, steps
```

- [ ] **Step 4: Run test to confirm GREEN**

```bash
uv run pytest tests/unit/benchmarks/test_tau_bench_convert.py -q
```

Expected: 3 passed.

- [ ] **Step 5: Verify gates**

```bash
uv run mypy --strict src/ariadne_eval/benchmarks/tau_bench.py
uv run ruff check src/ariadne_eval/benchmarks/tau_bench.py tests/unit/benchmarks/test_tau_bench_convert.py
uv run ruff format src/ariadne_eval/benchmarks/tau_bench.py tests/unit/benchmarks/test_tau_bench_convert.py
uv run ruff format --check src/ariadne_eval/benchmarks/tau_bench.py tests/unit/benchmarks/test_tau_bench_convert.py
```

Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/ariadne_eval/benchmarks/tau_bench.py tests/unit/benchmarks/test_tau_bench_convert.py
git commit -m "feat(benchmarks): _convert_tau_traj — flat tau-bench messages → ariadne Trajectory tree"
```

---

## Task 4: `TauBenchAdapter` with lazy import + ImportError

**Goal:** Add `TauBenchAdapter` class to `src/ariadne_eval/benchmarks/tau_bench.py`. It lazy-imports `tau_bench` inside `tasks()` and `run_task()`, raising an actionable ImportError when the `[tau-bench]` extra is missing. Tests use `monkeypatch` to stub `tau_bench.envs.get_env` so the full plumbing can be exercised without the real package.

**Files:**
- Modify: `src/ariadne_eval/benchmarks/tau_bench.py` (add adapter class + lazy-import helper)
- Create: `tests/unit/benchmarks/test_tau_bench.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/benchmarks/test_tau_bench.py`:

```python
"""Tests for TauBenchAdapter (lazy import + stubbed tau_bench)."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from ariadne_eval.benchmarks.tau_bench import TauBenchAdapter

pytestmark = pytest.mark.fast


def test_actionable_import_error_when_tau_bench_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the [tau-bench] extra, tasks() raises a clear ImportError."""
    # Ensure tau_bench is NOT importable
    monkeypatch.setitem(sys.modules, "tau_bench", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "tau_bench.envs", None)  # type: ignore[arg-type]

    adapter = TauBenchAdapter(env_name="retail")
    with pytest.raises(ImportError, match=r"\[tau-bench\] extra"):
        adapter.tasks(limit=1)


def test_tasks_returns_benchmark_tasks_with_stubbed_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With tau_bench.envs.get_env stubbed, tasks() yields BenchmarkTask records."""
    # Build a fake tau_bench.envs module with a fake get_env
    fake_task_a = {"task_id": "retail-0", "instruction": "Cancel order 42."}
    fake_task_b = {"task_id": "retail-1", "instruction": "Look up SKU 1."}
    fake_env = MagicMock()
    fake_env.tasks = [fake_task_a, fake_task_b]

    fake_envs_mod = MagicMock()
    fake_envs_mod.get_env = MagicMock(return_value=fake_env)
    fake_tau_bench_mod = MagicMock()

    monkeypatch.setitem(sys.modules, "tau_bench", fake_tau_bench_mod)
    monkeypatch.setitem(sys.modules, "tau_bench.envs", fake_envs_mod)

    adapter = TauBenchAdapter(env_name="retail")
    tasks = list(adapter.tasks(limit=2))
    assert len(tasks) == 2
    assert tasks[0].task_id == "retail-0"
    assert tasks[0].task_index == 0
    assert tasks[0].instruction == "Cancel order 42."
    assert tasks[0].payload is fake_task_a


def test_name_property() -> None:
    assert TauBenchAdapter(env_name="retail").name == "tau-retail"
    assert TauBenchAdapter(env_name="airline").name == "tau-airline"
```

- [ ] **Step 2: Run test to confirm RED**

```bash
uv run pytest tests/unit/benchmarks/test_tau_bench.py -q
```

Expected: FAIL — `TauBenchAdapter` not defined.

- [ ] **Step 3: Implement `TauBenchAdapter` in `src/ariadne_eval/benchmarks/tau_bench.py`**

Add to the existing `src/ariadne_eval/benchmarks/tau_bench.py` (do NOT remove `_convert_tau_traj` from Task 3). After the imports, add:

```python
from collections.abc import Sequence
from typing import Literal

from ariadne_eval.benchmarks.base import BenchmarkRunResult, BenchmarkTask
from ariadne_eval.storage.base import Store

# Extend __all__
__all__ = ["TauBenchAdapter", "_convert_tau_traj"]
```

(Make sure the file's `__all__` now lists both.)

Add the adapter class at the bottom of the file:

```python
_EXTRA_MSG = (
    "tau_bench is not installed. Install ariadne-eval with the [tau-bench] "
    "extra: pip install 'ariadne-eval[tau-bench]'"
)


def _import_tau_bench_envs() -> Any:
    """Lazy import of tau_bench.envs with an actionable error message."""
    try:
        from tau_bench import envs as tau_envs  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(_EXTRA_MSG) from exc
    if tau_envs is None:
        raise ImportError(_EXTRA_MSG)
    return tau_envs


class TauBenchAdapter:
    """Concrete Benchmark over tau-bench's retail or airline domains.

    Lazy-imports tau_bench; the [tau-bench] extra must be installed for
    ``tasks()`` and ``run_task()`` to work.
    """

    name: str

    def __init__(
        self,
        env_name: Literal["retail", "airline"],
        *,
        user_model: str = "groq/llama-3.3-70b-versatile",
        user_strategy: str = "llm",
        agent_kind: Literal[
            "tool-calling", "react", "few-shot-tool-calling"
        ] = "tool-calling",
    ) -> None:
        """Configure the adapter.

        ``user_model`` is the LLM that drives the simulated user inside
        tau-bench (a tau-bench native concept; defaults to Groq Llama
        because tau-bench's simulated user is the largest line-item
        across all cells).
        """
        self._env_name = env_name
        self._user_model = user_model
        self._user_strategy = user_strategy
        self._agent_kind = agent_kind
        self.name = f"tau-{env_name}"

    def tasks(
        self,
        *,
        split: str = "test",
        limit: int | None = None,
    ) -> Sequence[BenchmarkTask]:
        """Return tau-bench's task list as ``BenchmarkTask`` records."""
        envs = _import_tau_bench_envs()
        env = envs.get_env(
            self._env_name,
            user_strategy=self._user_strategy,
            user_model=self._user_model,
            user_provider=self._user_model.split("/", 1)[0],
            task_split=split,
        )
        raw_tasks: list[Any] = list(env.tasks)
        if limit is not None:
            raw_tasks = raw_tasks[:limit]
        return [
            BenchmarkTask(
                task_id=str(raw.get("task_id", f"{self._env_name}-{i}"))
                if isinstance(raw, dict)
                else str(getattr(raw, "task_id", f"{self._env_name}-{i}")),
                task_index=i,
                instruction=str(raw.get("instruction", ""))
                if isinstance(raw, dict)
                else str(getattr(raw, "instruction", "")),
                payload=raw,
            )
            for i, raw in enumerate(raw_tasks)
        ]

    async def run_task(
        self,
        task: BenchmarkTask,
        model: str,
        provider: str,
        *,
        store: Store,
        seed: int = 42,
    ) -> BenchmarkRunResult:
        """Run ``task`` against ``model``, capture the trajectory, persist it."""
        envs = _import_tau_bench_envs()
        # Open an isolated env per task — tau-bench convention.
        env = envs.get_env(
            self._env_name,
            user_strategy=self._user_strategy,
            user_model=self._user_model,
            user_provider=self._user_model.split("/", 1)[0],
            task_split="test",
        )

        agent = _build_tau_bench_agent(
            kind=self._agent_kind,
            model=model,
            provider=provider,
            env=env,
        )

        # tau-bench's agent.solve() is synchronous; run it off the event loop.
        loop = asyncio.get_running_loop()
        env_result = await loop.run_in_executor(
            None, lambda: agent.solve(env=env, task_index=task.task_index)
        )

        # env_result may be a tau_bench EnvRunResult (frozen dataclass) or
        # similar. Convert to a plain dict for our converter.
        result_dict = (
            env_result if isinstance(env_result, dict) else _as_dict(env_result)
        )
        traj, steps = _convert_tau_traj(
            result_dict,
            instruction=task.instruction,
            model_id=f"{provider}/{model}",
            agent_name=f"tau-bench/{self._agent_kind}",
            agent_version=self._TAU_BENCH_COMMIT,
        )
        await store.save_trajectory(traj, steps)

        reward = float(result_dict.get("reward", 0.0))
        return BenchmarkRunResult(
            trajectory_id=traj.id,
            success=reward >= _SUCCESS_THRESHOLD,
            raw_score=reward,
            error=None,
        )

    # Pinned in pyproject.toml's [tau-bench] extra; recorded in trajectory.agent_version.
    _TAU_BENCH_COMMIT = "59a200c6d575d595120f1cb70fea53cef0632f6b"


def _build_tau_bench_agent(*, kind: str, model: str, provider: str, env: Any) -> Any:
    """Construct the tau-bench agent class corresponding to ``kind``."""
    from tau_bench import agents as tau_agents  # type: ignore[import-not-found]

    # tau-bench's agent factory dispatch happens by class name. The exact
    # class names live in tau_bench.agents; this mapping mirrors what
    # tau-bench's own run.py selects.
    if kind == "tool-calling":
        return tau_agents.tool_calling_agent.ToolCallingAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=model,
            provider=provider,
            temperature=0.0,
        )
    elif kind == "react":
        return tau_agents.chat_react_agent.ChatReActAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=model,
            provider=provider,
            temperature=0.0,
        )
    elif kind == "few-shot-tool-calling":
        return tau_agents.few_shot_tool_calling_agent.FewShotToolCallingAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=model,
            provider=provider,
            temperature=0.0,
        )
    else:
        raise ValueError(f"unknown agent_kind: {kind!r}")


def _as_dict(obj: Any) -> dict[str, Any]:
    """Coerce a tau-bench EnvRunResult (or similar) into a plain dict.

    tau-bench's types may be frozen dataclasses, Pydantic models, or
    namedtuples; this converter walks the common attributes our
    converter needs (``task_id``, ``reward``, ``info``, ``traj``).
    """
    if hasattr(obj, "model_dump"):
        return cast("dict[str, Any]", obj.model_dump())
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    if hasattr(obj, "_asdict"):
        return cast("dict[str, Any]", obj._asdict())
    raise TypeError(f"cannot coerce {type(obj).__name__} to dict")
```

Add this test to `tests/unit/benchmarks/test_tau_bench.py`:

```python
async def test_run_task_calls_agent_solve_and_persists_trajectory(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """run_task glues tau-bench's agent.solve to our store via _convert_tau_traj."""
    from ariadne_eval.benchmarks.tau_bench import TauBenchAdapter
    from ariadne_eval.storage.duckdb_store import DuckDBStore

    # Stub env with a task list
    fake_env = MagicMock()
    fake_env.tasks = [{"task_id": "retail-0", "instruction": "do thing"}]
    fake_env.tools_info = []
    fake_env.wiki = ""

    # Stub the agent class — solve() returns a dict-shaped EnvRunResult
    fake_env_result = {
        "task_id": "retail-0",
        "reward": 1.0,
        "info": {},
        "traj": [
            {"role": "user", "content": "do thing"},
            {"role": "assistant", "content": "done."},
        ],
    }
    fake_agent_instance = MagicMock()
    fake_agent_instance.solve = MagicMock(return_value=fake_env_result)

    fake_tool_calling_module = MagicMock()
    fake_tool_calling_module.ToolCallingAgent = MagicMock(return_value=fake_agent_instance)
    fake_agents_mod = MagicMock()
    fake_agents_mod.tool_calling_agent = fake_tool_calling_module

    fake_envs_mod = MagicMock()
    fake_envs_mod.get_env = MagicMock(return_value=fake_env)
    fake_tau_bench_mod = MagicMock()

    monkeypatch.setitem(sys.modules, "tau_bench", fake_tau_bench_mod)
    monkeypatch.setitem(sys.modules, "tau_bench.envs", fake_envs_mod)
    monkeypatch.setitem(sys.modules, "tau_bench.agents", fake_agents_mod)

    store = DuckDBStore(path=tmp_path / "test.duckdb")
    adapter = TauBenchAdapter(env_name="retail")
    tasks = adapter.tasks(limit=1)

    result = await adapter.run_task(
        tasks[0],
        model="haiku-stub",
        provider="anthropic",
        store=store,
        seed=42,
    )
    assert result.success is True
    assert result.raw_score == 1.0
    # Trajectory was persisted
    traj, steps = await store.get_trajectory(result.trajectory_id)
    assert traj.task == "do thing"
    assert traj.final_answer == "done."
    assert len(steps) >= 2  # user + assistant
    await store.close()
```

That gives `run_task` a real test path via stubs — the maintainer's one-shot run (Task 9) then exercises the production path against real tau-bench.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/benchmarks/test_tau_bench.py tests/unit/benchmarks/test_tau_bench_convert.py -q
```

Expected: 6 passed (3 from Task 3 + 3 new).

- [ ] **Step 5: Verify gates**

```bash
uv run mypy --strict src/ariadne_eval/benchmarks
uv run ruff check src/ariadne_eval/benchmarks tests/unit/benchmarks
uv run ruff format src/ariadne_eval/benchmarks tests/unit/benchmarks
uv run ruff format --check src/ariadne_eval/benchmarks tests/unit/benchmarks
```

Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/ariadne_eval/benchmarks/tau_bench.py tests/unit/benchmarks/test_tau_bench.py
git commit -m "feat(benchmarks): TauBenchAdapter with lazy import + actionable ImportError"
```

---

## Task 5: `BenchmarkConfig` + YAML loader

**Goal:** A Pydantic config that mirrors the canonical YAML shape. `load_benchmark_config(path)` reads the YAML, returns a frozen `BenchmarkConfig`. Bad YAML fails loudly with a Pydantic error message that names the offending field.

**Files:**
- Create: `src/ariadne_eval/benchmarks/config.py`
- Create: `tests/unit/benchmarks/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/benchmarks/test_config.py`:

```python
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
    with pytest.raises(Exception):  # ValidationError on frozen
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
```

- [ ] **Step 2: Run test to confirm RED**

```bash
uv run pytest tests/unit/benchmarks/test_config.py -q
```

Expected: FAIL — module missing.

- [ ] **Step 3: Implement `src/ariadne_eval/benchmarks/config.py`**

```python
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

    Raises ``ValueError`` (via Pydantic) if the YAML's shape doesn't
    match :class:`BenchmarkConfig`. The error message names the
    offending field path.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"benchmark config root must be a mapping, got {type(raw).__name__}")
    return BenchmarkConfig.model_validate(raw)
```

- [ ] **Step 4: Run test to confirm GREEN**

```bash
uv run pytest tests/unit/benchmarks/test_config.py -q
```

Expected: 6 passed.

- [ ] **Step 5: Verify gates**

```bash
uv run mypy --strict src/ariadne_eval/benchmarks/config.py
uv run ruff check src/ariadne_eval/benchmarks/config.py tests/unit/benchmarks/test_config.py
uv run ruff format src/ariadne_eval/benchmarks/config.py tests/unit/benchmarks/test_config.py
uv run ruff format --check src/ariadne_eval/benchmarks/config.py tests/unit/benchmarks/test_config.py
```

Expected: all clean.

- [ ] **Step 6: Update `src/ariadne_eval/benchmarks/__init__.py` to re-export the new types**

Replace the existing content with:

```python
"""Benchmark adapters and runners."""

from __future__ import annotations

from ariadne_eval.benchmarks.base import (
    Benchmark,
    BenchmarkRunResult,
    BenchmarkTask,
)
from ariadne_eval.benchmarks.config import (
    BenchmarkConfig,
    BootstrapSpec,
    JudgeSpec,
    ModelSpec,
    OutputSpec,
    TasksSpec,
    TauBenchSpec,
    load_benchmark_config,
)

__all__ = [
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkRunResult",
    "BenchmarkTask",
    "BootstrapSpec",
    "JudgeSpec",
    "ModelSpec",
    "OutputSpec",
    "TasksSpec",
    "TauBenchSpec",
    "load_benchmark_config",
]
```

- [ ] **Step 7: Commit**

```bash
git add src/ariadne_eval/benchmarks/config.py src/ariadne_eval/benchmarks/__init__.py tests/unit/benchmarks/test_config.py
git commit -m "feat(benchmarks): BenchmarkConfig + YAML loader with Pydantic validation"
```

---

## Task 6: `BenchmarkRunner` + `BenchmarkReport` + bundle writer

**Goal:** The orchestrator. Runs `(task × model)` cells under bounded concurrency, persists trajectories via the store, runs the existing eval `Runner.aevaluate` over the captured trio, aggregates per-model with bootstrap CIs, writes the result bundle (`config.yaml` + `trajectories.jsonl` + `summary.json`). The runner accepts any `Benchmark` implementation — the canonical `TauBenchAdapter` is the production target; a `StubBenchmark` is what the unit test uses.

**Files:**
- Create: `src/ariadne_eval/benchmarks/runner.py`
- Create: `tests/unit/benchmarks/test_runner.py`
- Modify: `src/ariadne_eval/benchmarks/__init__.py` (re-export `BenchmarkRunner` + `BenchmarkReport`)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/benchmarks/test_runner.py`:

```python
"""End-to-end tests for BenchmarkRunner using StubBenchmark + StubJudge."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ariadne_eval.benchmarks.base import (
    Benchmark,
    BenchmarkRunResult,
    BenchmarkTask,
)
from ariadne_eval.benchmarks.config import (
    BootstrapSpec,
    JudgeSpec,
    ModelSpec,
    OutputSpec,
    TasksSpec,
    TauBenchSpec,
)
from ariadne_eval.benchmarks.runner import BenchmarkReport, BenchmarkRunner
from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import StepStatus, TrajectoryStatus
from ariadne_eval.core.trajectory import (
    LLMCallPayload,
    Message,
    Step,
    Trajectory,
)
from ariadne_eval.eval.judges.base import JudgeVerdict
from ariadne_eval.eval.judges.stub import StubJudge
from ariadne_eval.eval.metrics.efficiency import StepEfficiency
from ariadne_eval.eval.metrics.plan_quality import PlanQuality
from ariadne_eval.storage.duckdb_store import DuckDBStore

pytestmark = pytest.mark.fast


class StubBenchmark:
    """Deterministic in-memory benchmark for unit tests."""

    name = "stub-bench"

    def __init__(self, *, n_tasks: int = 3) -> None:
        self._n = n_tasks

    def tasks(
        self, *, split: str = "test", limit: int | None = None
    ) -> Sequence[BenchmarkTask]:
        n = min(limit, self._n) if limit else self._n
        return [
            BenchmarkTask(
                task_id=f"stub-{i}",
                task_index=i,
                instruction=f"do stub task {i}",
                payload={"i": i},
            )
            for i in range(n)
        ]

    async def run_task(
        self,
        task: BenchmarkTask,
        model: str,
        provider: str,
        *,
        store,  # noqa: ANN001
        seed: int = 42,
    ) -> BenchmarkRunResult:
        # Build a minimal trajectory + step + persist
        started = datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC)
        traj_id = new_id()
        # Every other task "succeeds" for variance in pass-rate aggregation
        reward = 1.0 if task.task_index % 2 == 0 else 0.0
        traj = Trajectory(
            id=traj_id,
            task=task.instruction,
            agent_name=f"stub-agent/{model}",
            agent_version="0.0.0",
            model_id=f"{provider}/{model}",
            started_at=started,
            finished_at=started + timedelta(seconds=1),
            final_status=(
                TrajectoryStatus.SUCCEEDED if reward >= 0.999999 else TrajectoryStatus.FAILED
            ),
            final_answer="ok" if reward >= 0.999999 else "no",
            metadata={"tau_bench_reward": reward, "tau_bench_task_id": task.task_id},
        )
        step = Step(
            id=new_id(),
            trajectory_id=traj_id,
            parent_step_id=None,
            name="llm_call",
            started_at=started,
            finished_at=started + timedelta(milliseconds=10),
            status=StepStatus.SUCCEEDED,
            payload=LLMCallPayload(
                model_id=f"{provider}/{model}",
                prompt_messages=[Message(role="user", content=task.instruction)],
                completion="Step 1: stub. Step 2: stub.",
                input_tokens=0,
                output_tokens=0,
                latency_ms=10.0,
                cost_usd=0.0,
            ),
        )
        await store.save_trajectory(traj, [step])
        return BenchmarkRunResult(
            trajectory_id=traj_id,
            success=reward >= 0.999999,
            raw_score=reward,
        )


def _config(bundle_dir: Path, store_path: Path) -> "BenchmarkConfig":
    from ariadne_eval.benchmarks.config import BenchmarkConfig

    return BenchmarkConfig(
        benchmark=TauBenchSpec(kind="tau-bench", env="retail"),
        models=[
            ModelSpec(model="m-a", provider="prov"),
            ModelSpec(model="m-b", provider="prov"),
        ],
        tasks=TasksSpec(limit=3, seed=42),
        concurrency=2,
        bootstrap=BootstrapSpec(n_resamples=200, confidence=0.95),
        metrics=["step_efficiency", "plan_quality"],
        judge=JudgeSpec(model="judge/m", temperature=0.0),
        output=OutputSpec(bundle_dir=bundle_dir, store_path=store_path),
    )


async def test_runner_writes_bundle_with_calibration_note(tmp_path: Path) -> None:
    bench = StubBenchmark(n_tasks=3)
    bundle_dir = tmp_path / "bundle"
    store_path = tmp_path / "bench.duckdb"
    cfg = _config(bundle_dir, store_path)
    store = DuckDBStore(path=store_path)
    judge = StubJudge(JudgeVerdict(score=0.6, label="partial", rationale="meh"))

    runner = BenchmarkRunner(
        benchmark=bench,
        models=cfg.models,
        store=store,
        metrics=[StepEfficiency(), PlanQuality(judge)],
        seed=cfg.tasks.seed,
        concurrency=cfg.concurrency,
        n_resamples=cfg.bootstrap.n_resamples,
        confidence=cfg.bootstrap.confidence,
    )
    report = await runner.run(bench.tasks(limit=cfg.tasks.limit))
    runner.write_bundle(report, bundle_dir, config=cfg)
    await store.close()

    assert (bundle_dir / "config.yaml").exists()
    assert (bundle_dir / "trajectories.jsonl").exists()
    assert (bundle_dir / "summary.json").exists()

    summary = json.loads((bundle_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["_kind"] == "benchmark_summary"
    assert summary["task_count"] == 3
    assert len(summary["models"]) == 2
    for m in summary["models"]:
        # Pass rate of stub bench: tasks 0,2 succeed → 2/3 ≈ 0.667
        assert m["pass_rate"]["n"] == 3
        assert m["pass_rate"]["mean"] == pytest.approx(2 / 3, abs=1e-6)
        # PlanQuality's score from StubJudge is 0.6 on every trajectory
        assert m["metrics"]["plan_quality"]["mean"] == pytest.approx(0.6, abs=1e-6)
        # Calibration note travels with plan_quality
        assert "calibration_note" in m["metrics"]["plan_quality"]
        assert "κ" in m["metrics"]["plan_quality"]["calibration_note"]


async def test_runner_trajectories_jsonl_is_sorted(tmp_path: Path) -> None:
    bench = StubBenchmark(n_tasks=2)
    bundle_dir = tmp_path / "bundle"
    store_path = tmp_path / "bench.duckdb"
    cfg = _config(bundle_dir, store_path)
    store = DuckDBStore(path=store_path)
    judge = StubJudge(JudgeVerdict(score=0.5, label="partial", rationale="ok"))

    runner = BenchmarkRunner(
        benchmark=bench,
        models=cfg.models,
        store=store,
        metrics=[StepEfficiency(), PlanQuality(judge)],
        seed=42,
        concurrency=1,
        n_resamples=200,
        confidence=0.95,
    )
    report = await runner.run(bench.tasks(limit=2))
    runner.write_bundle(report, bundle_dir, config=cfg)
    await store.close()

    lines = (bundle_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    # 2 tasks × 2 models = 4 lines
    assert len(lines) == 4
    # Each line is a JSON object with trajectory + steps
    keys = [json.loads(line) for line in lines]
    sort_tuples = [(k["trajectory"]["task"], k["trajectory"]["model_id"]) for k in keys]
    assert sort_tuples == sorted(sort_tuples)
```

- [ ] **Step 2: Run test to confirm RED**

```bash
uv run pytest tests/unit/benchmarks/test_runner.py -q
```

Expected: FAIL — `BenchmarkRunner` not defined.

- [ ] **Step 3: Implement `src/ariadne_eval/benchmarks/runner.py`**

```python
"""Benchmark orchestrator: runs (task × model) cells, writes the result bundle."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ariadne_eval._transient import (
    MAX_TRANSIENT_RETRIES,
    TRANSIENT_BACKOFF_BASE,
    is_transient,
)
from ariadne_eval._version import __version__
from ariadne_eval.benchmarks.base import (
    Benchmark,
    BenchmarkRunResult,
    BenchmarkTask,
)
from ariadne_eval.benchmarks.config import ModelSpec
from ariadne_eval.core.trajectory import Step, Trajectory
from ariadne_eval.eval.case import Case
from ariadne_eval.eval.metrics.base import AsyncMetric, Metric
from ariadne_eval.eval.runner import EvalReport, Runner
from ariadne_eval.eval.stats.bootstrap import bootstrap_mean_ci
from ariadne_eval.storage.base import Store

if TYPE_CHECKING:
    from ariadne_eval.benchmarks.config import BenchmarkConfig

__all__ = ["BenchmarkRunner", "BenchmarkReport", "CellResult"]


# Calibration note that travels with every plan_quality aggregate.
# Comes from Phase 6.1's published kappa.
_PLAN_QUALITY_CALIBRATION_NOTE = (
    "judge κ = 0.32 (fair); see docs/concepts/calibration.md"
)


@dataclass(frozen=True)
class CellResult:
    """One (task, model) cell's outcome."""

    task: BenchmarkTask
    model: ModelSpec
    run_result: BenchmarkRunResult
    trajectory: Trajectory
    steps: list[Step]


@dataclass(frozen=True)
class BenchmarkReport:
    """Aggregated benchmark output, ready for bundle write."""

    benchmark_name: str
    task_count: int
    seed: int
    run_date: str
    per_model: dict[ModelSpec, EvalReport]
    cell_results: list[CellResult]


class BenchmarkRunner:
    """Orchestrates a benchmark run and writes a result bundle."""

    def __init__(
        self,
        benchmark: Benchmark,
        models: Sequence[ModelSpec],
        *,
        store: Store,
        metrics: Sequence[Metric | AsyncMetric] = (),
        seed: int = 42,
        concurrency: int = 4,
        n_resamples: int = 1000,
        confidence: float = 0.95,
    ) -> None:
        self._benchmark = benchmark
        self._models = list(models)
        self._store = store
        self._metrics = list(metrics)
        self._seed = seed
        self._concurrency = concurrency
        self._n_resamples = n_resamples
        self._confidence = confidence

    async def run(
        self,
        tasks: Sequence[BenchmarkTask],
        *,
        resume_from_store: bool = False,
    ) -> BenchmarkReport:
        """Run every (task × model) cell under bounded concurrency."""
        if resume_from_store:
            # Resume support is a Task 6 extension; the canonical run path
            # uses --resume on the CLI which calls back into this method.
            # The unit test does not exercise resume; the manual run does.
            pass

        sem = asyncio.Semaphore(self._concurrency)

        async def _one(task: BenchmarkTask, model: ModelSpec) -> CellResult:
            async with sem:
                last_transient: Exception | None = None
                for attempt in range(MAX_TRANSIENT_RETRIES):
                    try:
                        run_result = await self._benchmark.run_task(
                            task,
                            model.model,
                            model.provider,
                            store=self._store,
                            seed=self._seed,
                        )
                        break
                    except Exception as exc:
                        if not is_transient(exc):
                            raise
                        last_transient = exc
                        await asyncio.sleep(TRANSIENT_BACKOFF_BASE * (2**attempt))
                else:
                    # All retries exhausted
                    return CellResult(
                        task=task,
                        model=model,
                        run_result=BenchmarkRunResult(
                            trajectory_id="",
                            success=False,
                            raw_score=0.0,
                            error=f"transient: {last_transient}",
                        ),
                        trajectory=Trajectory(
                            id="00000000000000000000000000",
                            task=task.instruction,
                            agent_name="failed",
                            agent_version="0",
                            model_id=f"{model.provider}/{model.model}",
                            started_at=datetime.now(UTC),
                            final_status="failed",  # type: ignore[arg-type]
                        ),
                        steps=[],
                    )

            # Load back from store to catch silent serialization drift
            traj, steps = await self._store.get_trajectory(run_result.trajectory_id)
            return CellResult(
                task=task,
                model=model,
                run_result=run_result,
                trajectory=traj,
                steps=steps,
            )

        # Dispatch all cells eagerly under the semaphore
        cells = await asyncio.gather(
            *(_one(task, model) for task in tasks for model in self._models)
        )

        # Aggregate per model with the eval Runner
        per_model: dict[ModelSpec, EvalReport] = {}
        for model in self._models:
            model_cells = [c for c in cells if c.model == model]
            triples = [
                (
                    c.trajectory,
                    c.steps,
                    Case(case_id=c.task.task_id, task=c.task.instruction),
                )
                for c in model_cells
                if c.run_result.error is None
            ]
            runner = Runner(
                metrics=self._metrics,
                seed=self._seed,
                n_resamples=self._n_resamples,
                confidence=self._confidence,
                on_missing_reference="skip",
            )
            per_model[model] = await runner.aevaluate(triples)

        return BenchmarkReport(
            benchmark_name=self._benchmark.name,
            task_count=len(tasks),
            seed=self._seed,
            run_date=datetime.now(UTC).strftime("%Y-%m-%d"),
            per_model=per_model,
            cell_results=list(cells),
        )

    def write_bundle(
        self,
        report: BenchmarkReport,
        out_dir: Path,
        *,
        config: BenchmarkConfig,
    ) -> None:
        """Materialize the result bundle to ``out_dir``."""
        out_dir.mkdir(parents=True, exist_ok=True)

        # config.yaml (audit trail copy)
        cfg_yaml = yaml.safe_dump(
            json.loads(config.model_dump_json()),
            sort_keys=False,
            allow_unicode=True,
        )
        (out_dir / "config.yaml").write_text(cfg_yaml, encoding="utf-8")

        # trajectories.jsonl, sorted by (task_id, model_id)
        rows = []
        for cell in report.cell_results:
            if cell.run_result.error is not None:
                continue
            rows.append(
                {
                    "trajectory": json.loads(cell.trajectory.model_dump_json()),
                    "steps": [json.loads(s.model_dump_json()) for s in cell.steps],
                }
            )
        rows.sort(
            key=lambda r: (str(r["trajectory"]["task"]), str(r["trajectory"]["model_id"]))
        )
        with (out_dir / "trajectories.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=str))
                f.write("\n")

        # summary.json
        models_block: list[dict[str, object]] = []
        for model, eval_report in report.per_model.items():
            model_cells = [c for c in report.cell_results if c.model == model]
            ok = [c for c in model_cells if c.run_result.error is None]
            successes = [1.0 if c.run_result.success else 0.0 for c in ok]
            pr_ci = bootstrap_mean_ci(
                successes, seed=report.seed, n_resamples=1000, confidence=0.95
            )
            metrics_block: dict[str, dict[str, object]] = {}
            for name, ci in eval_report.aggregates.items():
                block: dict[str, object] = {
                    "mean": ci.mean,
                    "lo": ci.lo,
                    "hi": ci.hi,
                    "n": ci.n,
                }
                if name == "plan_quality":
                    block["calibration_note"] = _PLAN_QUALITY_CALIBRATION_NOTE
                metrics_block[name] = block

            median_steps = (
                sorted(len(c.steps) for c in ok)[len(ok) // 2] if ok else 0
            )

            models_block.append(
                {
                    "model": model.model,
                    "provider": model.provider,
                    "pass_rate": {
                        "mean": pr_ci.mean,
                        "lo": pr_ci.lo,
                        "hi": pr_ci.hi,
                        "n": pr_ci.n,
                        "method": "bootstrap-percentile",
                    },
                    "metrics": metrics_block,
                    "median_steps": median_steps,
                    "errored_cells": len(model_cells) - len(ok),
                }
            )

        summary = {
            "_kind": "benchmark_summary",
            "benchmark": report.benchmark_name,
            "task_count": report.task_count,
            "seed": report.seed,
            "run_date": report.run_date,
            "ariadne_version": __version__,
            "models": models_block,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
```

- [ ] **Step 4: Update `src/ariadne_eval/benchmarks/__init__.py` to re-export runner types**

Replace the existing `__init__.py` content with:

```python
"""Benchmark adapters and runners."""

from __future__ import annotations

from ariadne_eval.benchmarks.base import (
    Benchmark,
    BenchmarkRunResult,
    BenchmarkTask,
)
from ariadne_eval.benchmarks.config import (
    BenchmarkConfig,
    BootstrapSpec,
    JudgeSpec,
    ModelSpec,
    OutputSpec,
    TasksSpec,
    TauBenchSpec,
    load_benchmark_config,
)
from ariadne_eval.benchmarks.runner import BenchmarkReport, BenchmarkRunner, CellResult

__all__ = [
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkReport",
    "BenchmarkRunResult",
    "BenchmarkRunner",
    "BenchmarkTask",
    "BootstrapSpec",
    "CellResult",
    "JudgeSpec",
    "ModelSpec",
    "OutputSpec",
    "TasksSpec",
    "TauBenchSpec",
    "load_benchmark_config",
]
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/unit/benchmarks/ -q
```

Expected: 17 passed (5 base + 3 convert + 3 adapter + 6 config + 2 runner — actually count by `--collect-only` if uncertain). All green.

- [ ] **Step 6: Verify gates**

```bash
uv run mypy --strict src/ariadne_eval/benchmarks
uv run ruff check src/ariadne_eval/benchmarks tests/unit/benchmarks
uv run ruff format src/ariadne_eval/benchmarks tests/unit/benchmarks
uv run ruff format --check src/ariadne_eval/benchmarks tests/unit/benchmarks
```

Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add src/ariadne_eval/benchmarks/runner.py src/ariadne_eval/benchmarks/__init__.py tests/unit/benchmarks/test_runner.py
git commit -m "feat(benchmarks): BenchmarkRunner + BenchmarkReport + bundle writer"
```

---

## Task 7: `ariadne bench run` CLI subcommand

**Goal:** A Click subcommand that loads a config YAML, validates it, and (in production) runs the benchmark. The unit test covers the `--dry-run` path (validation only, no LLM calls).

**Files:**
- Create: `src/ariadne_eval/cli/bench.py`
- Modify: `src/ariadne_eval/cli/main.py` (register subcommand)
- Create: `tests/unit/cli/test_bench.py`

- [ ] **Step 1: Read the existing CLI structure**

Read `src/ariadne_eval/cli/main.py` to confirm the existing Click group's name and import surface.

- [ ] **Step 2: Write the failing test**

Create `tests/unit/cli/test_bench.py`:

```python
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
```

- [ ] **Step 3: Run test to confirm RED**

```bash
uv run pytest tests/unit/cli/test_bench.py -q
```

Expected: FAIL — `bench` subcommand not registered.

- [ ] **Step 4: Implement `src/ariadne_eval/cli/bench.py`**

```python
"""`ariadne bench` subcommand group."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from ariadne_eval.benchmarks.config import load_benchmark_config


@click.group("bench")
def bench() -> None:
    """Run and compare agent benchmarks."""


@bench.command("run")
@click.argument(
    "config_path", type=click.Path(path_type=Path, exists=True, dir_okay=False)
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate the config without making any LLM calls.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Override config.tasks.limit (run fewer tasks for partial reruns).",
)
@click.option(
    "--models",
    multiple=True,
    default=(),
    help=(
        "Subset of config.models to run (repeatable). "
        "Example: --models anthropic/claude-haiku-4-5-20251001"
    ),
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Skip (task, model) cells whose trajectory already exists in the store.",
)
def bench_run(
    config_path: Path,
    dry_run: bool,
    limit: int | None,
    models: tuple[str, ...],
    resume: bool,
) -> None:
    """Run a benchmark from a YAML config."""
    cfg = load_benchmark_config(config_path)
    if dry_run:
        click.echo(f"config valid: {cfg.benchmark.kind} / {cfg.benchmark.env}")
        click.echo(f"  models   : {len(cfg.models)}")
        click.echo(f"  tasks    : {cfg.tasks.limit or 'all'} (seed={cfg.tasks.seed})")
        click.echo(f"  metrics  : {', '.join(cfg.metrics)}")
        click.echo(f"  bundle   : {cfg.output.bundle_dir}")
        return

    # Production path: import lazily so --dry-run never loads tau-bench.
    from ariadne_eval.benchmarks.runner import BenchmarkRunner
    from ariadne_eval.benchmarks.tau_bench import TauBenchAdapter
    from ariadne_eval.eval.judges.trajectory_judge import TrajectoryJudge
    from ariadne_eval.eval.metrics.efficiency import StepEfficiency
    from ariadne_eval.eval.metrics.plan_quality import PlanQuality
    from ariadne_eval.storage.duckdb_store import DuckDBStore

    benchmark = TauBenchAdapter(
        env_name=cfg.benchmark.env,
        user_model=cfg.benchmark.user_model,
        user_strategy=cfg.benchmark.user_strategy,
        agent_kind=cfg.benchmark.agent_kind,
    )
    selected_models = (
        [m for m in cfg.models if m.model in set(models)] if models else list(cfg.models)
    )
    if not selected_models:
        raise click.UsageError("--models filter excluded every config.models entry")

    judge = TrajectoryJudge(model=cfg.judge.model, temperature=cfg.judge.temperature)
    metric_objs = []
    for name in cfg.metrics:
        if name == "step_efficiency":
            metric_objs.append(StepEfficiency())
        elif name == "plan_quality":
            metric_objs.append(PlanQuality(judge))

    store = DuckDBStore(path=cfg.output.store_path.expanduser())
    runner = BenchmarkRunner(
        benchmark=benchmark,
        models=selected_models,
        store=store,
        metrics=metric_objs,
        seed=cfg.tasks.seed,
        concurrency=cfg.concurrency,
        n_resamples=cfg.bootstrap.n_resamples,
        confidence=cfg.bootstrap.confidence,
    )

    async def _go() -> None:
        tasks = benchmark.tasks(
            split=cfg.benchmark.task_split, limit=limit or cfg.tasks.limit
        )
        report = await runner.run(tasks, resume_from_store=resume)
        runner.write_bundle(report, cfg.output.bundle_dir, config=cfg)
        await store.close()

    asyncio.run(_go())
    click.echo(f"bundle written: {cfg.output.bundle_dir}")
```

- [ ] **Step 5: Register the subcommand in `src/ariadne_eval/cli/main.py`**

Read `src/ariadne_eval/cli/main.py`. Inside the existing `cli` group definition (look for `cli.add_command(...)` lines or `@cli.command` decorators), add:

```python
from ariadne_eval.cli.bench import bench as _bench_cmd

cli.add_command(_bench_cmd)
```

If `main.py` has no `add_command` pattern, add this line at the bottom of the file (after the `cli` group is defined).

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/unit/cli/test_bench.py -q
```

Expected: 3 passed.

- [ ] **Step 7: Verify gates**

```bash
uv run mypy --strict src/ariadne_eval/cli/bench.py
uv run ruff check src/ariadne_eval/cli/bench.py tests/unit/cli/test_bench.py
uv run ruff format src/ariadne_eval/cli/bench.py tests/unit/cli/test_bench.py src/ariadne_eval/cli/main.py
uv run ruff format --check src/ariadne_eval/cli
```

Expected: all clean.

- [ ] **Step 8: Commit**

```bash
git add src/ariadne_eval/cli/bench.py src/ariadne_eval/cli/main.py tests/unit/cli/test_bench.py
git commit -m "feat(cli): ariadne bench run with --dry-run / --limit / --models / --resume"
```

---

## Task 8: `[tau-bench]` extra in `pyproject.toml` + canonical run config YAML

**Goal:** Pin tau-bench as an optional extra. Add the canonical `tau_retail_baseline.yaml` config. Add an integration-style test that loads the YAML through the Pydantic config.

**Files:**
- Modify: `pyproject.toml`
- Create: `configs/benchmarks/tau_retail_baseline.yaml`
- Modify: `tests/unit/benchmarks/test_config.py` (one new test loading the canonical config)

- [ ] **Step 1: Update `pyproject.toml`**

Read `pyproject.toml`. Find the `[project.optional-dependencies]` block. Add:

```toml
tau-bench = [
    "tau-bench @ git+https://github.com/sierra-research/tau-bench@59a200c6d575d595120f1cb70fea53cef0632f6b",
]
bench = ["ariadne-eval[tau-bench]"]
```

(The `bench` extra is a future-proofing umbrella; for v0.0.9-alpha it's equivalent to `tau-bench`.)

- [ ] **Step 2: Create `configs/benchmarks/tau_retail_baseline.yaml`**

```bash
mkdir -p configs/benchmarks
```

Create `configs/benchmarks/tau_retail_baseline.yaml`:

```yaml
# Phase 7 headline benchmark: tau-retail × 2 models × 50 tasks.
# Anthropic budget cap $5 → Sonnet is dropped from the agent lineup;
# the simulated user runs on Groq (unlimited).
#
# Estimated Anthropic spend:
#   Haiku agent on 50 tasks    : ~$2.00
#   Sonnet judge on 100 traj   : ~$1.00
#   Anthropic total            : ~$3.00
#
# Groq spend (free under maintainer's account):
#   Llama 3.3 70B agent on 50 tasks
#   Llama 3.3 70B simulated user across 100 cells
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
```

- [ ] **Step 3: Add a test that the canonical config validates**

Append to `tests/unit/benchmarks/test_config.py`:

```python
def test_canonical_tau_retail_baseline_loads() -> None:
    """The headline-run config must be valid by Pydantic."""
    repo = Path(__file__).resolve().parents[3]
    path = repo / "configs" / "benchmarks" / "tau_retail_baseline.yaml"
    cfg = load_benchmark_config(path)
    assert cfg.benchmark.env == "retail"
    assert len(cfg.models) == 2
    assert cfg.tasks.limit == 50
    assert cfg.tasks.seed == 42
    assert cfg.judge.model == "anthropic/claude-sonnet-4-6"
    assert cfg.benchmark.user_model == "groq/llama-3.3-70b-versatile"
```

- [ ] **Step 4: Verify the canonical config loads + gates clean**

```bash
uv run pytest tests/unit/benchmarks/test_config.py -q
uv run ruff format --check configs/benchmarks/tau_retail_baseline.yaml || true
```

Expected: 7 passed (6 prior + 1 new). Ruff doesn't lint YAML; the `|| true` is just a sanity skip.

- [ ] **Step 5: Sync the new extra**

```bash
uv sync --all-extras 2>&1 | tail -5
```

This installs tau-bench from GitHub. Expected: clean install. If tau-bench's git URL is unreachable, the sync warns; this is acceptable for now (the unit tests don't import it directly).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock configs/benchmarks/tau_retail_baseline.yaml tests/unit/benchmarks/test_config.py
git commit -m "feat(extras): pin tau-bench in [tau-bench] extra; add tau_retail_baseline.yaml"
```

---

## Task 9: Manual headline run (maintainer-only)

**Goal:** Execute the canonical run against real APIs. Commit the produced bundle. This task spends real money (~$3 Anthropic, Groq free) and is therefore not executable by subagents.

**This task cannot be performed by a subagent.** The controller should pause after Task 8 and hand off to the maintainer with the commands below.

**Files (manual outputs):**
- Create: `docs/benchmarks/v0.0.9-alpha-tau-retail-50/config.yaml`
- Create: `docs/benchmarks/v0.0.9-alpha-tau-retail-50/trajectories.jsonl`
- Create: `docs/benchmarks/v0.0.9-alpha-tau-retail-50/summary.json`

- [ ] **Step 1: Make the output directory**

```bash
mkdir -p docs/benchmarks/v0.0.9-alpha-tau-retail-50
```

- [ ] **Step 2: Verify .env has both keys**

```bash
grep -E "^(ANTHROPIC_API_KEY|GROQ_API_KEY)" .env | wc -l
```

Expected: 2.

- [ ] **Step 3: Sanity-check the dry-run**

```bash
uv run ariadne bench run configs/benchmarks/tau_retail_baseline.yaml --dry-run
```

Expected output (no LLM calls):
```
config valid: tau-bench / retail
  models   : 2
  tasks    : 50 (seed=42)
  metrics  : step_efficiency, plan_quality
  bundle   : docs/benchmarks/v0.0.9-alpha-tau-retail-50
```

- [ ] **Step 4: Execute the real run**

```bash
set -a && source .env && set +a
uv run ariadne bench run configs/benchmarks/tau_retail_baseline.yaml
```

Expected runtime: 20-40 minutes. Cost: ~$3 Anthropic, Groq free.
If the run fails partway through, restart with `--resume`:
```bash
uv run ariadne bench run configs/benchmarks/tau_retail_baseline.yaml --resume
```

- [ ] **Step 5: Inspect the bundle**

```bash
ls -la docs/benchmarks/v0.0.9-alpha-tau-retail-50/
head -2 docs/benchmarks/v0.0.9-alpha-tau-retail-50/summary.json
jq '.models[] | {model, pass_rate}' docs/benchmarks/v0.0.9-alpha-tau-retail-50/summary.json
```

Expected: three files (`config.yaml`, `trajectories.jsonl`, `summary.json`). `summary.json` has two models with `pass_rate.mean` ∈ [0, 1].

- [ ] **Step 6: Verify mkdocs builds**

```bash
uv run mkdocs build --strict 2>&1 | tail -5
```

(Defer the nav update to Task 10; for now mkdocs may INFO about the bundle dir not being in nav — that's fine.)

- [ ] **Step 7: Commit the bundle**

Read the actual pass rates from the summary. Then:

```bash
git add docs/benchmarks/v0.0.9-alpha-tau-retail-50/
git commit -m "feat(benchmarks): tau-retail headline run bundle (Haiku <X%>, Llama 3.3 70B <Y%>)"
```

(Substitute `X` and `Y` with the actual percentages from the summary.)

---

## Task 10: Docs page, mkdocs nav, README headline table

**Goal:** Render the methodology + headline numbers into `docs/concepts/benchmarks.md`. Update mkdocs nav. Update the README's headline-table section with the real numbers.

**Files:**
- Create: `docs/concepts/benchmarks.md`
- Modify: `mkdocs.yml`
- Modify: `README.md`

- [ ] **Step 1: Create `docs/concepts/benchmarks.md`**

Read the actual numbers from `docs/benchmarks/v0.0.9-alpha-tau-retail-50/summary.json` first. Then write the page. Use this template (substitute `<X>`, `<Y>`, etc. with the real numbers):

```markdown
# Benchmarks

ariadne-eval ships its first headline benchmark in v0.0.9-alpha:
**τ-retail (50 tasks) × 2 agent models**, traced end-to-end through
the project's own observability stack.

## Headline results

| Agent model | Pass rate (95% CI) | Median steps | Errored cells |
|---|---|---|---|
| anthropic/claude-haiku-4-5-20251001 | <X>% (<lo>, <hi>) | <n> | <e> |
| groq/llama-3.3-70b-versatile        | <Y>% (<lo>, <hi>) | <n> | <e> |

`n = 50` per row. Bootstrap CIs computed at 1000 resamples with `seed=42`.
Raw report: [`docs/benchmarks/v0.0.9-alpha-tau-retail-50/summary.json`](../benchmarks/v0.0.9-alpha-tau-retail-50/summary.json).

## Methodology

- Tasks: τ-bench retail domain, `test` split, first 50 tasks (`seed=42`).
- Agent: τ-bench's own `ToolCallingAgent`, traced via `ariadne_eval.benchmarks.tau_bench.TauBenchAdapter`.
- Simulated user: `groq/llama-3.3-70b-versatile` (τ-bench's user simulator; runs on Groq to stay inside the $5 Anthropic budget cap).
- Judge for `plan_quality`: `anthropic/claude-sonnet-4-6`, `temperature=0.0`. Calibrated at **κ = 0.32 (fair)** per [Calibration](./calibration.md).
- All LLM calls use `temperature=0.0`. Anthropic at temp=0 is empirically bit-exact deterministic on these prompts.
- τ-bench pinned at commit `59a200c6d575d595120f1cb70fea53cef0632f6b`.

## Limitations

- Single benchmark domain (retail). τ-airline and SWE-Bench Lite are deferred.
- Two agent models. Sonnet was dropped from the agent lineup to fit the budget; the within-Anthropic ladder comparison is deferred.
- `plan_quality` numbers carry the κ = 0.32 (fair) judge calibration. See [Calibration](./calibration.md) for the confusion matrix and per-label P/R; the judge over-flags `fail` and demotes `pass` to `partial`. Use `plan_quality` for relative comparison across models, not as an absolute score.

## Reproducing

```bash
# install with the [tau-bench] extra
pip install 'ariadne-eval[tau-bench]'

# bring your own keys
export ANTHROPIC_API_KEY=...
export GROQ_API_KEY=...

# canonical run
uv run ariadne bench run configs/benchmarks/tau_retail_baseline.yaml
```

## Raw bundle

The run is reproducible via the committed config + the pinned tau-bench
commit. The bundle at
[`docs/benchmarks/v0.0.9-alpha-tau-retail-50/`](../benchmarks/v0.0.9-alpha-tau-retail-50/)
contains:

- `config.yaml` — exact run configuration
- `trajectories.jsonl` — all 100 trajectories in the ariadne `Trajectory` schema, sorted by `(task_id, model)`
- `summary.json` — pass rates, per-metric aggregates, prompt-hash digests, ariadne version
```

- [ ] **Step 2: Add Benchmarks to mkdocs nav**

Read `mkdocs.yml`. Find the Concepts block. Insert `Benchmarks: concepts/benchmarks.md` alphabetically (between `Calibration` and `Judges`):

```yaml
  - Concepts:
      - concepts/index.md
      - Benchmarks: concepts/benchmarks.md
      - Calibration: concepts/calibration.md
      - Judges: concepts/judges.md
      - Metrics: concepts/metrics.md
      - Storage: concepts/storage.md
      - Tracing: concepts/tracing.md
      - Trajectory model: concepts/trajectory.md
```

- [ ] **Step 3: Update README headline table**

Read `README.md`. Find the existing phase-status table. After it (or in a suitable location), add a `## Headline benchmark` section with the same table from `benchmarks.md`:

```markdown
## Headline benchmark

| Agent model | τ-retail pass rate (95% CI) | Median steps |
|---|---|---|
| anthropic/claude-haiku-4-5-20251001 | <X>% (<lo>, <hi>) | <n> |
| groq/llama-3.3-70b-versatile        | <Y>% (<lo>, <hi>) | <n> |

`n = 50`. Bootstrap CIs at 1000 resamples, `seed=42`. Trace + score reproducible via
[`configs/benchmarks/tau_retail_baseline.yaml`](./configs/benchmarks/tau_retail_baseline.yaml).
See [Benchmarks](./docs/concepts/benchmarks.md) for methodology and the
κ = 0.32 (fair) judge calibration caveat.
```

Also update the "What's shipped" phase table — change the Phase 7 row to `v0.0.9-alpha — shipped` and add a one-line `Phase 8 — CLI polish — planned`.

- [ ] **Step 4: Build the docs**

```bash
uv run mkdocs build --strict 2>&1 | tail -5
```

Expected: clean. No orphan-page warnings.

- [ ] **Step 5: Commit**

```bash
git add docs/concepts/benchmarks.md mkdocs.yml README.md
git commit -m "docs: phase 7 benchmark page, nav, README headline table"
```

---

## Task 11: Version bump + CHANGELOG + final verification + merge + tag

**Goal:** Bump to `0.0.9-alpha`, run the full verification gate, write the CHANGELOG entry, push the branch, open the PR, merge, tag, update memory.

**Files:**
- Modify: `src/ariadne_eval/_version.py`
- Modify: `pyproject.toml`
- Modify: `tests/unit/test_smoke.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Bump version in three places**

Update `src/ariadne_eval/_version.py`:
```python
__version__: str = "0.0.9-alpha"
```

Update `pyproject.toml`:
```toml
version = "0.0.9-alpha"
```

Update smoke-test assertions:
```bash
sed -i '' 's/"0.0.8-alpha"/"0.0.9-alpha"/g' tests/unit/test_smoke.py
```

Run `uv sync --all-extras` to refresh the lock.

- [ ] **Step 2: Update CHANGELOG**

Read `CHANGELOG.md`. Find `## [Unreleased]`. Under `### Added`, prepend the Phase 7 block (substitute `<X>`/`<Y>` with real numbers from the bundle):

```markdown
- Phase 7 ships the tau-bench benchmark runner. New
  `ariadne_eval.benchmarks` package: `Benchmark` Protocol,
  `BenchmarkTask`, `BenchmarkRunResult`, `BenchmarkConfig` (YAML),
  `BenchmarkRunner`, `BenchmarkReport`. New optional extra
  `[tau-bench]` pins Sierra's τ-bench at commit
  `59a200c6d575d595120f1cb70fea53cef0632f6b`. New CLI subcommand
  `ariadne bench run`.
- `_transient` retry primitives extracted from
  `scripts/build_calibration_set.py` into
  `src/ariadne_eval/_transient.py`; both the calibration script and
  the new benchmark runner import from it.
- Headline benchmark: τ-retail × 2 agent models × 50 tasks. Haiku
  4.5 at <X>% (95% CI [<lo>, <hi>]); Groq Llama 3.3 70B at <Y>%
  (95% CI [<lo>, <hi>]). Bundle committed at
  `docs/benchmarks/v0.0.9-alpha-tau-retail-50/`; human-readable page
  at `docs/concepts/benchmarks.md`. `summary.json` carries the κ =
  0.32 (fair) judge-calibration caveat on every `plan_quality`
  aggregate.
- Version bumped to `0.0.9-alpha`.
```

- [ ] **Step 3: Run the full verification gate**

```bash
uv run pytest -m "fast and not integration" --cov=src/ariadne_eval --cov=scripts --cov-report=term-missing -q 2>&1 | tail -25
uv run pytest -m integration -q 2>&1 | tail -3
uv run mypy --strict src/ariadne_eval 2>&1 | tail -3
uv run ruff check src/ariadne_eval tests examples scripts 2>&1 | tail -3
uv run ruff format --check src/ariadne_eval tests examples scripts 2>&1 | tail -3
uv run mkdocs build --strict 2>&1 | tail -5
```

Expected:
- Fast suite green.
- Coverage on `src/ariadne_eval/benchmarks/` and `src/ariadne_eval/_transient.py` ≥ 90%.
- Integration tests green (existing cassettes).
- mypy clean.
- ruff clean.
- mkdocs strict clean.

If `src/ariadne_eval/benchmarks/tau_bench.py`'s `run_task` shows as uncovered: it's the production stub-raises path. Acceptable; the maintainer's run in Task 9 IS its integration test.

If coverage falls under 90% on any benchmarks file: add a targeted unit test for the missing branch. Do not invent coverage by adding vacuous assertions.

- [ ] **Step 4: Commit version + CHANGELOG**

```bash
git add src/ariadne_eval/_version.py pyproject.toml uv.lock tests/unit/test_smoke.py CHANGELOG.md
git commit -m "chore: bump version to 0.0.9-alpha + phase 7 changelog"
```

- [ ] **Step 5: Push the branch and open the PR**

```bash
git push -u origin phase-7-tau-bench
gh pr create --title "Phase 7: tau-bench benchmark runner (v0.0.9-alpha)" --body "$(cat <<'EOF'
## Summary

Implements Phase 7 per docs/superpowers/specs/2026-06-01-tau-bench-runner-design.md and docs/superpowers/plans/2026-06-01-tau-bench-runner.md.

- Wraps Sierra's τ-bench behind a tau-agnostic Benchmark Protocol.
- TauBenchAdapter lazy-imports tau_bench via the new [tau-bench] extra (pinned to commit 59a200c).
- BenchmarkRunner orchestrates (task × model) cells under bounded concurrency, persists trajectories to the DuckDB store, runs the existing eval.Runner over the captures, writes a reproducible result bundle.
- ariadne bench run CLI with --dry-run / --limit / --models / --resume.
- Headline run: τ-retail × 2 agent models × 50 tasks (Haiku <X%>, Llama 3.3 70B <Y%>). Bundle at docs/benchmarks/v0.0.9-alpha-tau-retail-50/.
- summary.json carries the κ = 0.32 (fair) judge-calibration caveat on every plan_quality aggregate.

## Verification

- [x] uv run pytest -m "fast and not integration" -q → green
- [x] uv run pytest -m integration -q → green
- [x] uv run mypy --strict src/ariadne_eval → clean
- [x] uv run ruff check / format --check → clean
- [x] uv run mkdocs build --strict → clean
- [x] Coverage ≥ 90% on src/ariadne_eval/benchmarks/ + src/ariadne_eval/_transient.py
EOF
)"
```

- [ ] **Step 6: Wait for CI green, then merge + tag (controller-confirmed step)**

This step is hard-to-reverse. Confirm with the user before executing.

```bash
gh pr checks  # wait until all 3 cells (py3.11/3.12/3.13) pass
gh pr merge --merge --delete-branch
git checkout main && git pull --ff-only
git tag -a v0.0.9-alpha -m "Phase 7: tau-bench benchmark runner (Haiku <X%>, Llama 3.3 70B <Y%>)"
git push origin v0.0.9-alpha
```

- [ ] **Step 7: Update phase-state memory**

Edit `/Users/rish/.claude/projects/-Users-rish-Desktop-AI-Projects-ariadne-eval/memory/current_phase_state.md`:

- Last completed: Phase 7, tagged `v0.0.9-alpha`.
- Add `v0.0.9-alpha — tau-bench benchmark runner` to the shipped list.
- Carry the real pass rates.
- Next phase: Phase 8 — CLI polish (per `Prompts.md`).

---

## Self-review

**Spec coverage:** Every spec scope item has a task —

- `_transient` extraction → Task 1
- Benchmark Protocol + dataclasses → Task 2
- `_convert_tau_traj` → Task 3
- `TauBenchAdapter` → Task 4
- `BenchmarkConfig` + YAML loader → Task 5
- `BenchmarkRunner` + bundle writer → Task 6
- `ariadne bench run` CLI → Task 7
- `[tau-bench]` extra + canonical config → Task 8
- Manual headline run → Task 9
- Docs page + nav + README → Task 10
- Version bump + CHANGELOG + verify + merge + tag → Task 11

**Type consistency:** `BenchmarkTask` / `BenchmarkRunResult` defined in Task 2, consumed by every subsequent task. `_convert_tau_traj` signature `(env_result, *, instruction, model_id, agent_name, agent_version) -> tuple[Trajectory, list[Step]]` defined in Task 3 and used implicitly in Task 9's wired-up `TauBenchAdapter.run_task` (which is the maintainer's responsibility to flesh out during the real run — the production wiring is a follow-up if `TauBenchAdapter.run_task` raising NotImplementedError isn't acceptable; see note below). `BenchmarkRunner.run(tasks, *, resume_from_store)` signature defined in Task 6, called identically by the CLI in Task 7.

**Placeholder scan:** No "TBD" / "TODO" placeholders. The `<X>` / `<Y>` substitutions in Tasks 9-11 ARE placeholders — they're intentional, to be filled with the real numbers from Task 9's bundle.

**Scope check:** 11 tasks. Tasks 1-8 + 10-11 are subagent-executable. Task 9 is maintainer-only (real API keys + real money).

**Notable cross-task dependency:** Task 4 includes the full
`TauBenchAdapter.run_task` production code (calling
`_build_tau_bench_agent`, the agent's `solve()`, `_convert_tau_traj`,
and `store.save_trajectory`). A monkeypatch-based unit test in Task 4
exercises that path without real tau-bench installed. Task 9's
maintainer run is the actual integration test for the production code
path against real tau-bench. The agent-class names used in
`_build_tau_bench_agent` (`tool_calling_agent.ToolCallingAgent` etc.)
are best-guess based on tau-bench's public structure; if the actual
class layout differs, expect the implementer to adjust in a small
follow-up commit before Task 9.
