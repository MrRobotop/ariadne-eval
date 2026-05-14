# Phase 5 Design: Deterministic Metrics + Bootstrap CIs

**Status:** Approved (2026-05-14) · **Phase:** 5 · **Target version:** 0.0.6

## Goal

Ship the first usable evaluation loop: a small set of deterministic
trajectory metrics, a percentile-bootstrap confidence interval, and a
`Runner` that turns a stream of `(Trajectory, Case)` pairs into an
`EvalReport` containing per-trajectory results and aggregated 95% CIs.

This is the smallest scope that satisfies Hard Rule #4 ("never report a
metric without a confidence interval") and unlocks the rest of the eval
stack (judges in Phase 6, drift in Phase 7) without committing to either.

## Scope

In scope: `Case` sidecar model, `Metric` protocol, `MetricResult`,
three deterministic metrics (`FinalAnswerMatch`, `ToolAccuracy`,
`StepEfficiency`), `bootstrap_mean_ci`, `Runner` + `EvalReport`,
docs page, custom-metric example, public API re-exports.

Out of scope (deferred):
- `plan_quality`, `recovery` metrics — require LLM judges (Phase 6).
- Judge calibration, agreement statistics (Phase 6).
- Drift detection (CUSUM / ADWIN) — Phase 7.
- `ariadne eval` CLI surface — Phase 8, after a UI page consumes
  `EvalReport` directly.
- Storage of `EvalReport` in DuckDB — Phase 8 (JSONL round-trip is enough
  for v0.0.6).

## Architecture

| File | Responsibility |
|---|---|
| `src/ariadne_eval/eval/case.py` | `Case` and `ExpectedTool` Pydantic models. |
| `src/ariadne_eval/eval/errors.py` | `MissingReferenceError`, `BootstrapInsufficientDataWarning`. |
| `src/ariadne_eval/eval/metrics/base.py` | `Metric` Protocol + `MetricResult` Pydantic model. |
| `src/ariadne_eval/eval/metrics/final_answer.py` | `FinalAnswerMatch` (configurable comparator). |
| `src/ariadne_eval/eval/metrics/tool_accuracy.py` | `ToolAccuracy` (set / ordered_prefix). |
| `src/ariadne_eval/eval/metrics/efficiency.py` | `StepEfficiency`. |
| `src/ariadne_eval/eval/stats/bootstrap.py` | `bootstrap_mean_ci`, `BootstrapCI`. |
| `src/ariadne_eval/eval/runner.py` | `Runner`, `EvalReport`. |
| `src/ariadne_eval/eval/__init__.py` | Public re-exports for `ariadne_eval.eval`. |
| `src/ariadne_eval/__init__.py` | Add the new public symbols to the top-level surface. |
| `docs/concepts/metrics.md` | What each metric measures, the bootstrap explanation, the missing-reference policy. |
| `docs/reference/eval.md` | Auto API reference via mkdocstrings. |
| `examples/03_custom_metric/main.py` | Walkthrough: write a `Metric`, feed it through `Runner`, print the `EvalReport`. |
| `examples/03_custom_metric/README.md` | How to run it, what to expect. |

## Data model

```python
# eval/case.py

class ExpectedTool(BaseModel, frozen=True):
    name: str
    args: dict[str, JsonValue] | None = None  # None = ignore args, only check name


class Case(BaseModel, frozen=True):
    case_id: str                                  # ULID by default via new_id()
    task: str                                     # the prompt the agent was given
    expected_answer: str | None = None
    expected_tools: tuple[ExpectedTool, ...] = ()
    expected_max_steps: int | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
```

`Case` uses `tuple[...]` for `expected_tools` (frozen models cannot have
mutable defaults). The `ExpectedTool.args` field is the strict
"this exact arg dict" gate; `None` means "any args are acceptable, just
match by tool name".

```python
# eval/metrics/base.py

class MetricResult(BaseModel, frozen=True):
    metric: str                                   # e.g. "final_answer_match"
    case_id: str
    trajectory_id: str
    score: float                                  # in [0, 1] for the Phase-5 metrics
    label: Literal["pass", "fail", "partial"] | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class Metric(Protocol):
    name: str
    def score(self, trajectory: Trajectory, case: Case) -> MetricResult: ...
```

The `Metric` protocol is sync. Async metrics (judges) arrive in Phase 6
behind a separate `AsyncMetric` protocol; the `Runner` will gain an async
overload then. Phase 5 keeps everything sync — there is no IO.

## Metric semantics

### `FinalAnswerMatch`

```python
FinalAnswerMatch(
    comparator: Literal["normalized_exact", "exact"] | Callable[[str, str], float] = "normalized_exact",
    name: str = "final_answer_match",
)
```

- Reads `trajectory.final_output` and compares against `case.expected_answer`.
- `normalized_exact` (default): both sides are lowercased, leading/trailing
  whitespace stripped, internal whitespace runs collapsed to a single space.
  Score is `1.0` on equality, `0.0` otherwise. Label is `pass` / `fail`.
- `exact`: byte-for-byte equality. Same scoring as above.
- Custom callable: returns a float in `[0, 1]`. Label is `pass` if
  `score >= 0.99`, `fail` if `score <= 0.01`, else `partial`.
- If `case.expected_answer is None` → `MissingReferenceError`.
- If `trajectory.final_output is None` → `score=0.0`, `label="fail"`,
  `details={"reason": "no_final_output"}`.

### `ToolAccuracy`

```python
ToolAccuracy(
    mode: Literal["set", "ordered_prefix"] = "set",
    match_args: bool = False,
    name: str = "tool_accuracy",
)
```

- Walks the trajectory and collects every step whose payload is a
  `ToolCallPayload`, in order.
- `set` mode: treats expected and actual as multisets of `name` (or
  `(name, args)` if `match_args=True`, with `args` compared as JSON-canonical
  dicts). If `match_args=True` but a particular `ExpectedTool.args is None`,
  that expected tool matches *any* args for an actual call of the same name
  (per-tool wildcard). Score is F1. Details:
  `{precision, recall, matched, missing, extra}`.
- `ordered_prefix` mode: `len(longest_common_prefix) / len(expected)`.
  Useful when the agent must call tools in a specific order. Details:
  `{prefix_length, expected_length, first_divergence_index}`.
- `expected_tools == ()` and there are no actual tool calls → `score=1.0`,
  `label="pass"`. `expected_tools == ()` with extras → in `set` mode score
  is `0.0` with `label="fail"` (the case explicitly said "no tools");
  in `ordered_prefix` mode score is `1.0` (an empty prefix is always
  matched) — documented quirk, called out in the docstring.
- Label thresholds: `pass` if `score >= 0.99`, `fail` if `score <= 0.01`,
  else `partial`.

### `StepEfficiency`

```python
StepEfficiency(name: str = "step_efficiency")
```

- `actual_steps = len(trajectory.steps)`.
- `score = min(1.0, case.expected_max_steps / max(actual_steps, 1))`.
- If `case.expected_max_steps is None` → `MissingReferenceError`.
- Label is `pass` if `actual_steps <= expected_max_steps`, else `partial`
  (never `fail` — going over budget is a smell, not a correctness failure).
- Details: `{actual_steps, expected_max_steps}`.

## Bootstrap

```python
# eval/stats/bootstrap.py

class BootstrapCI(BaseModel, frozen=True):
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
) -> BootstrapCI: ...
```

- Standard percentile bootstrap: resample `n_resamples` times with
  replacement, take the empirical `α/2` and `1 - α/2` percentiles of the
  resampled means.
- Pure NumPy. Seeded via `np.random.default_rng(seed)`. No SciPy
  dependency added in this phase — keeps install lean. We can switch to
  `scipy.stats.bootstrap` (BCa) later without changing the function
  signature.
- Edge cases:
  - `n == 0`: `BootstrapCI(mean=NaN, lo=NaN, hi=NaN, n=0, ...)`, plus a
    `BootstrapInsufficientDataWarning`.
  - `n == 1`: `BootstrapCI(mean=values[0], lo=values[0], hi=values[0], n=1, ...)`,
    plus a `BootstrapInsufficientDataWarning`.
- `confidence` must be in `(0, 1)`; otherwise `ValueError`.
- `n_resamples` must be `>= 1`; otherwise `ValueError`.

## Runner & EvalReport

```python
# eval/runner.py

class EvalReport(BaseModel, frozen=True):
    results: tuple[MetricResult, ...]                # per (case, metric)
    aggregates: dict[str, BootstrapCI]               # keyed by Metric.name
    n_cases: int
    seed: int

    def to_jsonl(self, path: str | Path) -> None: ...
    @classmethod
    def from_jsonl(cls, path: str | Path) -> "EvalReport": ...


class Runner:
    def __init__(
        self,
        metrics: Sequence[Metric],
        *,
        seed: int = 0,
        n_resamples: int = 1000,
        confidence: float = 0.95,
        on_missing_reference: Literal["skip", "error"] = "skip",
    ) -> None: ...

    def evaluate(self, pairs: Iterable[tuple[Trajectory, Case]]) -> EvalReport: ...
```

- Sync. Pure compute. The Runner does not touch the `Store`; callers pass
  in already-loaded trajectories. (A `from_store(store, case_index)`
  convenience can land in Phase 8 once the UI needs it.)
- `on_missing_reference="skip"` (default): if a metric raises
  `MissingReferenceError` for a given `(traj, case)`, that single
  `(metric, case)` pair is silently omitted from `results`. The metric's
  aggregate is still computed over the remaining cases. The skipped count
  is exposed via `EvalReport.aggregates[metric_name].n` < `n_cases`.
- `on_missing_reference="error"`: re-raises `MissingReferenceError`.
- Other exceptions inside a metric propagate. Metrics are expected to be
  pure compute; a bug in a metric is a bug, not a graceful skip.
- `seed` is fed into `bootstrap_mean_ci` for every aggregate. Two `Runner`
  instances with the same `seed` and the same input produce identical
  reports.
- `EvalReport.to_jsonl` writes one JSON object per line: a header line
  with `{"_kind": "header", "n_cases", "seed", "aggregates": {...}}`
  followed by one line per `MetricResult`. `from_jsonl` is the inverse.
  Round-trip parity is a unit test.

## Public API additions

Re-exported from `ariadne_eval`:

- `Case`, `ExpectedTool`
- `Metric`, `MetricResult`
- `FinalAnswerMatch`, `ToolAccuracy`, `StepEfficiency`
- `Runner`, `EvalReport`
- `bootstrap_mean_ci`, `BootstrapCI`
- `MissingReferenceError`, `BootstrapInsufficientDataWarning`

Also re-exported from `ariadne_eval.eval` for users who prefer the
namespaced import.

## Testing strategy

| Test | Type | File |
|---|---|---|
| `Case` / `ExpectedTool` validation, immutability | unit (`fast`) | `tests/unit/eval/test_case.py` |
| `MetricResult` validation | unit (`fast`) | `tests/unit/eval/metrics/test_base.py` |
| `FinalAnswerMatch`: pass / fail / normalized / custom callable / missing reference / no `final_output` | unit (`fast`) | `tests/unit/eval/metrics/test_final_answer.py` |
| `ToolAccuracy`: set mode F1, ordered_prefix, `match_args` true vs false, empty-expected quirks (both modes) | unit (`fast`) | `tests/unit/eval/metrics/test_tool_accuracy.py` |
| `StepEfficiency`: under / over budget, missing reference | unit (`fast`) | `tests/unit/eval/metrics/test_efficiency.py` |
| `bootstrap_mean_ci`: reproducibility under seed, `n=0`, `n=1`, invalid args | unit (`fast`) | `tests/unit/eval/stats/test_bootstrap.py` |
| `bootstrap_mean_ci` coverage property: for `Uniform[0,1]` samples of size 50, the true mean (`0.5`) is inside the 95% CI on `≥ 90 / 100` seeds | property (`hypothesis`, `fast`) | same file |
| `Runner.evaluate`: 5 hand-built `(traj, case)` pairs through 3 metrics → `EvalReport` with the right per-(case,metric) results, aggregates, and `n_cases` | unit (`fast`) | `tests/unit/eval/test_runner.py` |
| `Runner` `on_missing_reference` skip vs error | unit (`fast`) | same file |
| `EvalReport.to_jsonl` / `from_jsonl` round-trip | unit (`fast`) | same file |

Coverage gate: ≥90% line coverage on `src/ariadne_eval/eval/` per the
project DoD.

## Performance

The Phase 5 code is pure compute on small data. There is no overhead
benchmark in this phase — the existing `benchmarks/overhead.py` covers
the `@trace_step` path, which is unchanged.

## Documentation

- `docs/concepts/metrics.md` — what each metric measures, the
  missing-reference policy, the bootstrap explanation, when CIs lie.
- `docs/reference/eval.md` — auto API reference via mkdocstrings.
- `examples/03_custom_metric/` — minimal walkthrough writing a `Metric`
  and running it through `Runner`.
- `CHANGELOG.md` `[Unreleased]` entry describing the new public surface.
- `mkdocs.yml` nav additions.

## Risks & non-obvious decisions

- **Why no DuckDB persistence for `EvalReport`?** Storing eval results
  alongside trajectories is genuinely useful but the right schema depends
  on what the metrics-page UI wants to query. JSONL round-trip is enough
  to unblock everything else; DuckDB persistence lands when the UI needs
  it (Phase 8).
- **Why percentile bootstrap and not BCa?** Percentile is correct enough
  for the `[0, 1]` scores these metrics produce, has zero external deps,
  and is trivial to swap later (BCa is `scipy.stats.bootstrap` with
  `method="BCa"`). Keeping the install lean matters more than CI bias for
  symmetric distributions on bounded scores.
- **Why pure NumPy (no SciPy yet)?** Same reason. SciPy enters in Phase 6
  (judges) or Phase 7 (drift) where it earns its keep.
- **Why is `Runner` sync?** Every Phase-5 metric is pure compute. Adding
  async now is premature; a separate `AsyncMetric` protocol and `Runner`
  async overload land naturally in Phase 6 when judges need IO.
- **Empty-expected `ToolAccuracy` quirk in `ordered_prefix` mode is
  documented, not fixed.** Empty prefix matches everything by definition;
  users who want "no tools allowed" should use `mode="set"`. Fixing it
  silently would be more surprising than the documented quirk.
