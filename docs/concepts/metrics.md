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
