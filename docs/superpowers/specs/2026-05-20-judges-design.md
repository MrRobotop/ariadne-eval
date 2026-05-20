# Phase 6 Design: Async Runner, LLM Judges, and Calibration Harness

**Status:** Approved (2026-05-20) · **Phase:** 6 · **Target version:** 0.0.7

## Goal

Ship the infrastructure for LLM-as-judge evaluation: an async evaluation
path, a `Judge` Protocol, one concrete LLM-backed judge
(`TrajectoryJudge`), one judge-backed metric (`PlanQuality`), the Cohen's
kappa statistic for judge–human agreement, and a calibration harness CLI
that produces a reproducible kappa report. Also: absorb the Phase 5.1
follow-up (`EvalReport.to_jsonl`/`from_jsonl` NaN ↔ null round-trip)
since Phase 6 already touches `Runner` for the async overload.

## Scope

In, in build order:

1. **Phase 5.1 absorbed.** `EvalReport.to_jsonl` writes RFC-8259-valid
   JSON: non-finite floats (`NaN`, `+Inf`, `-Inf`) serialize as `null`.
   `EvalReport.from_jsonl` re-hydrates `null` in float fields back to
   `NaN` for `BootstrapCI`. The Known-Issues note added in Phase 5.1
   is removed from `CHANGELOG.md` when the fix lands.
2. **Async metric infrastructure.** Extend `Metric` Protocol with an
   optional `ascore`. Add `AsyncMetric` Protocol (ascore-only). Add
   `Runner.aevaluate` with bounded concurrency (default 4). Sync
   `Runner.evaluate` stays sync-only and raises a clear `RuntimeError`
   if any metric is async-only.
3. **Judge Protocol + verdict + stub.** `Judge` Protocol (async-only,
   `judge(trajectory, steps, case|None) -> JudgeVerdict`). `JudgeVerdict`
   Pydantic. `StubJudge` for tests.
4. **`TrajectoryJudge` + prompts.** Calls `litellm.acompletion`; prompts
   live in `eval/judges/prompts.py` as constants; verdict parser exposed
   for testing.
5. **`PlanQuality` metric.** Async-only, takes any `Judge`, extracts
   "plan" as the first `LLMCallPayload` step's completion text.
6. **`cohens_kappa`.** Pure NumPy; returns a `KappaResult` Pydantic with
   `kappa`, `n`, Landis–Koch interpretation band, and the label set.
7. **Calibration harness CLI.** `scripts/build_calibration_set.py`:
   `click`-based, takes a DuckDB store and a JSONL gold-labels file,
   runs the judge, writes a JSONL report ending in a summary line with
   kappa.
8. **One end-to-end VCR-cassette integration test** for
   `TrajectoryJudge`, mirroring the Phase-4 cassette pattern.

Out (deferred):

- `StepwiseJudge` and the `Recovery` metric — Phase 7.
- A published ≥50-example gold calibration set and the kappa table in
  docs (Headline Deliverable #3) — Phase 6.1.
- Krippendorff's alpha — later, when n>2 raters become realistic.
- `ariadne eval` CLI surface for judges — Phase 8.

## Public API discipline

Phase 6 honors Hard Rule #5 ("never ship an LLM-as-judge without
calibration data") by keeping judge-backed symbols out of the top-level
`ariadne_eval.__all__` until calibration data ships in Phase 6.1.

Newly importable but **only via `ariadne_eval.eval` / `ariadne_eval.eval.judges`**
(NOT re-exported from top level):

- `Judge`, `JudgeVerdict`, `JudgeParseError`
- `TrajectoryJudge`, `StubJudge`
- `PlanQuality`

Newly re-exported from the top-level `ariadne_eval.__all__`:

- `AsyncMetric` (general infrastructure, no calibration concern)
- `cohens_kappa`, `KappaResult` (general statistics)

When Phase 6.1 ships the gold set + kappa table, the judge-backed
symbols move to top-level in one batch.

## Architecture

| File | Responsibility |
|---|---|
| `src/ariadne_eval/eval/runner.py` | Fix NaN ↔ null JSONL round-trip. Add `Runner.aevaluate` with bounded `asyncio.Semaphore` concurrency. Add the sync-runner error for async-only metrics. |
| `src/ariadne_eval/eval/metrics/base.py` | Extend `Metric` Protocol with optional `ascore`. Add `AsyncMetric` Protocol. |
| `src/ariadne_eval/eval/judges/__init__.py` | Re-export `Judge`, `JudgeVerdict`, `JudgeParseError`, `TrajectoryJudge`, `StubJudge`. |
| `src/ariadne_eval/eval/judges/base.py` | `Judge` Protocol (runtime_checkable, async-only) + `JudgeVerdict` + `JudgeParseError`. |
| `src/ariadne_eval/eval/judges/prompts.py` | `PLAN_QUALITY_SYSTEM`, `PLAN_QUALITY_USER_TEMPLATE`, `parse_plan_quality_verdict`. |
| `src/ariadne_eval/eval/judges/trajectory_judge.py` | `TrajectoryJudge(model, *, system_prompt, user_prompt_template, response_parser, client=None)`. |
| `src/ariadne_eval/eval/judges/stub.py` | `StubJudge(verdict_fn)` — test-only deterministic judge. |
| `src/ariadne_eval/eval/metrics/plan_quality.py` | `PlanQuality(judge)` — async-only metric. |
| `src/ariadne_eval/eval/stats/agreement.py` | `cohens_kappa(a, b, *, labels=None) -> KappaResult`; `KappaResult` Pydantic. |
| `src/ariadne_eval/eval/errors.py` | Append `KappaInsufficientDataWarning` (subclass `UserWarning`, mirrors `BootstrapInsufficientDataWarning`). |
| `src/ariadne_eval/eval/__init__.py` | Add re-exports for the new symbols listed above. |
| `src/ariadne_eval/__init__.py` | Add `AsyncMetric`, `cohens_kappa`, `KappaResult` only. |
| `scripts/build_calibration_set.py` | Click CLI; loads store + gold labels; runs judge; emits report. |
| `tests/integration/test_trajectory_judge.py` + cassette | One end-to-end recorded LLM call. |
| `docs/concepts/judges.md` | Judge architecture, verdict shape, calibration policy. |
| `docs/reference/judges.md` | mkdocstrings auto-ref. |
| `examples/04_plan_quality/main.py` + `README.md` | Async runner + StubJudge + PlanQuality walkthrough (no network). |

## Data shapes

```python
# eval/judges/base.py

class JudgeVerdict(BaseModel):
    model_config = {"frozen": True}
    score: float                                              # in [0, 1]
    label: Literal["pass", "partial", "fail"]
    rationale: str
    raw: dict[str, JsonValue] = Field(default_factory=dict)


class JudgeParseError(ValueError):
    """Raised when a judge's textual response cannot be parsed into a
    JudgeVerdict. Not caught by Runner — judge bugs fail loud."""


@runtime_checkable
class Judge(Protocol):
    name: str
    async def judge(
        self,
        trajectory: Trajectory,
        steps: list[Step],
        case: Case | None,
    ) -> JudgeVerdict: ...
```

```python
# eval/stats/agreement.py

class KappaResult(BaseModel):
    model_config = {"frozen": True}
    kappa: float
    n: int
    label_set: tuple[str, ...]
    interpretation: Literal[
        "poor", "slight", "fair", "moderate", "substantial", "almost_perfect"
    ]


def cohens_kappa(
    rater_a: Sequence[str],
    rater_b: Sequence[str],
    *,
    labels: tuple[str, ...] | None = None,
) -> KappaResult: ...
```

Edge cases:

- `len(rater_a) != len(rater_b)` → `ValueError`.
- `n == 0` → `KappaResult(kappa=NaN, n=0, ..., interpretation="poor")`
  + a `KappaInsufficientDataWarning` (new, in `eval/errors.py`,
  subclass `UserWarning`, mirrors `BootstrapInsufficientDataWarning`).
- Pure NumPy implementation. The standard formula:
  `(p_o - p_e) / (1 - p_e)`. If `p_e == 1` (single-label degenerate case)
  → kappa = `1.0` if all agree else `NaN` with warning. Documented.

Landis–Koch (1977) interpretation bands:

| Kappa | Interpretation |
|---|---|
| < 0.0 | poor |
| 0.0 – 0.2 | slight |
| 0.2 – 0.4 | fair |
| 0.4 – 0.6 | moderate |
| 0.6 – 0.8 | substantial |
| > 0.8 | almost_perfect |

(Half-open intervals on the right, closed on the left; pin in docs.)

## Async `Metric` / `AsyncMetric` / `Runner.aevaluate`

```python
class Metric(Protocol):
    name: str
    def score(self, trajectory, steps, case) -> MetricResult: ...

@runtime_checkable
class AsyncMetric(Protocol):
    name: str
    async def ascore(self, trajectory, steps, case) -> MetricResult: ...
```

`Metric` keeps the sync `score` requirement. A metric MAY also implement
`ascore` (then it's a `Metric & AsyncMetric` hybrid). `AsyncMetric` is
the protocol for async-only metrics that intentionally do not provide a
sync path.

`Runner.aevaluate`:

```python
async def aevaluate(
    self,
    items: Iterable[tuple[Trajectory, list[Step], Case]],
) -> EvalReport: ...
```

- Construction: `Runner(metrics, *, seed=0, n_resamples=1000, confidence=0.95, on_missing_reference="skip"|"error", concurrency=4)`.
- For each input triple, iterate metrics in order. For each metric:
  - If `isinstance(metric, AsyncMetric)` or `hasattr(metric, "ascore")`,
    schedule `metric.ascore(...)` under a shared
    `asyncio.Semaphore(concurrency)`.
  - Else call `metric.score(...)` inline (pure compute, no offload).
- All per-(item, metric) tasks for a given item are dispatched eagerly;
  results are collected in deterministic per-(item, metric) order
  regardless of completion order (the `EvalReport.results` ordering
  matches the sync runner's invariant).
- `MissingReferenceError` is caught per the existing `on_missing_reference`
  policy. Any other exception cancels in-flight tasks (via `TaskGroup`)
  and propagates.
- Aggregates: identical to the sync path — `bootstrap_mean_ci` per
  metric with the same `seed`/`n_resamples`/`confidence`.

`Runner.evaluate` (sync) raises `RuntimeError` if any element of
`self._metrics` is async-only (i.e. lacks a callable `.score` attribute).
The error message points at `aevaluate`.

## `PlanQuality`

```python
PlanQuality(judge: Judge, *, name: str = "plan_quality")
```

- Async-only: implements `ascore` only.
- Reference-free: does not require any `Case` field; `case.task` provides
  the goal to the judge prompt.
- "Plan" extraction: the **completion text of the first `LLMCallPayload`
  step in `started_at` order**. If no such step exists →
  `MetricResult(score=0.0, label="fail", details={"reason": "no_llm_step"})`
  (analogous to `FinalAnswerMatch`'s no-final-answer path).
- Calls `await judge.judge(trajectory, steps, case)`. Maps the verdict:
  `MetricResult(metric=name, case_id=case.case_id, trajectory_id=trajectory.id,
  score=verdict.score, label=verdict.label,
  details={"rationale": verdict.rationale, "raw": verdict.raw})`.
- `JudgeParseError` from the judge is **not** caught — it propagates and
  fails the Runner loudly (matches the project's "judge bugs are bugs"
  stance).

## `TrajectoryJudge`

```python
TrajectoryJudge(
    model: str,                                       # e.g. "claude-sonnet"
    *,
    system_prompt: str = PLAN_QUALITY_SYSTEM,
    user_prompt_template: str = PLAN_QUALITY_USER_TEMPLATE,
    response_parser: Callable[[str], JudgeVerdict] = parse_plan_quality_verdict,
    client: Callable[..., Awaitable[str]] | None = None,
    temperature: float = 0.0,
    name: str = "trajectory_judge",
)
```

- `client=None` → uses `litellm.acompletion` directly with the given
  `model`. The injected client lets tests pass a deterministic
  `async def stub(model, messages, temperature, ...) -> str` returning
  the assistant text.
- `judge(trajectory, steps, case)`:
  1. Render the user prompt: `user_prompt_template.format(task=trajectory.task, plan=_extract_plan(steps))`.
  2. Call the client with `[{"role":"system","content":system_prompt},
     {"role":"user","content":user_prompt}]`, `temperature=0.0`.
  3. Pass the completion text to `response_parser`. Parser raises
     `JudgeParseError` on malformed responses.
  4. Return the parsed `JudgeVerdict`.
- The TrajectoryJudge does not catch its own errors — they propagate.
- `_extract_plan(steps)` is a module-private helper shared with
  `PlanQuality` (single source of truth for "what's the plan").

### Prompt structure (`eval/judges/prompts.py`)

`PLAN_QUALITY_SYSTEM`: a short system prompt asking the model to act as
a strict but fair judge of agent plans, scoring on clarity (does it
state what it will do?), decomposition (is the task broken into
actionable steps?), and executability (does the plan correspond to
tools the agent can actually use?).

`PLAN_QUALITY_USER_TEMPLATE`: includes the `task` and `plan` and asks
for a response in this exact format:

```
SCORE: <0.0 to 1.0>
LABEL: <pass|partial|fail>
RATIONALE: <one to three sentences>
CLARITY: <1 to 5>
DECOMPOSITION: <1 to 5>
EXECUTABILITY: <1 to 5>
```

`parse_plan_quality_verdict(text) -> JudgeVerdict`:

- Tolerates leading/trailing whitespace and case-insensitive keys.
- Required keys: `SCORE`, `LABEL`, `RATIONALE`. Missing any → `JudgeParseError`.
- Optional keys: `CLARITY`, `DECOMPOSITION`, `EXECUTABILITY` go into `raw`.
- Validates `0.0 <= score <= 1.0` and `label in {pass, partial, fail}`;
  out-of-range → `JudgeParseError`.

### Concurrency, cancellation, determinism

- `Runner.aevaluate` uses `asyncio.Semaphore(concurrency)` to bound
  in-flight async metric calls (default 4).
- Cancellation: built on `asyncio.TaskGroup` (Python 3.11+, already the
  project minimum). An exception from any task cancels the rest cleanly.
- Determinism: `EvalReport.results` is appended in the deterministic
  per-(item, metric) order — async completions are awaited before
  appending to preserve order, so a slow metric on item 1 does not
  reorder against a fast metric on item 2.

## `scripts/build_calibration_set.py`

Click CLI. Single subcommand (the script itself).

```
uv run python scripts/build_calibration_set.py \
    --store ~/.ariadne/store.duckdb \
    --gold-labels tests/data/gold_plan_quality.jsonl \
    --judge-model claude-sonnet \
    --out calibration_report.jsonl \
    --concurrency 4
```

- Each gold-labels line: `{"trajectory_id": "01J...", "label": "pass|partial|fail"}`.
- For each line:
  - `await store.get_trajectory(trajectory_id)` → (Trajectory, list[Step]).
  - `verdict = await judge.judge(trajectory, steps, case=None)`.
  - Write `{"trajectory_id": ..., "gold_label": ..., "judge_label": verdict.label, "judge_score": verdict.score, "judge_rationale": verdict.rationale}`.
- After all lines: write a final summary line:
  `{"_kind": "summary", "n": ..., "kappa": ..., "interpretation": ...}`.
- Bounded concurrency via the same `asyncio.Semaphore` pattern as
  `aevaluate`.
- Errors per trajectory: log and continue (calibration runs are
  long-running; one parse failure shouldn't abort the whole set). The
  summary's `n` reflects only successfully judged trajectories. The
  per-line records for failed trajectories include
  `{"error": "<JudgeParseError message>"}` instead of judge fields.
- This is the ONE place in Phase 6 where errors are caught — and
  documented as such.

## Testing strategy

| Test | Type | File |
|---|---|---|
| `to_jsonl`/`from_jsonl` NaN round-trip | unit (`fast`) | `tests/unit/eval/test_runner.py` (extend) |
| `Runner.aevaluate` mix of sync + async metrics, order preserved | unit (`fast`, `pytest-asyncio`) | `tests/unit/eval/test_runner_async.py` (new) |
| `Runner.aevaluate` concurrency bound observed | unit (`fast`) | same file; `SlowStubJudge` records max in-flight count |
| `Runner.evaluate` with async-only metric raises `RuntimeError` | unit (`fast`) | extend `test_runner.py` |
| `JudgeVerdict` shape, immutability, label validation | unit (`fast`) | `tests/unit/eval/judges/test_base.py` |
| `parse_plan_quality_verdict` happy + malformed (missing key, out-of-range score, bad label) | unit (`fast`) | `tests/unit/eval/judges/test_prompts.py` |
| `StubJudge` deterministic | unit (`fast`) | `tests/unit/eval/judges/test_stub.py` |
| `PlanQuality` with `StubJudge`: pass / fail / no-llm-step / case=None | unit (`fast`, `pytest-asyncio`) | `tests/unit/eval/metrics/test_plan_quality.py` |
| `TrajectoryJudge` with injected stub client (no LLM, no cassette) — full plumbing including parser | unit (`fast`, `pytest-asyncio`) | `tests/unit/eval/judges/test_trajectory_judge.py` |
| `TrajectoryJudge` end-to-end via VCR cassette | integration (`integration`) | `tests/integration/test_trajectory_judge.py` + cassette |
| `cohens_kappa` known values (perfect / no agreement / chance) | unit (`fast`) | `tests/unit/eval/stats/test_agreement.py` |
| `cohens_kappa` edge cases: n=0 warning, length mismatch raises, single-label degenerate | unit (`fast`) | same file |
| `scripts/build_calibration_set.py` happy path with `StubJudge` injected via env or constructor + tmp DuckDB store | unit (`fast`, `pytest-asyncio`) | `tests/unit/scripts/test_build_calibration_set.py` (new) |

Coverage gate: ≥90% on touched `src/ariadne_eval/eval/**` files and on
`scripts/build_calibration_set.py`. Same as Phase 5.

## Documentation

- `docs/concepts/judges.md` — what judges are, the verdict shape, why
  judges live behind calibration, how to write your own. References the
  pending Phase 6.1 calibration table.
- `docs/reference/judges.md` — mkdocstrings for `eval/judges/**`.
- `docs/concepts/metrics.md` — append a "Judge-backed metrics" section
  pointing at `judges.md` and explaining that `PlanQuality` is
  reference-free.
- `docs/reference/eval.md` — add the new symbols (`AsyncMetric`,
  `agreement`).
- `examples/04_plan_quality/` — end-to-end async example using
  `StubJudge` (no API key required). README explains how to swap in
  `TrajectoryJudge` once the user provides one.
- `CHANGELOG.md` `[Unreleased]`:
  - `### Added`: the new symbols, async runner, calibration script,
    judges concept/reference pages, example 04.
  - `### Fixed`: the NaN-JSON round-trip.
  - Remove the Phase 5.1 Known-Issues entry (the issue is fixed).
- `mkdocs.yml`: add the two new doc pages to nav.

## Risks & non-obvious decisions

- **Why judge symbols are not top-level in Phase 6.** Hard Rule #5 says
  the project will never ship an LLM-as-judge without calibration data.
  The judge code itself is shippable now, but the *public commitment*
  encoded by top-level export waits for the Phase 6.1 gold set. Users
  who want to experiment early can still
  `from ariadne_eval.eval.judges import TrajectoryJudge`.
- **Why async-only judges and async-only `PlanQuality`.** Judges call
  LLMs; there's no plausible production use that wants a blocking sync
  judge. A sync wrapper would just be `asyncio.run` around the async
  call and would invite double-running event loops. We provide no sync
  wrapper; users who want sync call `asyncio.run(runner.aevaluate(...))`.
- **Why bounded concurrency = 4 by default.** Most LLM provider rate
  limits punish > 5–10 concurrent requests per key. 4 is conservative
  and configurable. Users with higher limits raise it.
- **Why `temperature=0.0` by default.** Reproducibility for kappa
  computation. Documented; users can pass higher temperatures for
  ensembling experiments later. Note: some providers ignore `seed` even
  at `temperature=0`; the docs flag this and recommend pinning the
  judge model across calibration and production.
- **Why a regex-tolerant text parser instead of structured outputs
  (JSON / function calling).** Provider-portable: works across
  Anthropic / OpenAI / open-weights via litellm without per-provider
  branching. Structured outputs are a later optimization once we know
  which providers are in scope.
- **Why catch errors in the calibration script but not in the metric.**
  Calibration runs over hundreds of trajectories and shouldn't abort
  on one parse failure. Metric evaluation runs over an individual
  trajectory where a parse failure means the metric is broken — fail
  loud. These two stances are documented in their respective files.
- **Why kappa as `KappaResult` Pydantic, not a bare float.** The
  interpretation band and n are load-bearing for the docs page and the
  calibration script's summary. A bare float forces every caller to
  re-derive these.

## Definition of Done (Phase 6)

Standard project DoD plus:

- [ ] NaN-JSON round-trip fix verified with a unit test.
- [ ] `Runner.aevaluate` covered by mixed-mode and concurrency-bound
      tests.
- [ ] `cohens_kappa` covered by known-value and edge-case tests.
- [ ] `TrajectoryJudge` exercised by the VCR cassette integration test.
- [ ] `scripts/build_calibration_set.py` happy-path covered.
- [ ] `docs/concepts/judges.md` and `docs/reference/judges.md` exist
      and `mkdocs build --strict` passes.
- [ ] CHANGELOG `[Unreleased]` updated; the Phase 5.1 Known-Issues
      entry removed.
- [ ] Phase-state memory updated: tagged `v0.0.7-alpha`, judges are
      namespace-public but top-level-private pending Phase 6.1.
