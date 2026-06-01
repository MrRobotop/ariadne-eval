# Judges

A *judge* in `ariadne-eval` is an asynchronous function from a
`(Trajectory, list[Step], Case | None)` triple to a `JudgeVerdict` — a
float `score`, a categorical `label` (`pass`/`partial`/`fail`), a short
`rationale`, and optional structured `raw` sub-scores. Judges are how
LLM-as-judge evaluation enters the pipeline.

## Why judges live behind calibration

Per Hard Rule #5 in `CLAUDE.md`: the project does not ship LLM-as-judge
symbols on the top-level public API until calibration data exists.
Phase 6.1 ships the calibration evidence: see [Calibration](./calibration.md) for the
κ = 0.32 (fair) maintainer-vs-judge agreement number, the 3×3 confusion matrix, and
per-label precision/recall against the 51-fixture synthetic gold set. As of v0.0.8-alpha,
`Judge`, `JudgeVerdict`, `JudgeParseError`, `TrajectoryJudge`, `StubJudge`, and
`PlanQuality` are now top-level public (`from ariadne_eval import TrajectoryJudge`).

## The verdict shape

```python
class JudgeVerdict(BaseModel, frozen=True):
    score: float                                       # in [0, 1]
    label: Literal["pass", "partial", "fail"]
    rationale: str
    raw: dict[str, JsonValue] = {}                     # e.g. {"clarity": 4}
```

`score` becomes `MetricResult.score`. `label` is what Cohen's kappa is
computed against — kappa requires categorical labels. `rationale` is
what the UI surfaces for debugging judge disagreements. `raw` carries
optional structured fields the judge produced (per-dimension rubric
scores).

## `TrajectoryJudge`

`TrajectoryJudge(model, *, system_prompt, user_prompt_template, response_parser, client, temperature)`
calls `litellm.acompletion` by default. The injected `client` lets tests
supply a deterministic stub. `temperature=0.0` by default for kappa
reproducibility; some providers ignore `seed` even at `temperature=0`,
so pin the same judge model across your calibration and production runs.

## `PlanQuality`

`PlanQuality(judge)` is async-only and reference-free: it does not
require any field on `Case`. The "plan" is the completion text of the
**first `LLMCallPayload` step in `started_at` order**. If no LLM step
exists, the metric returns `score=0.0` / `label="fail"` /
`details={"reason": "no_llm_step"}` without calling the judge.

## Cohen's kappa

`cohens_kappa(rater_a, rater_b, *, labels=None) -> KappaResult` produces
a kappa value plus a Landis-Koch interpretation band (`poor`, `slight`,
`fair`, `moderate`, `substantial`, `almost_perfect`). `n == 0` emits a
`KappaInsufficientDataWarning` and returns NaN.

## The calibration harness

`scripts/build_calibration_set.py` is a click-based CLI that:

1. Loads each `{trajectory_id, label}` from a gold-labels JSONL file.
2. Looks up the trajectory + steps in a DuckDB store.
3. Runs the judge under bounded concurrency.
4. Writes one line per judged trajectory plus a `_kind: "summary"` line
   with `n`, `kappa`, and `interpretation`.

This is the one place in the eval stack where judge errors are caught
per-trajectory rather than failing loud — calibration runs are long and
should not abort on a single parse failure.

## Writing your own judge

A judge is anything that satisfies the `Judge` Protocol — a `name`
attribute and an `async def judge(trajectory, steps, case) -> JudgeVerdict`.
You can compose multiple judges into an ensemble by writing a wrapper
Judge whose `judge` method awaits several and aggregates their verdicts.
