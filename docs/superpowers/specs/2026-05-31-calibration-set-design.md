# Phase 6.1 Design: Calibration Set, Kappa Table, and Judge Promotion

**Status:** Approved (2026-05-31) · **Phase:** 6.1 · **Target version:** 0.0.8-alpha

## Goal

Ship the calibration evidence required by Hard Rule #5 ("never ship an
LLM-as-judge without calibration data"), then promote the judge symbols
from namespace-private (`ariadne_eval.eval.judges.*`) to top-level
(`ariadne_eval.*`). Concretely: a 51-trajectory hand-crafted synthetic
gold set, a one-shot run of `TrajectoryJudge` against a pinned
Claude Sonnet snapshot, a committed JSONL report, a human-readable
`docs/concepts/calibration.md` with kappa + confusion matrix +
per-label precision/recall, and a `__all__` change that exposes
`Judge`, `JudgeParseError`, `JudgeVerdict`, `PlanQuality`, `StubJudge`,
`TrajectoryJudge` at the top level.

## Scope

In, in build order:

1. **Synthetic gold set.** `tests/data/gold_plans.jsonl`: 51 lines, each
   carrying a full `(Trajectory, list[Step], gold_label)` triple. 17
   `pass`, 17 `partial`, 17 `fail`. Task diversity spans arithmetic /
   lookup / multi-hop reasoning / off-task to prevent the gold set from
   collapsing into one prompt pattern.
2. **`build_calibration_set.py` extension.** Add a `--source synth|store`
   flag. `--source synth` loads the JSONL fixtures directly (no DuckDB).
   `--source store` is unchanged (still the production path against a
   user's DuckDB store). Default: `store` (preserves existing behavior).
3. **Extended report shape.** The script's output JSONL now ends with
   three trailing lines instead of one: the existing `_kind: "summary"`
   plus new `_kind: "confusion"` and `_kind: "meta"` lines.
4. **`scripts/render_calibration_md.py`.** Reads the JSONL report, emits
   `docs/concepts/calibration.md`. Golden-file tested against a known
   input.
5. **One-shot calibration run.** Manual: maintainer runs the script
   once with a real `ANTHROPIC_API_KEY` and the pinned model snapshot.
   Output committed at `docs/calibration/v0.0.8-alpha-report.jsonl`.
6. **`docs/concepts/calibration.md`.** Rendered from the report.
   Committed.
7. **API promotion.** `src/ariadne_eval/__init__.py` adds six new
   imports + `__all__` entries: `Judge`, `JudgeParseError`,
   `JudgeVerdict`, `PlanQuality`, `StubJudge`, `TrajectoryJudge`.
   Top-level `__all__` grows from 54 → 60.
8. **Docs & CHANGELOG.** `docs/concepts/judges.md` cross-links
   calibration; mkdocs nav adds Calibration under Concepts;
   `CHANGELOG.md [Unreleased]` records the promotion and the kappa.
9. **Tag `v0.0.8-alpha`** on merge.

Out (deferred):

- `StepwiseJudge` (per-step verdicts) — Phase 6.2 or absorbed into
  Phase 7.
- Multiple judge models (gpt-4o-mini, open-weights) and multi-row kappa
  table — needs Phase 6.2.
- Prompt variants (zero-shot / rubric / few-shot) comparison.
- Ensemble judges (majority-vote / averaged-rubric).
- Krippendorff's α, percent agreement, FPR/FNR statistics.
- Interactive labeling CLI (rich-based, pause/resume) — unnecessary
  for hand-crafted synthetic fixtures; will be needed when sourcing
  trajectories from tau-bench (Phase 7).
- Kappa heatmap SVG — only meaningful with ≥2 judge models.
- METHODOLOGY.md "Judge Calibration" section — grows in Phase 6.2 when
  multiple judges exist to compare.
- README headline benchmark table — Phase 7 produces this from tau-bench.
- Tau-bench-sourced calibration set — Phase 7 produces real-agent
  trajectories that can replace or augment the synthetic set in a
  later phase.

## Why synthetic and why now

The Prompts.md Phase 6 prompt called for "50 hand-labeled trajectories
sampled from the reference agent on tau-bench tasks." Tau-bench
integration is Phase 7, so that path is unavailable. Two options:

- **Wait for tau-bench.** Defer judge promotion another full phase.
  Judges remain namespace-private. README cannot say what kappa the
  judge achieves. Hard Rule #5 stays in force.
- **Ship synthetic now.** Author 51 hand-crafted trajectories spanning
  known plan-quality buckets, label them, run the judge, publish the
  kappa with documented caveats ("synthetic gold set; real-agent
  recalibration coming in a later phase").

We chose **synthetic now**. Honesty about the gold-set provenance
preserves the methodology; kappa against a balanced synthetic set still
measures whether the judge can distinguish plan-quality categories,
which is what the metric needs to do.

## Gold-set construction

**File:** `tests/data/gold_plans.jsonl`

**One line per fixture:**

```json
{
  "trajectory": {
    "id": "01J…",
    "task": "Find the population of Berlin and divide it by 100",
    "agent_name": "synth",
    "agent_version": "0.0.0",
    "model_id": "synth/agent",
    "started_at": "2026-05-31T00:00:00Z",
    "finished_at": "2026-05-31T00:00:01Z",
    "final_status": "succeeded",
    "final_answer": "37000"
  },
  "steps": [
    {
      "id": "01J…",
      "trajectory_id": "01J…",
      "name": "llm",
      "started_at": "2026-05-31T00:00:00Z",
      "finished_at": "2026-05-31T00:00:00.010000Z",
      "status": "succeeded",
      "payload": {
        "step_type": "llm_call",
        "model_id": "synth/agent",
        "prompt_messages": [{"role": "user", "content": "…"}],
        "completion": "I'll first search for Berlin's population, then divide by 100 using the calculator.",
        "input_tokens": 1,
        "output_tokens": 1,
        "latency_ms": 1.0,
        "cost_usd": 0.0
      }
    }
  ],
  "gold_label": "pass"
}
```

**Bucket recipes** (the editorial spec for fixture authoring):

| Bucket | n | What the LLM-step `completion` looks like |
|---|---|---|
| `pass` | 17 | States what will happen, decomposes into 2-3 steps, names actual tool names (or tool-like verbs that map cleanly to the agent's tools) |
| `partial` | 17 | One axis missing: clear-but-not-decomposed, OR decomposed-but-vague-tools, OR decomposed-but-irrelevant-tools |
| `fail` | 17 | Vague filler ("I'll figure this out"), or generic chatter, or off-topic, or no plan at all ("Let me think…") |

**Task diversity** (across all 51): arithmetic, fact lookup, multi-hop
reasoning (lookup → arithmetic), explicitly off-task or
under-constrained tasks. The diversity prevents the gold set from being
trivially easy for the judge by having every example follow the same
template.

**Loader:** `tests/data/_load_gold_plans.py` — test-private module
that reads the JSONL, validates with the existing
`ariadne_eval.core.trajectory` Pydantic models, and yields
`(Trajectory, list[Step], gold_label)`. The calibration script's
`--source synth` path imports this helper via a path mutation
(`sys.path` already covers `scripts/`; we extend the same pattern for
`tests/data/`).

## Calibration run

**Command (run once, manually, with a real API key):**

```bash
uv run python scripts/build_calibration_set.py \
    --source synth \
    --gold-labels tests/data/gold_plans.jsonl \
    --judge-model anthropic/claude-sonnet-4-6 \
    --out docs/calibration/v0.0.8-alpha-report.jsonl \
    --concurrency 4
```

**Model pinning.** The dated snapshot alias
(`claude-sonnet-4-6`) is what gets committed to the report and to
CHANGELOG. A floating alias (`claude-sonnet`) would silently rebind to
new snapshots and quietly invalidate the kappa.

**Temperature.** `temperature=0.0` (the `TrajectoryJudge` default).
Documented in the report's `meta` line.

## Extended report shape

The Phase 6 script produces per-row lines + one trailing summary.
Phase 6.1 adds two more trailing lines:

```jsonl
{"trajectory_id":"01J…","gold_label":"pass","judge_label":"pass","judge_score":0.85,"judge_rationale":"…"}
…
{"_kind":"summary","n":51,"kappa":0.62,"interpretation":"substantial","label_set":["fail","partial","pass"]}
{"_kind":"confusion","labels":["fail","partial","pass"],"matrix":[[14,3,0],[2,12,3],[0,2,15]],"per_label":{"pass":{"precision":0.833,"recall":0.882,"support":17},"partial":{"precision":0.706,"recall":0.706,"support":17},"fail":{"precision":0.875,"recall":0.824,"support":17}}}
{"_kind":"meta","judge_model":"anthropic/claude-sonnet-4-6","temperature":0.0,"system_prompt_sha256":"…","user_template_sha256":"…","run_date":"2026-05-31","ariadne_version":"0.0.8-alpha","n_gold":51}
```

**`confusion` line semantics:**
- `labels`: the label set in sorted order (matches the `summary` line).
- `matrix[i][j]` = count of `(gold=labels[i], judge=labels[j])`.
- `per_label[lbl].precision` = `tp / (tp + fp)` where tp = matrix[i][i],
  fp = sum of column i minus tp.
- `per_label[lbl].recall` = `tp / (tp + fn)` where fn = sum of row i
  minus tp.
- `per_label[lbl].support` = sum of row i (count of gold=lbl).
- Rounded to 3 decimals in the JSON; the docs renderer formats further.

**`meta` line semantics:**
- `judge_model`: the exact litellm model string passed at run time.
- `temperature`: floating point.
- `system_prompt_sha256` / `user_template_sha256`: sha256 hex digests
  of `PLAN_QUALITY_SYSTEM` and `PLAN_QUALITY_USER_TEMPLATE` at run
  time. Lets a future maintainer prove the prompt that produced this
  kappa is the prompt currently in code (or detect drift).
- `run_date`: `YYYY-MM-DD` (UTC) of the run.
- `ariadne_version`: `ariadne_eval.__version__` at run time.
- `n_gold`: total gold entries (= 51).

**Computation in script.** New code is pure NumPy + stdlib hashlib. No
new dependencies.

**Errors.** The script's existing per-trajectory error path (load-fail
/ parse-fail) is unchanged. The `confusion` block runs over the
`judged_pairs` list only — failed rows do not poison the matrix or the
per-label counts. The `meta.n_gold` reports total fixtures; `summary.n`
reports successfully-judged trajectories. If they differ, the docs
page surfaces that.

## Rendered docs page

**File:** `docs/concepts/calibration.md`

**Generator:** `scripts/render_calibration_md.py` — reads the JSONL
report and writes the Markdown. Idempotent. Golden-file tested.

**Page structure (sections in order):**

1. **Headline** (one sentence):
   > **TrajectoryJudge agrees with the maintainer at κ = 0.XX (interpretation), n = 51, anthropic/claude-sonnet-4-6, 2026-05-31.**

2. **Confusion matrix** as a Markdown table. Rows = gold,
   columns = judge, raw counts, row totals on the right edge.

3. **Per-label precision / recall / support** as a second small table.

4. **Methodology** (3-4 short paragraphs):
   - Gold-set construction (51 fixtures, 17 per label, task diversity).
   - One-person hand-labeling, single judge model, `temperature=0.0`.
   - Prompt hashes (first 12 hex chars of sha256) — proves the
     calibrated prompts match the code.
   - Honest caveats: synthetic gold set, single labeler, single judge.

5. **Limitations** (short bullets):
   - Synthetic set written by one person — unknown biases.
   - Single judge model — multi-model agreement deferred.
   - Landis-Koch bands are conventional; "good kappa" depends on use
     case.

6. **Recalibration recipe** (3-line code block) showing how a user
   recreates the report against their own gold-labels file.

7. **Link to raw report** at `docs/calibration/v0.0.8-alpha-report.jsonl`.

**Nav placement.** `Concepts > Calibration`, alphabetically between
`Judges` and `Metrics`. `docs/concepts/judges.md` adds a one-line link:
"See [Calibration](./calibration.md) for current judge-agreement numbers."

## API promotion

**`src/ariadne_eval/__init__.py`:**
- Add `from ariadne_eval.eval.judges import Judge, JudgeParseError, JudgeVerdict, StubJudge, TrajectoryJudge`.
- Add `from ariadne_eval.eval.metrics.plan_quality import PlanQuality`.
- Append 6 names to `__all__` (alphabetized):
  `Judge`, `JudgeParseError`, `JudgeVerdict`, `PlanQuality`,
  `StubJudge`, `TrajectoryJudge`.
- Result: top-level `__all__` 54 → 60.

**`src/ariadne_eval/eval/judges/__init__.py`:** Replace the existing
"intentionally NOT re-exported until Phase 6.1 calibration ships"
docstring sentence with a calibration-result sentence:
> "Calibrated at κ = X.XX (Y) against the 51-example synthetic gold set
> in `docs/concepts/calibration.md`."

The actual κ and band values are filled in after the calibration run.
The docstring update is part of the same commit that ships the report.

**Public-API test:**
`tests/unit/eval/test_public_api_phase6_1.py` asserts:
- Each of the 6 names is in `ariadne_eval.__all__`.
- `from ariadne_eval import Judge, ...` works.
- The file `docs/calibration/v0.0.8-alpha-report.jsonl` exists from the
  repo root (resolved via a `_REPO = Path(__file__).resolve().parents[3]`
  pattern). This is what makes Hard Rule #5 enforced by tests, not by
  reviewer attention — if anyone deletes the report, the test fails.

## Testing strategy

| Test | Type | File |
|---|---|---|
| `gold_plans.jsonl` loads via Pydantic; exactly 51 entries; 17 per bucket | fast unit | `tests/unit/eval/test_gold_plans.py` |
| `--source synth` reads JSONL directly and produces the same per-row shape as `--source store` | fast unit | extend `tests/unit/scripts/test_build_calibration_set.py` |
| Confusion-matrix computation matches a hand-computed expected on a small known input | fast unit | same |
| Per-label precision/recall on a known input matches manually-derived values | fast unit | same |
| `meta` block carries `judge_model`, `temperature`, `system_prompt_sha256`, `user_template_sha256`, `run_date`, `ariadne_version`, `n_gold` | fast unit | same |
| `render_calibration_md.py` produces a known-good Markdown output (golden-file diff) | fast unit | `tests/unit/scripts/test_render_calibration_md.py` |
| Top-level `__all__` contains the 6 judge symbols AND the report file exists at the expected path | fast unit | `tests/unit/eval/test_public_api_phase6_1.py` |
| `mkdocs build --strict` succeeds | manual gate | in Task 14 verification step |

**Coverage gate:** ≥90% on `scripts/build_calibration_set.py`
(already 94% from Phase 6, will stay ≥90% with the extension) and
≥90% on `scripts/render_calibration_md.py`.

**Out of test surface:** the calibration run itself. The run is a
one-shot manual step the maintainer executes once with a real API key;
its output is the committed evidence. CI never re-runs the judge —
the report file IS the evidence.

## Risks & non-obvious decisions

- **Why synthetic over tau-bench-sourced.** Tau-bench is Phase 7.
  Waiting would defer judge promotion another full phase and prevent
  the README from carrying a calibration number. Synthetic with
  documented provenance is the honest compromise.
- **Why hand-crafted instead of LLM-generated fixtures.** An
  LLM-generated gold set evaluated by another LLM is just
  judge-vs-judge agreement, not judge-vs-human — undermining the kappa
  claim. The maintainer's hand-authored set is the human side of the
  agreement metric.
- **Why one judge model.** Multi-model comparison (the original Phase 6
  prompt's six-cell kappa heatmap) needs 6× the cassettes plus a
  multi-row docs table. Phase 6.1 deliberately ships one row to keep
  the cycle time short; Phase 6.2 (or a later 6.x) widens to the full
  comparison.
- **Why 51 (and not 50).** Mathematically a balanced 17/17/17 gives a
  cleaner confusion matrix than 16/17/17 + one tiebreaker. 51 > 50
  meets the "≥50" headline-deliverable threshold even after a single
  judge-parse failure.
- **Why commit the report verbatim instead of regenerating in CI.**
  Regenerating means CI needs an API key (security exposure) and pays
  per-build (cost). Committed evidence is reproducible by anyone with
  the same model snapshot + key.
- **Why prompt hashes in `meta`.** A prompt edit between Phase 6.1 and
  a future re-calibration is exactly the silent invalidation Hard
  Rule #5 protects against. Hashes in the report make that diffable.
- **Why a separate render script instead of hand-editing the
  Markdown.** Hand-editing lets the rendered table drift from the
  JSONL. The renderer is small, testable, and re-runnable.
- **Why test enforces report-file-exists.** Hard Rule #5 says
  judges-without-calibration cannot ship. Encoding that as a test
  failure when the report file is missing makes the rule enforceable
  by the test suite, not by reviewer attention.

## Definition of Done (Phase 6.1)

Standard project DoD plus:

- [ ] `tests/data/gold_plans.jsonl` has 51 entries, 17 per label, all
      loadable through the existing Pydantic models.
- [ ] `scripts/build_calibration_set.py` accepts `--source synth|store`
      and emits `confusion` + `meta` trailing lines.
- [ ] `scripts/render_calibration_md.py` exists with a golden-file
      test.
- [ ] Calibration run executed once with `anthropic/claude-sonnet-4-6`;
      `docs/calibration/v0.0.8-alpha-report.jsonl` committed.
- [ ] `docs/concepts/calibration.md` rendered from the report,
      committed; mkdocs nav adds Calibration under Concepts;
      `docs/concepts/judges.md` cross-links it.
- [ ] Top-level `__all__` adds `Judge`, `JudgeParseError`,
      `JudgeVerdict`, `PlanQuality`, `StubJudge`, `TrajectoryJudge`.
- [ ] `docs/reference/judges.md` continues to render the now-top-level
      surface via existing mkdocstrings blocks (no doc change needed).
- [ ] `eval/judges/__init__.py` docstring updated with the calibrated
      kappa + band; pre-existing "deferred to 6.1" sentence removed.
- [ ] `CHANGELOG.md [Unreleased]`: judge symbols moved to top-level
      public; calibration report shipped; kappa = X (band).
- [ ] All gates: fast tests + integration tests + mypy strict + ruff +
      mkdocs strict — all clean. Coverage on
      `scripts/build_calibration_set.py` and
      `scripts/render_calibration_md.py` ≥ 90%.
- [ ] Tagged `v0.0.8-alpha` on `main` after `--no-ff` merge.
- [ ] Memory updated: phase state notes the judge symbols are now
      top-level public, the kappa value, the next phase = Phase 7
      tau-bench (canonical sequence from `Prompts.md`, not the
      earlier mis-cached "Phase 7 = StepwiseJudge" understanding).
