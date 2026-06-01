# Calibration

> **TrajectoryJudge agrees with the maintainer at κ = 0.32 (fair), n = 51, anthropic/claude-sonnet-4-6, 2026-06-01.**

## Confusion matrix

Rows = maintainer (gold) labels; columns = judge labels. Cell values are raw counts.

| gold ↓ / judge → | fail | partial | pass | total |
|---|---|---|---|---|
| **fail** | 17 | 0 | 0 | 17 |
| **partial** | 15 | 2 | 0 | 17 |
| **pass** | 0 | 8 | 9 | 17 |

## Per-label precision, recall, and support

| label | precision | recall | support |
|---|---|---|---|
| fail | 0.531 | 1.000 | 17 |
| partial | 0.200 | 0.118 | 17 |
| pass | 1.000 | 0.529 | 17 |

## Methodology

- Gold set: 51 synthetic trajectories spanning plan-quality buckets (`pass`, `partial`, `fail`), balanced 17/17/17.
- One maintainer authored the labels; one judge model was evaluated.
- Judge configuration: model `anthropic/claude-sonnet-4-6`, `temperature=0.0`. Prompt-hash digests (first 12 hex): system=`0bf624f77ae5`, user-template=`4d6a7524da3d`. Re-running with the same model and the same prompts (verified by hash) reproduces the numbers above modulo provider determinism.

## Limitations

- The gold set is synthetic and was written by one person; biases are present but unknown.
- A single judge model is evaluated. Cross-model agreement (e.g. `gpt-4o-mini`, open-weights) is deferred to a later phase.
- Kappa bands follow Landis-Koch (1977); "good enough" kappa depends on use case.

## Recalibration

```bash
uv run python scripts/build_calibration_set.py \
    --source synth \
    --gold-labels tests/data/gold_plans.jsonl \
    --judge-model anthropic/claude-sonnet-4-6 \
    --out docs/calibration/<version>-report.jsonl \
    --concurrency 4

uv run python scripts/render_calibration_md.py \
    --report docs/calibration/<version>-report.jsonl \
    --out docs/concepts/calibration.md
```

## Raw report

The JSONL report this page was rendered from is committed at
[`docs/calibration/v0.0.8-alpha-report.jsonl`](../calibration/v0.0.8-alpha-report.jsonl).
