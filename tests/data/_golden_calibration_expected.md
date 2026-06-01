# Calibration

> **TrajectoryJudge agrees with the maintainer at κ = 0.50 (moderate), n = 3, anthropic/claude-sonnet-4-6, 2026-05-31.**

## Confusion matrix

Rows = maintainer (gold) labels; columns = judge labels. Cell values are raw counts.

| gold ↓ / judge → | fail | partial | pass | total |
|---|---|---|---|---|
| **fail** | 1 | 0 | 0 | 1 |
| **partial** | 0 | 0 | 1 | 1 |
| **pass** | 0 | 0 | 1 | 1 |

## Per-label precision, recall, and support

| label | precision | recall | support |
|---|---|---|---|
| fail | 1.000 | 1.000 | 1 |
| partial | 0.000 | 0.000 | 1 |
| pass | 0.500 | 1.000 | 1 |

## Methodology

- Gold set: 3 synthetic trajectories spanning plan-quality buckets (`pass`, `partial`, `fail`), balanced 1/1/1.
- One maintainer authored the labels; one judge model was evaluated.
- Judge configuration: model `anthropic/claude-sonnet-4-6`, `temperature=0.0`. Prompt-hash digests (first 12 hex): system=`abcdef123456`, user-template=`fedcba098765`. Re-running with the same model and the same prompts (verified by hash) reproduces the numbers above modulo provider determinism.

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
