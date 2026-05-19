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
