# Example 04: Plan-quality evaluation (async)

Shows how to compose deterministic metrics with the async `PlanQuality`
judge-backed metric, all through `Runner.aevaluate`. Uses `StubJudge` so
no API key is required.

## Run

```bash
uv run python examples/04_plan_quality/main.py
```

## Expected output

The example pins `seed=0`, so the means and CIs are exact and reproducible:

```
n_cases = 3
  final_answer_match     mean=0.667  95% CI=[0.000, 1.000]  n=3
  tool_accuracy          mean=1.000  95% CI=[1.000, 1.000]  n=3
  step_efficiency        mean=1.000  95% CI=[1.000, 1.000]  n=3
  plan_quality           mean=0.800  95% CI=[0.800, 0.800]  n=3
```

## Swapping in a real LLM judge

Replace:

```python
judge = StubJudge(JudgeVerdict(score=0.8, label="pass", rationale="..."))
```

with:

```python
from ariadne_eval.eval.judges import TrajectoryJudge
judge = TrajectoryJudge(model="claude-sonnet")
```

and ensure your `ANTHROPIC_API_KEY` (or other provider key) is set in
the environment.
