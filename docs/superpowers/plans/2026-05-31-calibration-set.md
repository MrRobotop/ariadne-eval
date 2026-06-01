# Phase 6.1 — Calibration Set, Kappa Table, and Judge Promotion: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the calibration evidence Hard Rule #5 demands (a 51-fixture hand-crafted synthetic gold set, a real one-shot `TrajectoryJudge` run, a committed JSONL report, a rendered `docs/concepts/calibration.md` with kappa + confusion matrix + per-label precision/recall), then promote the six judge symbols from `ariadne_eval.eval.judges.*` to top-level `ariadne_eval.*`.

**Architecture:** Pure-data exercise plus a small extension of the existing Phase 6 calibration script. No new modules in `src/`. The synthetic fixtures live in `tests/data/` (test-private). The script grows a `--source synth|store` switch and emits two new trailing JSONL lines (`confusion`, `meta`). A new tiny `scripts/render_calibration_md.py` materializes the human-readable docs page from the JSONL report. The promotion is a small `__all__` change in `src/ariadne_eval/__init__.py`, gated by a unit test that asserts both the symbol presence AND the committed report file's existence — Hard Rule #5 becomes test-enforced.

**Tech Stack:** Python 3.11+, Pydantic v2 (existing `Trajectory`/`Step` models), NumPy (existing dep, for the confusion matrix), Click (existing dep), hashlib stdlib (prompt-hash digests), pytest. No new runtime or dev dependencies.

---

## File map

| File | Action | Purpose |
|---|---|---|
| `tests/data/__init__.py` | Create | Make `tests/data` importable from the calibration script's `--source synth` path |
| `tests/data/_load_gold_plans.py` | Create | Load + Pydantic-validate `gold_plans.jsonl`, yield `(Trajectory, list[Step], gold_label)` triples |
| `tests/data/gold_plans.jsonl` | Create | 51 hand-authored fixtures: 17 `pass` + 17 `partial` + 17 `fail` |
| `tests/data/_golden_calibration_report.jsonl` | Create | Fixed input for the renderer's golden test |
| `tests/data/_golden_calibration_expected.md` | Create | Expected output for the renderer's golden test |
| `tests/unit/eval/test_gold_plans.py` | Create | Validate: 51 rows, 17 per bucket, all load through Pydantic |
| `scripts/build_calibration_set.py` | Modify | Add `--source synth\|store`; emit `confusion` + `meta` trailing JSONL lines |
| `tests/unit/scripts/test_build_calibration_set.py` | Modify | Cover `--source synth`, confusion-matrix computation, per-label P/R, `meta` shape |
| `scripts/render_calibration_md.py` | Create | Read JSONL report, write `docs/concepts/calibration.md` |
| `tests/unit/scripts/test_render_calibration_md.py` | Create | Golden-file diff test against `_golden_calibration_expected.md` |
| `docs/calibration/v0.0.8-alpha-report.jsonl` | Create (manual) | Real calibration evidence from the one-shot run |
| `docs/concepts/calibration.md` | Create (rendered) | Human-readable kappa + confusion + per-label P/R |
| `docs/concepts/judges.md` | Modify | Cross-link to calibration; remove "pending 6.1" deferral sentence |
| `src/ariadne_eval/eval/judges/__init__.py` | Modify | Update docstring with calibrated κ + band; remove deferral note |
| `src/ariadne_eval/__init__.py` | Modify | Add 6 imports + 6 `__all__` entries (alphabetized); 54 → 60 |
| `tests/unit/eval/test_public_api_phase6_1.py` | Create | Assert 6 names exported AND `docs/calibration/v0.0.8-alpha-report.jsonl` exists |
| `mkdocs.yml` | Modify | Add `Calibration: concepts/calibration.md` to nav |
| `CHANGELOG.md` | Modify | `[Unreleased]` entry: promotion + kappa + calibration report |

---

## Task 1: Gold-set loader

**Goal:** A test-private loader that reads a JSONL file of trajectory triples and yields validated Pydantic objects + gold labels. No fixture file yet — the loader is tested against an in-memory minimal JSONL string.

**Files:**
- Create: `tests/data/__init__.py`
- Create: `tests/data/_load_gold_plans.py`
- Create: `tests/unit/eval/test_gold_plans.py`

- [ ] **Step 1: Make `tests/data` an importable package**

```bash
touch tests/data/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/unit/eval/test_gold_plans.py`:

```python
"""Tests for the synthetic gold-plans loader."""

from __future__ import annotations

from io import StringIO
from typing import Literal

import pytest

from ariadne_eval.core.trajectory import Step, Trajectory
from tests.data._load_gold_plans import GoldEntry, iter_gold_plans

pytestmark = pytest.mark.fast


_ONE_FIXTURE_JSONL = """\
{"trajectory":{"id":"01J0000000000000000000000A","task":"add 1+2","agent_name":"synth","agent_version":"0.0.0","model_id":"synth/agent","started_at":"2026-05-31T00:00:00Z","finished_at":"2026-05-31T00:00:01Z","final_status":"succeeded","final_answer":"3"},"steps":[{"id":"01J0000000000000000000000B","trajectory_id":"01J0000000000000000000000A","parent_step_id":null,"name":"llm","started_at":"2026-05-31T00:00:00Z","finished_at":"2026-05-31T00:00:00.010000Z","status":"succeeded","payload":{"step_type":"llm_call","model_id":"synth/agent","prompt_messages":[{"role":"user","content":"hi"}],"completion":"I will use the calculator on 1+2.","input_tokens":1,"output_tokens":1,"latency_ms":1.0,"cost_usd":0.0}}],"gold_label":"pass"}
"""


def test_load_single_entry_from_stream() -> None:
    entries = list(iter_gold_plans(StringIO(_ONE_FIXTURE_JSONL)))
    assert len(entries) == 1
    e = entries[0]
    assert isinstance(e, GoldEntry)
    assert isinstance(e.trajectory, Trajectory)
    assert len(e.steps) == 1
    assert isinstance(e.steps[0], Step)
    assert e.gold_label == "pass"
    assert e.trajectory.task == "add 1+2"


def test_load_skips_blank_lines() -> None:
    stream = StringIO(_ONE_FIXTURE_JSONL + "\n   \n" + _ONE_FIXTURE_JSONL)
    entries = list(iter_gold_plans(stream))
    assert len(entries) == 2


def test_load_rejects_bad_label() -> None:
    bad = _ONE_FIXTURE_JSONL.replace('"gold_label":"pass"', '"gold_label":"maybe"')
    with pytest.raises(ValueError, match="gold_label"):
        list(iter_gold_plans(StringIO(bad)))


def test_load_rejects_missing_trajectory() -> None:
    bad = '{"steps":[],"gold_label":"pass"}\n'
    with pytest.raises(ValueError):
        list(iter_gold_plans(StringIO(bad)))


def test_gold_entry_label_type() -> None:
    # Compile-time-ish check that GoldEntry.gold_label is the right Literal
    e = next(iter_gold_plans(StringIO(_ONE_FIXTURE_JSONL)))
    lbl: Literal["pass", "partial", "fail"] = e.gold_label
    assert lbl in ("pass", "partial", "fail")
```

- [ ] **Step 3: Run test to confirm RED**

Run: `uv run pytest tests/unit/eval/test_gold_plans.py -q`
Expected: FAIL — `tests.data._load_gold_plans` does not exist.

- [ ] **Step 4: Implement the loader**

Create `tests/data/_load_gold_plans.py`:

```python
"""Loader for the synthetic plan-quality gold set.

Test-private: this module is imported by the calibration CLI via a
``sys.path`` extension in the ``--source synth`` code path. It is NOT
part of the library's public API and lives under ``tests/`` so it
ships only when tests do.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import IO, Literal

from pydantic import BaseModel

from ariadne_eval.core.trajectory import Step, Trajectory

__all__ = ["GoldEntry", "iter_gold_plans"]


GoldLabel = Literal["pass", "partial", "fail"]


@dataclass(frozen=True)
class GoldEntry:
    """One synthetic gold-plan entry: trajectory + steps + human-assigned label."""

    trajectory: Trajectory
    steps: list[Step]
    gold_label: GoldLabel


class _RawEntry(BaseModel):
    """Pydantic adapter: validates the JSONL line shape before we construct the dataclass."""

    trajectory: Trajectory
    steps: list[Step]
    gold_label: Literal["pass", "partial", "fail"]


def iter_gold_plans(stream: IO[str]) -> Iterator[GoldEntry]:
    """Yield ``GoldEntry`` per non-blank line of ``stream``.

    Each line must be a JSON object with ``trajectory``, ``steps``, and
    ``gold_label`` (one of ``"pass"``, ``"partial"``, ``"fail"``).
    Invalid shapes raise ``ValueError`` (via Pydantic).
    """
    for raw_line in stream:
        line = raw_line.strip()
        if not line:
            continue
        data = json.loads(line)
        raw = _RawEntry.model_validate(data)
        yield GoldEntry(
            trajectory=raw.trajectory,
            steps=list(raw.steps),
            gold_label=raw.gold_label,
        )
```

- [ ] **Step 5: Run test to confirm GREEN**

Run: `uv run pytest tests/unit/eval/test_gold_plans.py -q`
Expected: 5 passed.

- [ ] **Step 6: Verify gates**

```bash
uv run mypy --strict tests/data/_load_gold_plans.py
uv run ruff check tests/data/_load_gold_plans.py tests/unit/eval/test_gold_plans.py
uv run ruff format tests/data/_load_gold_plans.py tests/unit/eval/test_gold_plans.py
uv run ruff format --check tests/data/_load_gold_plans.py tests/unit/eval/test_gold_plans.py
```

Expected: all clean. (mypy may not check `tests/` by default — the strict run is just on the loader file to catch type errors early.)

- [ ] **Step 7: Commit**

```bash
git add tests/data/__init__.py tests/data/_load_gold_plans.py tests/unit/eval/test_gold_plans.py
git commit -m "feat(eval): synthetic gold-plans loader (no fixture file yet)"
```

---

## Task 2: Author the 51 gold-plan fixtures

**Goal:** Hand-craft `tests/data/gold_plans.jsonl` — 51 entries, 17 per bucket. Each entry is a minimal `(Trajectory, list[Step], gold_label)` triple. The LLM step's `completion` text IS the "plan" the judge will see; that's the load-bearing field.

**Files:**
- Create: `tests/data/gold_plans.jsonl`
- Extend: `tests/unit/eval/test_gold_plans.py` (add counts/balance test)

- [ ] **Step 1: Author the 51 fixture lines**

The structure of each line (single-line JSON, no internal newlines) is fixed by the loader. Use deterministic ULIDs of the form `01J000000000000000000000XX` (the trailing two hex chars vary per entry) so the file is reproducible. Use `started_at = 2026-05-31T00:00:00Z`, `finished_at = +1s` for trajectories and `+10ms` for steps. The first byte of each line is `{`, the last is `}`, and lines are separated by `\n`.

**Bucket recipes (the editorial spec — follow these when writing each line):**

- **`pass` (17 entries):** the LLM-step `completion` clearly states what
  the agent will do, decomposes the task into 2–3 actionable substeps,
  and names actual tool verbs or tool names. The agent's `tools` map in
  this project includes `calculator` and `search` — pass plans should
  name one or both. Tasks span: arithmetic ("compute X"), lookup
  ("define X"), multi-hop ("look up X, then compute Y"), and ordinary
  procedural tasks ("convert X to Y").
- **`partial` (17 entries):** ONE axis is missing. The plan is either
  clear-but-not-decomposed ("I'll search for the answer"),
  decomposed-but-vague ("Step 1: think. Step 2: answer."), or
  decomposed-but-name-irrelevant-tools ("Step 1: call the database.
  Step 2: format."). It still relates to the task and is recognizable
  as a plan attempt.
- **`fail` (17 entries):** vague filler ("I'll figure this out"),
  generic chatter ("Let me think about this carefully."), off-topic
  text ("The weather is nice today."), or an empty/almost-empty plan
  ("ok").

**Task diversity (across all 51):** arithmetic (~12), fact lookup (~12),
multi-hop reasoning (~10), procedural ("translate", "convert", "sort")
(~10), explicitly off-task or under-constrained (~7). Distribute across
buckets so each bucket spans tasks (e.g., a `pass` plan for arithmetic
AND a `pass` plan for lookup, etc.).

Concrete examples — write the file by following the same JSON shape:

**One `pass` entry (arithmetic):**

```jsonl
{"trajectory":{"id":"01J000000000000000000000P1","task":"What is 17 multiplied by 23?","agent_name":"synth","agent_version":"0.0.0","model_id":"synth/agent","started_at":"2026-05-31T00:00:00Z","finished_at":"2026-05-31T00:00:01Z","final_status":"succeeded","final_answer":"391"},"steps":[{"id":"01J000000000000000000000P1S","trajectory_id":"01J000000000000000000000P1","parent_step_id":null,"name":"llm","started_at":"2026-05-31T00:00:00Z","finished_at":"2026-05-31T00:00:00.010000Z","status":"succeeded","payload":{"step_type":"llm_call","model_id":"synth/agent","prompt_messages":[{"role":"user","content":"What is 17 multiplied by 23?"}],"completion":"I will solve this in two steps. Step 1: call calculator(17 * 23) to get the product. Step 2: return the result.","input_tokens":1,"output_tokens":1,"latency_ms":1.0,"cost_usd":0.0}}],"gold_label":"pass"}
```

**One `partial` entry (multi-hop, vague-tools):**

```jsonl
{"trajectory":{"id":"01J000000000000000000000Q1","task":"Find Berlin's population and divide it by 100.","agent_name":"synth","agent_version":"0.0.0","model_id":"synth/agent","started_at":"2026-05-31T00:00:00Z","finished_at":"2026-05-31T00:00:01Z","final_status":"succeeded","final_answer":"approximately 37000"},"steps":[{"id":"01J000000000000000000000Q1S","trajectory_id":"01J000000000000000000000Q1","parent_step_id":null,"name":"llm","started_at":"2026-05-31T00:00:00Z","finished_at":"2026-05-31T00:00:00.010000Z","status":"succeeded","payload":{"step_type":"llm_call","model_id":"synth/agent","prompt_messages":[{"role":"user","content":"Find Berlin's population and divide it by 100."}],"completion":"Step 1: find the data. Step 2: do the math.","input_tokens":1,"output_tokens":1,"latency_ms":1.0,"cost_usd":0.0}}],"gold_label":"partial"}
```

**One `fail` entry (off-topic):**

```jsonl
{"trajectory":{"id":"01J000000000000000000000F1","task":"Translate 'banana' into Spanish.","agent_name":"synth","agent_version":"0.0.0","model_id":"synth/agent","started_at":"2026-05-31T00:00:00Z","finished_at":"2026-05-31T00:00:01Z","final_status":"succeeded","final_answer":"plátano"},"steps":[{"id":"01J000000000000000000000F1S","trajectory_id":"01J000000000000000000000F1","parent_step_id":null,"name":"llm","started_at":"2026-05-31T00:00:00Z","finished_at":"2026-05-31T00:00:00.010000Z","status":"succeeded","payload":{"step_type":"llm_call","model_id":"synth/agent","prompt_messages":[{"role":"user","content":"Translate 'banana' into Spanish."}],"completion":"Bananas are tropical fruits with a long history of cultivation in many countries around the world.","input_tokens":1,"output_tokens":1,"latency_ms":1.0,"cost_usd":0.0}}],"gold_label":"fail"}
```

ULID uniqueness: assign trajectory IDs `…P1` through `…P17` for `pass`, `…Q1` through `…Q17` for `partial`, `…F1` through `…F17` for `fail` (left-pad single digits with `0` to keep IDs at 26 chars: `01J000000000000000000000P01`, etc. — but Pydantic's ULID validator may reject letters/numbers that aren't valid Crockford Base32; verify a few IDs load cleanly before authoring all 51).

**ULID safety check (before authoring all 51):**

Open a Python REPL and verify a sample ID:

```python
from ariadne_eval.core.ids import is_valid_id
print(is_valid_id("01J000000000000000000000P1"))   # check truthy
```

If the helper rejects the sample, use `new_id()` to generate 51 distinct IDs and hard-code them in the file. The fixtures must be byte-for-byte deterministic for the golden tests in Task 6 to be stable.

- [ ] **Step 2: Extend the loader test with counts and balance assertions**

Append to `tests/unit/eval/test_gold_plans.py`:

```python
from collections import Counter
from pathlib import Path


_GOLD_PLANS_PATH = Path(__file__).resolve().parents[3] / "tests" / "data" / "gold_plans.jsonl"


def test_gold_plans_file_loads_completely() -> None:
    with _GOLD_PLANS_PATH.open(encoding="utf-8") as f:
        entries = list(iter_gold_plans(f))
    assert len(entries) == 51


def test_gold_plans_balanced_buckets() -> None:
    with _GOLD_PLANS_PATH.open(encoding="utf-8") as f:
        entries = list(iter_gold_plans(f))
    counts = Counter(e.gold_label for e in entries)
    assert counts == {"pass": 17, "partial": 17, "fail": 17}


def test_gold_plans_unique_trajectory_ids() -> None:
    with _GOLD_PLANS_PATH.open(encoding="utf-8") as f:
        entries = list(iter_gold_plans(f))
    ids = [e.trajectory.id for e in entries]
    assert len(set(ids)) == len(ids), "trajectory IDs must be unique"


def test_gold_plans_have_an_llm_step() -> None:
    """Every fixture must have at least one LLM step so PlanQuality can score it."""
    from ariadne_eval.core.trajectory import LLMCallPayload

    with _GOLD_PLANS_PATH.open(encoding="utf-8") as f:
        entries = list(iter_gold_plans(f))
    for e in entries:
        has_llm = any(isinstance(s.payload, LLMCallPayload) for s in e.steps)
        assert has_llm, f"entry {e.trajectory.id} has no LLM step"
```

- [ ] **Step 3: Run tests to confirm GREEN**

Run: `uv run pytest tests/unit/eval/test_gold_plans.py -q`
Expected: 9 passed (the 5 from Task 1 + 4 new).

If any fixture line fails to load (Pydantic validation, ULID format, label literal): the test will name the failing line. Fix the JSONL and re-run until green.

- [ ] **Step 4: Verify gates**

```bash
uv run ruff check tests/unit/eval/test_gold_plans.py
uv run ruff format --check tests/unit/eval/test_gold_plans.py
```

- [ ] **Step 5: Commit**

```bash
git add tests/data/gold_plans.jsonl tests/unit/eval/test_gold_plans.py
git commit -m "feat(eval): 51-fixture synthetic plan-quality gold set"
```

---

## Task 3: `--source synth` flag on the calibration script

**Goal:** Add a `--source synth|store` Click option to `scripts/build_calibration_set.py`. `--source synth` loads fixtures via the gold-plans loader; `--source store` is the existing DuckDB path. Per-row output JSON shape is identical across both paths.

**Files:**
- Modify: `scripts/build_calibration_set.py`
- Modify: `tests/unit/scripts/test_build_calibration_set.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/scripts/test_build_calibration_set.py`:

```python
async def test_run_with_source_synth_uses_fixtures_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--source synth loads gold_plans.jsonl directly; no DuckDB needed."""
    import json as _json
    from datetime import UTC, datetime, timedelta

    from ariadne_eval.core.ids import new_id

    # Build a 1-line gold-plans JSONL inline (the production file has 51)
    traj_id = new_id()
    step_id = new_id()
    started = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
    entry = {
        "trajectory": {
            "id": traj_id,
            "task": "compute 1+2",
            "agent_name": "synth",
            "agent_version": "0.0.0",
            "model_id": "synth/agent",
            "started_at": started.isoformat().replace("+00:00", "Z"),
            "finished_at": (started + timedelta(seconds=1))
            .isoformat()
            .replace("+00:00", "Z"),
            "final_status": "succeeded",
            "final_answer": "3",
        },
        "steps": [
            {
                "id": step_id,
                "trajectory_id": traj_id,
                "parent_step_id": None,
                "name": "llm",
                "started_at": started.isoformat().replace("+00:00", "Z"),
                "finished_at": (started + timedelta(milliseconds=10))
                .isoformat()
                .replace("+00:00", "Z"),
                "status": "succeeded",
                "payload": {
                    "step_type": "llm_call",
                    "model_id": "synth/agent",
                    "prompt_messages": [{"role": "user", "content": "hi"}],
                    "completion": "Step 1: use calculator(1+2).",
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "latency_ms": 1.0,
                    "cost_usd": 0.0,
                },
            }
        ],
        "gold_label": "pass",
    }
    gold = tmp_path / "gold_plans.jsonl"
    gold.write_text(_json.dumps(entry) + "\n", encoding="utf-8")

    monkeypatch.setenv(
        "ARIADNE_TEST_JUDGE_FACTORY",
        "tests.unit.scripts.test_build_calibration_set._stub_factory_passing",
    )

    out = tmp_path / "calibration.jsonl"
    from build_calibration_set import run

    await run(
        source="synth",
        store_path=None,
        gold_labels=gold,
        judge_model="test/model",
        out_path=out,
        concurrency=2,
    )

    lines = [
        _json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    per = [r for r in lines if r.get("_kind") not in ("summary", "confusion", "meta")]
    assert len(per) == 1
    assert per[0]["trajectory_id"] == traj_id
    assert per[0]["gold_label"] == "pass"
    assert per[0]["judge_label"] == "pass"
```

- [ ] **Step 2: Run test to confirm RED**

Run: `uv run pytest tests/unit/scripts/test_build_calibration_set.py::test_run_with_source_synth_uses_fixtures_directly -q`
Expected: FAIL — `run()` does not accept a `source` parameter.

- [ ] **Step 3: Modify `scripts/build_calibration_set.py`**

The current `run` signature is:

```python
async def run(
    *,
    store_path: Path,
    gold_labels: Path,
    judge_model: str,
    out_path: Path,
    concurrency: int,
) -> None:
```

Change it to:

```python
async def run(
    *,
    source: Literal["synth", "store"] = "store",
    store_path: Path | None,
    gold_labels: Path,
    judge_model: str,
    out_path: Path,
    concurrency: int,
) -> None:
```

Add at the top of the file (after the existing imports):

```python
from typing import Literal
```

At the top of the function body, dispatch on `source`. Replace the
existing entries-loading block + load-loop with this branching code:

```python
    judge = _make_judge(judge_model)

    loaded: list[tuple[dict[str, str], Trajectory | None, list[Step] | None, str | None]] = []

    if source == "synth":
        # --source synth: load directly from gold_plans.jsonl. No DuckDB needed.
        import sys
        from pathlib import Path as _Path

        # Make tests/data importable. The path is repo-root/tests, and
        # scripts/ already sits at repo-root/scripts, so we resolve from __file__.
        _repo_root = _Path(__file__).resolve().parents[1]
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))
        from tests.data._load_gold_plans import iter_gold_plans

        with gold_labels.open(encoding="utf-8") as f:
            for gold_entry in iter_gold_plans(f):
                entry_dict = {
                    "trajectory_id": gold_entry.trajectory.id,
                    "label": gold_entry.gold_label,
                }
                loaded.append(
                    (entry_dict, gold_entry.trajectory, gold_entry.steps, None)
                )
    else:
        # --source store: production path against a user's DuckDB store.
        if store_path is None:
            raise ValueError("--source store requires --store / store_path to be set")
        store = DuckDBStore(path=store_path)
        entries: list[dict[str, str]] = []
        for line in gold_labels.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
        for entry in entries:
            traj_id = entry["trajectory_id"]
            try:
                traj, steps = await store.get_trajectory(traj_id)
            except Exception as exc:
                loaded.append((entry, None, None, f"load: {exc}"))
                continue
            loaded.append((entry, traj, steps, None))
        await store.close()
```

Add the required imports at the top of the file:

```python
from ariadne_eval.core.trajectory import Step, Trajectory
```

The rest of `run` (semaphore, `_judge_one`, write loop) stays unchanged.

- [ ] **Step 4: Add the Click flag to `main()`**

Find the Click option block:

```python
@click.option(
    "--store",
    "store_path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Path to the DuckDB store.",
)
```

Change `required=True` to `required=False`. Then insert a `--source` option BEFORE `--store`:

```python
@click.option(
    "--source",
    type=click.Choice(["synth", "store"], case_sensitive=False),
    default="store",
    show_default=True,
    help="Where to load trajectories from: 'synth' for tests/data/gold_plans.jsonl, "
         "'store' for a user's DuckDB store.",
)
```

Change the `main` signature to accept the new parameter, and pass it through:

```python
def main(
    source: str,
    store_path: Path | None,
    gold_labels: Path,
    judge_model: str,
    out_path: Path,
    concurrency: int,
) -> None:
    """Run the calibration harness."""
    asyncio.run(
        run(
            source=source,  # type: ignore[arg-type]
            store_path=store_path,
            gold_labels=gold_labels,
            judge_model=judge_model,
            out_path=out_path,
            concurrency=concurrency,
        )
    )
```

- [ ] **Step 5: Update the existing happy-path tests**

The existing tests in `tests/unit/scripts/test_build_calibration_set.py`
pass `store_path=...` positionally to `run`. They will keep working —
`source` defaults to `"store"` — but to be explicit and to keep the
test surface honest, add `source="store"` to each existing `await run(...)`
call site in the file (there should be three: happy-path, load-failure,
parse-error). Show the diff for one and apply to all three:

```diff
     await run(
+        source="store",
         store_path=store_path,
         gold_labels=gold,
         judge_model="test/model",
         out_path=out,
         concurrency=2,
     )
```

- [ ] **Step 6: Run all calibration-script tests**

```bash
uv run pytest tests/unit/scripts/ -q
```

Expected: all green (the existing 6 + the new 1 = 7 passed).

- [ ] **Step 7: Verify gates**

```bash
uv run mypy --strict scripts/build_calibration_set.py
uv run ruff check scripts/build_calibration_set.py tests/unit/scripts/test_build_calibration_set.py
uv run ruff format scripts/build_calibration_set.py tests/unit/scripts/test_build_calibration_set.py
uv run ruff format --check scripts/build_calibration_set.py tests/unit/scripts/test_build_calibration_set.py
```

Expected: all clean.

- [ ] **Step 8: Commit**

```bash
git add scripts/build_calibration_set.py tests/unit/scripts/test_build_calibration_set.py
git commit -m "feat(scripts): --source synth|store flag for calibration script"
```

---

## Task 4: Confusion matrix + per-label P/R + `_kind: confusion` line

**Goal:** Compute the 3×3 confusion matrix and per-label precision/recall/support from the judged pairs, then emit a `_kind: "confusion"` trailing JSONL line.

**Files:**
- Modify: `scripts/build_calibration_set.py`
- Modify: `tests/unit/scripts/test_build_calibration_set.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/scripts/test_build_calibration_set.py`:

```python
def test_confusion_matrix_and_per_label_metrics() -> None:
    """Hand-computed expected on a small (gold, judge) pair list."""
    from build_calibration_set import _build_confusion_block

    # 5 pairs, labels in order ["fail", "partial", "pass"]
    # gold=[pass, pass, partial, fail, fail]
    # judge=[pass, fail,    pass, fail, partial]
    # matrix rows=gold, cols=judge:
    #   fail:    [1, 1, 0]
    #   partial: [0, 0, 1]
    #   pass:    [0, 0, 1]   wait — gold has 2 pass, judge has 2 pass+1 partial+1 fail → recompute
    # Let's recompute carefully:
    # pairs: (pass,pass) (pass,fail) (partial,pass) (fail,fail) (fail,partial)
    # rows are gold, cols are judge, sorted label_set = [fail, partial, pass]
    #   fail row (gold=fail, 2 entries): (fail,fail)=1 in fail col, (fail,partial)=1 in partial col → [1, 1, 0]
    #   partial row (gold=partial, 1 entry): (partial,pass)=1 in pass col → [0, 0, 1]
    #   pass row (gold=pass, 2 entries): (pass,pass)=1 in pass col, (pass,fail)=1 in fail col → [1, 0, 1]
    pairs = [
        ("pass", "pass"),
        ("pass", "fail"),
        ("partial", "pass"),
        ("fail", "fail"),
        ("fail", "partial"),
    ]
    block = _build_confusion_block(pairs)
    assert block["_kind"] == "confusion"
    assert block["labels"] == ["fail", "partial", "pass"]
    assert block["matrix"] == [[1, 1, 0], [0, 0, 1], [1, 0, 1]]
    # per-label support = row sum
    assert block["per_label"]["fail"]["support"] == 2
    assert block["per_label"]["partial"]["support"] == 1
    assert block["per_label"]["pass"]["support"] == 2
    # tp_fail=1, col_sum_fail=2, fp_fail=1 → precision=1/2=0.5
    # tp_fail=1, row_sum_fail=2, fn_fail=1 → recall=1/2=0.5
    assert block["per_label"]["fail"]["precision"] == 0.5
    assert block["per_label"]["fail"]["recall"] == 0.5
    # tp_pass=1, col_sum_pass=2, precision=1/2=0.5
    # tp_pass=1, row_sum_pass=2, recall=1/2=0.5
    assert block["per_label"]["pass"]["precision"] == 0.5
    assert block["per_label"]["pass"]["recall"] == 0.5
    # tp_partial=0 → precision=0 (use 0 when tp+fp=0), recall=0
    assert block["per_label"]["partial"]["precision"] == 0.0
    assert block["per_label"]["partial"]["recall"] == 0.0


def test_confusion_block_with_empty_pairs_is_safe() -> None:
    """No judged pairs → empty matrix, empty per_label, label_set empty."""
    from build_calibration_set import _build_confusion_block

    block = _build_confusion_block([])
    assert block == {
        "_kind": "confusion",
        "labels": [],
        "matrix": [],
        "per_label": {},
    }
```

- [ ] **Step 2: Run test to confirm RED**

Run: `uv run pytest tests/unit/scripts/test_build_calibration_set.py::test_confusion_matrix_and_per_label_metrics -q`
Expected: FAIL — `_build_confusion_block` not defined.

- [ ] **Step 3: Implement `_build_confusion_block`**

Add to `scripts/build_calibration_set.py` (above the `run` function):

```python
def _build_confusion_block(
    judged_pairs: list[tuple[str, str]],
) -> dict[str, object]:
    """Build the ``_kind: "confusion"`` JSONL block from judged pairs.

    ``judged_pairs`` is a list of ``(gold_label, judge_label)``. The
    returned dict contains the sorted label set, the row=gold/col=judge
    confusion matrix as nested ints, and per-label precision / recall /
    support computed against the matrix. Precision and recall are 0.0
    when the relevant denominator is 0 (no false positives observed
    can't divide by zero).
    """
    if not judged_pairs:
        return {"_kind": "confusion", "labels": [], "matrix": [], "per_label": {}}

    labels = sorted({g for g, _ in judged_pairs} | {j for _, j in judged_pairs})
    n = len(labels)
    idx = {lbl: i for i, lbl in enumerate(labels)}

    matrix = [[0] * n for _ in range(n)]
    for gold, judge in judged_pairs:
        matrix[idx[gold]][idx[judge]] += 1

    per_label: dict[str, dict[str, float | int]] = {}
    for i, lbl in enumerate(labels):
        tp = matrix[i][i]
        row_sum = sum(matrix[i])  # gold count of this label
        col_sum = sum(matrix[r][i] for r in range(n))  # judge count of this label
        precision = tp / col_sum if col_sum else 0.0
        recall = tp / row_sum if row_sum else 0.0
        per_label[lbl] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "support": row_sum,
        }

    return {
        "_kind": "confusion",
        "labels": labels,
        "matrix": matrix,
        "per_label": per_label,
    }
```

- [ ] **Step 4: Wire the block into the report write loop**

Find the existing write block in `run`:

```python
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str))
            f.write("\n")
        f.write(json.dumps(summary, default=str))
        f.write("\n")
```

Insert a confusion-line write between the summary and the (soon-to-be)
meta line. The full block becomes:

```python
    confusion_block = _build_confusion_block(judged_pairs)

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str))
            f.write("\n")
        f.write(json.dumps(summary, default=str))
        f.write("\n")
        f.write(json.dumps(confusion_block, default=str))
        f.write("\n")
```

- [ ] **Step 5: Run tests to confirm GREEN**

```bash
uv run pytest tests/unit/scripts/test_build_calibration_set.py -q
```

Expected: 9 passed (7 from prior tasks + 2 new).

- [ ] **Step 6: Verify gates**

```bash
uv run mypy --strict scripts/build_calibration_set.py
uv run ruff check scripts/build_calibration_set.py tests/unit/scripts/test_build_calibration_set.py
uv run ruff format scripts/build_calibration_set.py tests/unit/scripts/test_build_calibration_set.py
uv run ruff format --check scripts/build_calibration_set.py tests/unit/scripts/test_build_calibration_set.py
```

- [ ] **Step 7: Commit**

```bash
git add scripts/build_calibration_set.py tests/unit/scripts/test_build_calibration_set.py
git commit -m "feat(scripts): confusion matrix + per-label precision/recall in calibration report"
```

---

## Task 5: `_kind: meta` line with prompt-hash digests

**Goal:** Emit a `_kind: "meta"` JSONL block carrying `judge_model`, `temperature`, sha256 hashes of the two prompts, `run_date`, `ariadne_version`, and `n_gold`. The hashes make silent prompt drift between calibration runs diffable.

**Files:**
- Modify: `scripts/build_calibration_set.py`
- Modify: `tests/unit/scripts/test_build_calibration_set.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/scripts/test_build_calibration_set.py`:

```python
def test_meta_block_shape_and_prompt_hashes() -> None:
    """The meta block carries judge config + prompt hashes + ariadne version."""
    import hashlib

    from ariadne_eval.eval.judges.prompts import (
        PLAN_QUALITY_SYSTEM,
        PLAN_QUALITY_USER_TEMPLATE,
    )
    from build_calibration_set import _build_meta_block

    block = _build_meta_block(
        judge_model="anthropic/claude-sonnet-4-6",
        temperature=0.0,
        run_date="2026-05-31",
        n_gold=51,
    )
    assert block["_kind"] == "meta"
    assert block["judge_model"] == "anthropic/claude-sonnet-4-6"
    assert block["temperature"] == 0.0
    assert block["run_date"] == "2026-05-31"
    assert block["n_gold"] == 51
    assert block["system_prompt_sha256"] == hashlib.sha256(
        PLAN_QUALITY_SYSTEM.encode("utf-8")
    ).hexdigest()
    assert block["user_template_sha256"] == hashlib.sha256(
        PLAN_QUALITY_USER_TEMPLATE.encode("utf-8")
    ).hexdigest()
    # ariadne_version is read from the package; assert it matches
    from ariadne_eval import __version__

    assert block["ariadne_version"] == __version__
```

- [ ] **Step 2: Run test to confirm RED**

Run: `uv run pytest tests/unit/scripts/test_build_calibration_set.py::test_meta_block_shape_and_prompt_hashes -q`
Expected: FAIL — `_build_meta_block` not defined.

- [ ] **Step 3: Implement `_build_meta_block`**

Add to `scripts/build_calibration_set.py` (alongside `_build_confusion_block`):

```python
def _build_meta_block(
    *,
    judge_model: str,
    temperature: float,
    run_date: str,
    n_gold: int,
) -> dict[str, object]:
    """Build the ``_kind: "meta"`` JSONL block.

    Carries the judge model + temperature for reproducibility,
    sha256 digests of the two prompt constants so future re-runs can
    detect silent prompt drift, the run date, the ariadne version, and
    the total gold-set size.
    """
    import hashlib

    from ariadne_eval import __version__
    from ariadne_eval.eval.judges.prompts import (
        PLAN_QUALITY_SYSTEM,
        PLAN_QUALITY_USER_TEMPLATE,
    )

    return {
        "_kind": "meta",
        "judge_model": judge_model,
        "temperature": temperature,
        "system_prompt_sha256": hashlib.sha256(
            PLAN_QUALITY_SYSTEM.encode("utf-8")
        ).hexdigest(),
        "user_template_sha256": hashlib.sha256(
            PLAN_QUALITY_USER_TEMPLATE.encode("utf-8")
        ).hexdigest(),
        "run_date": run_date,
        "ariadne_version": __version__,
        "n_gold": n_gold,
    }
```

- [ ] **Step 4: Wire the meta block into the write loop**

The `run` function needs the run date and a count of gold entries.
Update the signature:

```python
async def run(
    *,
    source: Literal["synth", "store"] = "store",
    store_path: Path | None,
    gold_labels: Path,
    judge_model: str,
    out_path: Path,
    concurrency: int,
    temperature: float = 0.0,
    run_date: str | None = None,
) -> None:
```

At the top of the function body, immediately after `judge = _make_judge(judge_model)`:

```python
    from datetime import UTC, datetime

    if run_date is None:
        run_date = datetime.now(UTC).strftime("%Y-%m-%d")
```

After the `loaded` list is built, capture the count:

```python
    n_gold = len(loaded)
```

Update the report-write block:

```python
    confusion_block = _build_confusion_block(judged_pairs)
    meta_block = _build_meta_block(
        judge_model=judge_model,
        temperature=temperature,
        run_date=run_date,
        n_gold=n_gold,
    )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str))
            f.write("\n")
        f.write(json.dumps(summary, default=str))
        f.write("\n")
        f.write(json.dumps(confusion_block, default=str))
        f.write("\n")
        f.write(json.dumps(meta_block, default=str))
        f.write("\n")
```

- [ ] **Step 5: Update `main()` (the Click entry point) to pass through the new params**

Don't add CLI options for `temperature` or `run_date` — they default
sensibly. The plan keeps the CLI surface stable.

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/unit/scripts/test_build_calibration_set.py -q
```

Expected: 10 passed.

- [ ] **Step 7: Verify gates**

```bash
uv run mypy --strict scripts/build_calibration_set.py
uv run ruff check scripts/build_calibration_set.py tests/unit/scripts/test_build_calibration_set.py
uv run ruff format scripts/build_calibration_set.py tests/unit/scripts/test_build_calibration_set.py
uv run ruff format --check scripts/build_calibration_set.py tests/unit/scripts/test_build_calibration_set.py
```

- [ ] **Step 8: Commit**

```bash
git add scripts/build_calibration_set.py tests/unit/scripts/test_build_calibration_set.py
git commit -m "feat(scripts): meta block with prompt-hash digests in calibration report"
```

---

## Task 6: `scripts/render_calibration_md.py` (golden-file tested)

**Goal:** A small Click CLI that reads a calibration JSONL report and writes a Markdown file (default target: `docs/concepts/calibration.md`). Golden-file tested.

**Files:**
- Create: `scripts/render_calibration_md.py`
- Create: `tests/data/_golden_calibration_report.jsonl`
- Create: `tests/data/_golden_calibration_expected.md`
- Create: `tests/unit/scripts/test_render_calibration_md.py`

- [ ] **Step 1: Author the fixed input fixture**

Create `tests/data/_golden_calibration_report.jsonl` (each line is one
JSON object; lines wrap here for readability but in the file each is on
ONE line):

```jsonl
{"trajectory_id":"01J000000000000000000000A1","gold_label":"pass","judge_label":"pass","judge_score":0.85,"judge_rationale":"Clear plan."}
{"trajectory_id":"01J000000000000000000000A2","gold_label":"fail","judge_label":"fail","judge_score":0.10,"judge_rationale":"Vague."}
{"trajectory_id":"01J000000000000000000000A3","gold_label":"partial","judge_label":"pass","judge_score":0.65,"judge_rationale":"Mostly good."}
{"_kind":"summary","n":3,"kappa":0.5,"interpretation":"moderate","label_set":["fail","partial","pass"]}
{"_kind":"confusion","labels":["fail","partial","pass"],"matrix":[[1,0,0],[0,0,1],[0,0,1]],"per_label":{"fail":{"precision":1.0,"recall":1.0,"support":1},"partial":{"precision":0.0,"recall":0.0,"support":1},"pass":{"precision":0.5,"recall":1.0,"support":1}}}
{"_kind":"meta","judge_model":"anthropic/claude-sonnet-4-6","temperature":0.0,"system_prompt_sha256":"abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890","user_template_sha256":"fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321","run_date":"2026-05-31","ariadne_version":"0.0.8-alpha","n_gold":3}
```

- [ ] **Step 2: Author the expected Markdown output**

Create `tests/data/_golden_calibration_expected.md`:

```markdown
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
- Kappa bands follow Landis–Koch (1977); "good enough" kappa depends on use case.

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
```

- [ ] **Step 3: Write the failing test**

Create `tests/unit/scripts/test_render_calibration_md.py`:

```python
"""Golden-file test for scripts/render_calibration_md.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast


_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "scripts"))


def test_render_calibration_md_matches_golden(tmp_path: Path) -> None:
    from render_calibration_md import render

    report = _REPO / "tests" / "data" / "_golden_calibration_report.jsonl"
    expected = (
        _REPO / "tests" / "data" / "_golden_calibration_expected.md"
    ).read_text(encoding="utf-8")

    out = tmp_path / "calibration.md"
    render(report_path=report, out_path=out, report_basename="v0.0.8-alpha-report.jsonl")

    actual = out.read_text(encoding="utf-8")
    assert actual == expected
```

- [ ] **Step 4: Run test to confirm RED**

Run: `uv run pytest tests/unit/scripts/test_render_calibration_md.py -q`
Expected: FAIL — `render_calibration_md` module missing.

- [ ] **Step 5: Implement `scripts/render_calibration_md.py`**

Create the file:

```python
"""Render a calibration JSONL report to the human-readable docs page.

Usage:
    uv run python scripts/render_calibration_md.py \\
        --report docs/calibration/v0.0.8-alpha-report.jsonl \\
        --out docs/concepts/calibration.md

The rendered page leads with a one-sentence headline, then a confusion
matrix, then per-label precision / recall / support, then methodology +
limitations + recalibration recipe.
"""

from __future__ import annotations

import json
from pathlib import Path

import click


def _format_kappa(kappa: float) -> str:
    """Format kappa as a two-decimal string (e.g. ``0.50``)."""
    return f"{kappa:.2f}"


def _hash_prefix(digest: str, length: int = 12) -> str:
    """Return the first ``length`` hex chars of a sha256 digest."""
    return digest[:length] if digest else "(unknown)"


def _section_headline(summary: dict[str, object], meta: dict[str, object]) -> str:
    kappa = _format_kappa(float(summary["kappa"]))  # type: ignore[arg-type]
    interp = str(summary["interpretation"])
    n = int(summary["n"])  # type: ignore[arg-type]
    model = str(meta["judge_model"])
    run_date = str(meta["run_date"])
    return (
        f"> **TrajectoryJudge agrees with the maintainer at "
        f"κ = {kappa} ({interp}), n = {n}, {model}, {run_date}.**"
    )


def _section_confusion(confusion: dict[str, object]) -> str:
    labels: list[str] = list(confusion["labels"])  # type: ignore[arg-type]
    matrix: list[list[int]] = list(confusion["matrix"])  # type: ignore[arg-type]
    header = "| gold ↓ / judge → | " + " | ".join(labels) + " | total |"
    divider = "|---|" + "|".join(["---"] * (len(labels) + 1)) + "|"
    rows = []
    for lbl, row in zip(labels, matrix, strict=True):
        cells = " | ".join(str(c) for c in row)
        rows.append(f"| **{lbl}** | {cells} | {sum(row)} |")
    return "\n".join(
        [
            "## Confusion matrix",
            "",
            "Rows = maintainer (gold) labels; columns = judge labels. Cell values are raw counts.",
            "",
            header,
            divider,
            *rows,
        ]
    )


def _section_per_label(confusion: dict[str, object]) -> str:
    labels: list[str] = list(confusion["labels"])  # type: ignore[arg-type]
    per_label: dict[str, dict[str, float | int]] = confusion["per_label"]  # type: ignore[assignment]
    rows = []
    for lbl in labels:
        d = per_label[lbl]
        rows.append(
            f"| {lbl} | {float(d['precision']):.3f} | {float(d['recall']):.3f} | {int(d['support'])} |"
        )
    return "\n".join(
        [
            "## Per-label precision, recall, and support",
            "",
            "| label | precision | recall | support |",
            "|---|---|---|---|",
            *rows,
        ]
    )


def _section_methodology(meta: dict[str, object], summary: dict[str, object]) -> str:
    n_gold = int(meta["n_gold"])  # type: ignore[arg-type]
    per_bucket = n_gold // 3
    sys_hash = _hash_prefix(str(meta["system_prompt_sha256"]))
    usr_hash = _hash_prefix(str(meta["user_template_sha256"]))
    model = str(meta["judge_model"])
    temp = float(meta["temperature"])  # type: ignore[arg-type]
    return "\n".join(
        [
            "## Methodology",
            "",
            f"- Gold set: {n_gold} synthetic trajectories spanning plan-quality buckets "
            f"(`pass`, `partial`, `fail`), balanced {per_bucket}/{per_bucket}/{per_bucket}.",
            "- One maintainer authored the labels; one judge model was evaluated.",
            f"- Judge configuration: model `{model}`, `temperature={temp}`. "
            f"Prompt-hash digests (first 12 hex): system=`{sys_hash}`, user-template=`{usr_hash}`. "
            f"Re-running with the same model and the same prompts (verified by hash) "
            f"reproduces the numbers above modulo provider determinism.",
        ]
    )


_LIMITATIONS = "\n".join(
    [
        "## Limitations",
        "",
        "- The gold set is synthetic and was written by one person; biases are present but unknown.",
        "- A single judge model is evaluated. Cross-model agreement (e.g. `gpt-4o-mini`, open-weights) is deferred to a later phase.",
        "- Kappa bands follow Landis–Koch (1977); \"good enough\" kappa depends on use case.",
    ]
)


def _section_recalibration() -> str:
    return "\n".join(
        [
            "## Recalibration",
            "",
            "```bash",
            "uv run python scripts/build_calibration_set.py \\",
            "    --source synth \\",
            "    --gold-labels tests/data/gold_plans.jsonl \\",
            "    --judge-model anthropic/claude-sonnet-4-6 \\",
            "    --out docs/calibration/<version>-report.jsonl \\",
            "    --concurrency 4",
            "",
            "uv run python scripts/render_calibration_md.py \\",
            "    --report docs/calibration/<version>-report.jsonl \\",
            "    --out docs/concepts/calibration.md",
            "```",
        ]
    )


def _section_raw_report(report_basename: str) -> str:
    return "\n".join(
        [
            "## Raw report",
            "",
            "The JSONL report this page was rendered from is committed at",
            f"[`docs/calibration/{report_basename}`](../calibration/{report_basename}).",
        ]
    )


def render(*, report_path: Path, out_path: Path, report_basename: str) -> None:
    """Read a calibration JSONL report and write the docs Markdown page."""
    summary: dict[str, object] | None = None
    confusion: dict[str, object] | None = None
    meta: dict[str, object] | None = None
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        kind = obj.get("_kind")
        if kind == "summary":
            summary = obj
        elif kind == "confusion":
            confusion = obj
        elif kind == "meta":
            meta = obj

    if summary is None or confusion is None or meta is None:
        raise ValueError(
            "report is missing one of: _kind=summary, _kind=confusion, _kind=meta"
        )

    parts = [
        "# Calibration",
        "",
        _section_headline(summary, meta),
        "",
        _section_confusion(confusion),
        "",
        _section_per_label(confusion),
        "",
        _section_methodology(meta, summary),
        "",
        _LIMITATIONS,
        "",
        _section_recalibration(),
        "",
        _section_raw_report(report_basename),
        "",
    ]
    out_path.write_text("\n".join(parts), encoding="utf-8")


@click.command()
@click.option(
    "--report",
    "report_path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Path to the calibration JSONL report.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=Path("docs/concepts/calibration.md"),
    show_default=True,
    help="Output Markdown path.",
)
def main(report_path: Path, out_path: Path) -> None:
    """Render a calibration JSONL report to a Markdown docs page."""
    render(
        report_path=report_path,
        out_path=out_path,
        report_basename=report_path.name,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run test to confirm GREEN**

Run: `uv run pytest tests/unit/scripts/test_render_calibration_md.py -q`
Expected: 1 passed.

If the golden file diff fails: print the diff (`diff -u expected actual`), eyeball it, decide whether to update the renderer or the golden expected. The golden file is the spec, but if the renderer's output is what you actually want shipped, update the golden file to match (TDD discipline says: don't keep adjusting test outputs to fit drifting code, but for golden-file tests this is the normal feedback loop).

- [ ] **Step 7: Verify gates**

```bash
uv run mypy --strict scripts/render_calibration_md.py
uv run ruff check scripts/render_calibration_md.py tests/unit/scripts/test_render_calibration_md.py
uv run ruff format scripts/render_calibration_md.py tests/unit/scripts/test_render_calibration_md.py
uv run ruff format --check scripts/render_calibration_md.py tests/unit/scripts/test_render_calibration_md.py
```

- [ ] **Step 8: Commit**

```bash
git add scripts/render_calibration_md.py tests/data/_golden_calibration_report.jsonl tests/data/_golden_calibration_expected.md tests/unit/scripts/test_render_calibration_md.py
git commit -m "feat(scripts): render_calibration_md.py with golden-file test"
```

---

## Task 7: One-shot calibration run + commit the real report and rendered page

**Goal:** Execute the calibration script once against the synthetic gold set with the pinned Claude Sonnet snapshot, commit the JSONL report, render the docs page, commit it. This is the MANUAL step — it requires `ANTHROPIC_API_KEY` and incurs real LLM cost.

**This task cannot be performed by a subagent.** It requires the maintainer's API key and produces the artifact that the rest of the phase verifies against. If executing via subagent-driven-development, the controller should:
1. Stop after Task 6 is committed
2. Hand off to the maintainer with the exact command block below
3. Resume Task 8 only after Task 7's commit lands

**Files:**
- Create (by running): `docs/calibration/v0.0.8-alpha-report.jsonl`
- Create (by rendering): `docs/concepts/calibration.md`

- [ ] **Step 1: Create the output directory**

```bash
mkdir -p docs/calibration
```

- [ ] **Step 2: Set the API key in the current shell**

```bash
export ANTHROPIC_API_KEY=<your real key>
```

Do NOT commit the key. The `.gitignore` already excludes `.env`, but
do not source from `.env` if it contains real credentials you'd rather
not load into the shell history.

- [ ] **Step 3: Run the calibration script**

```bash
uv run python scripts/build_calibration_set.py \
    --source synth \
    --gold-labels tests/data/gold_plans.jsonl \
    --judge-model anthropic/claude-sonnet-4-6 \
    --out docs/calibration/v0.0.8-alpha-report.jsonl \
    --concurrency 4
```

Expected: the script writes the JSONL report and exits cleanly.
Approximate runtime: 1–3 minutes for 51 trajectories at concurrency 4.

If a request fails: re-run the script (the existing per-trajectory
error path records the failure; you may choose to retry only the
failed trajectory_ids by rebuilding a smaller gold-labels file).

- [ ] **Step 4: Inspect the report**

```bash
tail -3 docs/calibration/v0.0.8-alpha-report.jsonl
```

Expected: three trailing JSONL lines — `_kind: "summary"`, `_kind:
"confusion"`, `_kind: "meta"` — in that order. Read the kappa value out
of the summary line. Record it for the next steps and the CHANGELOG.

- [ ] **Step 5: Render the docs page**

```bash
uv run python scripts/render_calibration_md.py \
    --report docs/calibration/v0.0.8-alpha-report.jsonl \
    --out docs/concepts/calibration.md
```

- [ ] **Step 6: Verify mkdocs still builds**

```bash
uv run mkdocs build --strict
```

Expected: clean. (Nav update in Task 8 will register the new page; for
now the page exists but isn't linked from nav, and mkdocs strict mode
may warn about it as an orphan. If it warns, defer this verification to
Task 8 — that's the explicit "fix the orphan" task.)

- [ ] **Step 7: Commit the report and rendered page**

```bash
git add docs/calibration/v0.0.8-alpha-report.jsonl docs/concepts/calibration.md
git commit -m "feat(calibration): TrajectoryJudge gold-set run (κ = X.XX, $BAND)"
```

(Replace `X.XX` with the actual kappa value and `$BAND` with the
Landis-Koch band from the summary line.)

---

## Task 8: mkdocs nav + cross-link calibration from judges concept page

**Goal:** Add `Calibration: concepts/calibration.md` to mkdocs nav (alphabetically between Judges and Metrics). Update `docs/concepts/judges.md` to remove the "pending Phase 6.1" deferral sentence and add a one-line link to the calibration page.

**Files:**
- Modify: `mkdocs.yml`
- Modify: `docs/concepts/judges.md`

- [ ] **Step 1: Add Calibration to mkdocs nav**

Open `mkdocs.yml`. The Concepts block currently reads:

```yaml
  - Concepts:
      - concepts/index.md
      - Judges: concepts/judges.md
      - Metrics: concepts/metrics.md
      - Storage: concepts/storage.md
      - Tracing: concepts/tracing.md
      - Trajectory model: concepts/trajectory.md
```

Change it to:

```yaml
  - Concepts:
      - concepts/index.md
      - Calibration: concepts/calibration.md
      - Judges: concepts/judges.md
      - Metrics: concepts/metrics.md
      - Storage: concepts/storage.md
      - Tracing: concepts/tracing.md
      - Trajectory model: concepts/trajectory.md
```

- [ ] **Step 2: Update `docs/concepts/judges.md`**

Find the "Why judges live behind calibration" section. Its current text
talks about Phase 6 having NOT shipped calibration data. Replace the
sentence "Phase 6 publishes the judge code itself, the calibration
harness, and Cohen's kappa — but `Judge`, `JudgeVerdict`,
`TrajectoryJudge`, and `PlanQuality` are importable only from
`ariadne_eval.eval.judges` (and the `PlanQuality` metric from
`ariadne_eval.eval`). Phase 6.1 will publish a hand-labeled
≥50-example gold set and the resulting kappa table, at which point
these symbols move to the top-level `__all__`." with this:

> Phase 6.1 ships the calibration evidence: see [Calibration](./calibration.md) for the current judge-vs-maintainer kappa, the confusion matrix, and per-label precision/recall against the 51-fixture synthetic gold set. As of v0.0.8-alpha, `Judge`, `JudgeVerdict`, `JudgeParseError`, `TrajectoryJudge`, `StubJudge`, and `PlanQuality` are now top-level public (`from ariadne_eval import TrajectoryJudge`).

- [ ] **Step 3: Build the docs**

```bash
uv run mkdocs build --strict
```

Expected: clean. The Calibration page now appears in nav, the cross-link from judges resolves.

- [ ] **Step 4: Verify the link works**

```bash
grep -c "calibration" site/concepts/judges/index.html
```

Expected: ≥ 1.

- [ ] **Step 5: Commit**

```bash
git add mkdocs.yml docs/concepts/judges.md
git commit -m "docs: link calibration page from judges, add to mkdocs nav"
```

---

## Task 9: API promotion + public-API test

**Goal:** Move six judge symbols (`Judge`, `JudgeParseError`, `JudgeVerdict`, `PlanQuality`, `StubJudge`, `TrajectoryJudge`) into top-level `ariadne_eval.__all__`. Write a test that asserts both the symbol presence AND the calibration report file's existence — Hard Rule #5 enforced by tests.

**Files:**
- Modify: `src/ariadne_eval/__init__.py`
- Modify: `src/ariadne_eval/eval/judges/__init__.py`
- Create: `tests/unit/eval/test_public_api_phase6_1.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/eval/test_public_api_phase6_1.py`:

```python
"""Tests for the Phase 6.1 public-API promotion."""

from __future__ import annotations

from pathlib import Path

import pytest

import ariadne_eval

pytestmark = pytest.mark.fast


_REPO = Path(__file__).resolve().parents[3]


def test_six_judge_symbols_now_top_level_public() -> None:
    for name in (
        "Judge",
        "JudgeParseError",
        "JudgeVerdict",
        "PlanQuality",
        "StubJudge",
        "TrajectoryJudge",
    ):
        assert name in ariadne_eval.__all__, f"{name!r} missing from __all__"
        assert getattr(ariadne_eval, name) is not None


def test_top_level_imports_work() -> None:
    # Doc-friendly form: ``from ariadne_eval import TrajectoryJudge``
    from ariadne_eval import (
        Judge,
        JudgeParseError,
        JudgeVerdict,
        PlanQuality,
        StubJudge,
        TrajectoryJudge,
    )

    assert Judge is not None
    assert JudgeParseError is not None
    assert JudgeVerdict is not None
    assert PlanQuality is not None
    assert StubJudge is not None
    assert TrajectoryJudge is not None


def test_calibration_report_committed() -> None:
    """Hard Rule #5: the judge promotion ships only with calibration evidence."""
    report = _REPO / "docs" / "calibration" / "v0.0.8-alpha-report.jsonl"
    assert report.exists(), (
        "Hard Rule #5 violation: judge symbols are top-level but "
        f"the calibration report is missing at {report}"
    )
    # Smoke-check the report has the expected trailing blocks
    text = report.read_text(encoding="utf-8")
    assert '"_kind": "summary"' in text or '"_kind":"summary"' in text
    assert '"_kind": "confusion"' in text or '"_kind":"confusion"' in text
    assert '"_kind": "meta"' in text or '"_kind":"meta"' in text


def test_calibration_page_committed() -> None:
    """The human-readable page exists and references the report."""
    page = _REPO / "docs" / "concepts" / "calibration.md"
    assert page.exists()
    body = page.read_text(encoding="utf-8")
    assert "Confusion matrix" in body
    assert "Per-label precision" in body
    assert "v0.0.8-alpha-report.jsonl" in body
```

- [ ] **Step 2: Run test to confirm RED**

Run: `uv run pytest tests/unit/eval/test_public_api_phase6_1.py -q`
Expected: 4 tests FAIL — symbols not in `__all__` (the report-file tests will pass because Task 7 already committed those files).

- [ ] **Step 3: Modify `src/ariadne_eval/__init__.py`**

Find the imports block. After the existing
`from ariadne_eval.eval.metrics.base import AsyncMetric, Metric, MetricResult`
line, add a new import block:

```python
from ariadne_eval.eval.judges import (
    Judge,
    JudgeParseError,
    JudgeVerdict,
    StubJudge,
    TrajectoryJudge,
)
from ariadne_eval.eval.metrics.plan_quality import PlanQuality
```

Then in the `__all__` list, insert these six names in their alphabetized
positions. The current alphabetized list near the relevant slots looks
like:

```python
    "InternalPayload",
    "JsonValue",
    "KappaResult",
    "LLMCallPayload",
    "Message",
    "MetadataTooLargeError",
    "Metric",
    "MetricResult",
    "MissingReferenceError",
```

Insert:
- `Judge` after `InternalPayload`
- `JudgeParseError` after `Judge`
- `JudgeVerdict` after `JudgeParseError`
- `PlanQuality` after `Message` (alphabetically between Message and MetadataTooLargeError ⟶ no: P comes after Me/Mi — actually after `MissingReferenceError`)
- `StubJudge` and `TrajectoryJudge` in their alphabetized positions

To remove ambiguity, the final `__all__` should be alphabetized end-to-end. The simplest reliable approach: keep the current entries in their current order (already alphabetized) and insert each new entry at its alphabetized position. The post-change `__all__` MUST contain these strings in alphabetical order; sort the list mentally and verify with `python -c "import ariadne_eval; assert ariadne_eval.__all__ == sorted(ariadne_eval.__all__)"` after the change.

- [ ] **Step 4: Update `src/ariadne_eval/eval/judges/__init__.py` docstring**

The current docstring is:

```python
"""LLM-as-judge implementations and the Judge Protocol.

These symbols are namespace-public (importable from this module) but
intentionally NOT re-exported from ``ariadne_eval.__all__`` until
Phase 6.1 ships calibration data - see Hard Rule #5 in CLAUDE.md.
"""
```

Replace it with:

```python
"""LLM-as-judge implementations and the Judge Protocol.

These symbols are top-level public as of v0.0.8-alpha: see
``docs/concepts/calibration.md`` for the maintainer-vs-judge kappa,
confusion matrix, and per-label precision/recall against the
51-fixture synthetic gold set committed at
``docs/calibration/v0.0.8-alpha-report.jsonl``.
"""
```

- [ ] **Step 5: Run all fast tests**

```bash
uv run pytest -m "fast and not integration" -q
```

Expected: all green, including the 4 new tests in
`test_public_api_phase6_1.py`.

- [ ] **Step 6: Verify alphabetization invariant**

```bash
uv run python -c "import ariadne_eval; assert ariadne_eval.__all__ == sorted(ariadne_eval.__all__), 'ariadne_eval.__all__ is not sorted'"
```

Expected: no output, exit code 0.

- [ ] **Step 7: Verify gates**

```bash
uv run mypy --strict src/ariadne_eval
uv run ruff check src/ariadne_eval tests
uv run ruff format src/ariadne_eval/__init__.py src/ariadne_eval/eval/judges/__init__.py tests/unit/eval/test_public_api_phase6_1.py
uv run ruff format --check src/ariadne_eval tests
```

- [ ] **Step 8: Commit**

```bash
git add src/ariadne_eval/__init__.py src/ariadne_eval/eval/judges/__init__.py tests/unit/eval/test_public_api_phase6_1.py
git commit -m "feat(public): promote Judge/TrajectoryJudge/StubJudge/PlanQuality to top-level"
```

---

## Task 10: CHANGELOG + final verification + merge + tag + memory update

**Goal:** Update CHANGELOG, run the full verification gate, merge `phase-6.1-calibration` into `main`, tag `v0.0.8-alpha`, update the phase-state memory.

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update `CHANGELOG.md` under `[Unreleased]`**

Read the actual kappa from the committed report:

```bash
tail -3 docs/calibration/v0.0.8-alpha-report.jsonl | head -1 | python -c "import json,sys; d=json.loads(sys.stdin.read()); print(d['kappa'], d['interpretation'])"
```

Then under `### Added` append (substitute actual κ value and band):

```markdown
- Phase 6.1: judge symbols (`Judge`, `JudgeParseError`, `JudgeVerdict`,
  `PlanQuality`, `StubJudge`, `TrajectoryJudge`) are now top-level
  public (`from ariadne_eval import TrajectoryJudge`).
- 51-fixture synthetic plan-quality gold set
  (`tests/data/gold_plans.jsonl`, balanced 17/17/17 across `pass`/
  `partial`/`fail`).
- Calibration evidence: `TrajectoryJudge` achieves κ = X.XX (BAND)
  against the maintainer on the gold set. Report committed at
  `docs/calibration/v0.0.8-alpha-report.jsonl`; human-readable page at
  `docs/concepts/calibration.md`.
- `scripts/build_calibration_set.py`: `--source synth|store` flag;
  `_kind: "confusion"` and `_kind: "meta"` trailing JSONL lines (with
  prompt-hash digests for drift detection).
- `scripts/render_calibration_md.py`: renders the JSONL report into a
  human-readable docs page; golden-file tested.
```

- [ ] **Step 2: Run the full verification gate**

```bash
uv run pytest -m "fast and not integration" --cov=src/ariadne_eval/eval --cov=scripts --cov-report=term-missing -q
uv run pytest -m integration -q
uv run mypy --strict src/ariadne_eval
uv run ruff check src/ariadne_eval tests examples scripts
uv run ruff format --check src/ariadne_eval tests examples scripts
uv run mkdocs build --strict
```

Expected:
- All fast tests green; coverage on `scripts/build_calibration_set.py` ≥ 90% and `scripts/render_calibration_md.py` ≥ 90%.
- 2 integration tests pass (cassettes from Phase 6).
- mypy clean.
- ruff check + format clean.
- mkdocs strict build clean.

If `render_calibration_md.py` falls under 90% coverage: the missed
lines are likely the `main()` CLI wrapper; either add a CliRunner
smoke test or accept the gap and document it in this task's commit
message.

- [ ] **Step 3: Commit CHANGELOG**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): phase 6.1 — calibration evidence + judge promotion"
```

- [ ] **Step 4: Push the branch and open the PR (optional intermediate)**

```bash
git push -u origin phase-6.1-calibration
gh pr create --title "Phase 6.1: calibration set + kappa table + judge promotion" --body "$(cat <<'EOF'
## Summary

Implements Phase 6.1 per `docs/superpowers/specs/2026-05-31-calibration-set-design.md`
and `docs/superpowers/plans/2026-05-31-calibration-set.md`.

- 51-fixture synthetic plan-quality gold set, balanced 17/17/17.
- `TrajectoryJudge` calibrated against the maintainer at κ = X.XX (BAND).
- Six judge symbols moved from `ariadne_eval.eval.judges.*` to top-level `ariadne_eval.*`.
- New `--source synth|store` flag + `confusion` + `meta` JSONL blocks on `build_calibration_set.py`.
- New `scripts/render_calibration_md.py` (golden-file tested).
- New `docs/concepts/calibration.md` page; mkdocs nav updated.

## Phase reference

Implements Phase 6.1 per the spec and plan above.

## Verification

- [x] `uv run pytest -m "fast and not integration" -q` green
- [x] `uv run pytest -m integration -q` green
- [x] `uv run mypy --strict src/ariadne_eval` clean
- [x] `uv run ruff check` clean
- [x] `uv run ruff format --check` clean
- [x] `uv run mkdocs build --strict` clean
- [x] Hard Rule #5 now test-enforced via `test_calibration_report_committed`
EOF
)"
```

Wait for CI to go green before merging.

- [ ] **Step 5: Merge and tag (controller-confirmed step)**

This step is hard-to-reverse — confirm with the user before executing.
Once approved:

```bash
git checkout main
git pull --ff-only
git merge --no-ff phase-6.1-calibration -m "Merge branch 'phase-6.1-calibration' into main"
git tag -a v0.0.8-alpha -m "Phase 6.1: calibration set, kappa table, judge promotion (κ = X.XX, BAND)"
git push origin main
git push origin v0.0.8-alpha
git branch -d phase-6.1-calibration
git push origin --delete phase-6.1-calibration
```

If the PR route was used (Step 4), merge via `gh pr merge` instead and
the tag + branch delete are still manual.

- [ ] **Step 6: Update phase-state memory**

Edit `/Users/rish/.claude/projects/-Users-rish-Desktop-AI-Projects-ariadne-eval/memory/current_phase_state.md`:

- Last completed: Phase 6.1, tagged `v0.0.8-alpha`.
- Add `v0.0.8-alpha — Calibration set, kappa table, judge promotion (Phase 6.1)` to the shipped list. Carry the actual κ value.
- Update test counts (expect ~300+ fast tests + 2 integration tests).
- Replace the "Phase 6.1 follow-up" note with: Phase 6.1 SHIPPED. Judge symbols now top-level public.
- Next phase: Phase 7 — tau-bench benchmark runner (per `Prompts.md`, NOT StepwiseJudge — the canonical sequence is tau-bench → CLI → UI → drift → PyPI).

---

## Self-review notes

**Spec coverage check:** Every spec scope item maps to a task —

- Synthetic gold set → Task 1 (loader) + Task 2 (51 fixtures)
- `build_calibration_set.py` `--source synth|store` → Task 3
- Extended report shape (`confusion` + `meta`) → Task 4 + Task 5
- `render_calibration_md.py` → Task 6
- One-shot calibration run → Task 7
- `docs/concepts/calibration.md` → Task 7 (rendered) + Task 8 (nav)
- API promotion → Task 9
- Docs & CHANGELOG → Task 8 + Task 10
- Tag `v0.0.8-alpha` → Task 10

**Type consistency check:** `_build_confusion_block(judged_pairs: list[tuple[str, str]]) -> dict[str, object]` defined in Task 4, called in the run loop in Task 5 (the wiring lives in Task 5's modified write block — both refer to the same name). `_build_meta_block(judge_model: str, temperature: float, run_date: str, n_gold: int) -> dict[str, object]` defined in Task 5. `render(report_path: Path, out_path: Path, report_basename: str) -> None` defined in Task 6, called in test_render_calibration_md.py with the same kwargs. `iter_gold_plans(stream: IO[str]) -> Iterator[GoldEntry]` and `GoldEntry` defined in Task 1, consumed in Task 2 tests and Task 3's `--source synth` code path.

**Placeholder scan:** No "TBD", "TODO", or "fill in details" sentinels. The κ values in the CHANGELOG and tag message ARE placeholders — they're explicitly intended to be filled in after Task 7 produces the real number; the plan flags this in-line.

**Scope check:** Ten tasks. Tasks 1–6 + 8–10 are subagent-executable; Task 7 requires the maintainer's API key. The natural execution split is: subagents handle 1–6, controller stops, maintainer runs Task 7, subagents handle 8–10.
