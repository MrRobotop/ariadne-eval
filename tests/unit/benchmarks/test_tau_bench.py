"""Tests for TauBenchAdapter (lazy import + stubbed tau_bench)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ariadne_eval.benchmarks.tau_bench import TauBenchAdapter

pytestmark = pytest.mark.fast


def test_actionable_import_error_when_tau_bench_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the [tau-bench] extra, tasks() raises a clear ImportError."""
    # Ensure tau_bench is NOT importable
    monkeypatch.setitem(sys.modules, "tau_bench", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "tau_bench.envs", None)  # type: ignore[arg-type]

    adapter = TauBenchAdapter(env_name="retail")
    with pytest.raises(ImportError, match=r"\[tau-bench\] extra"):
        adapter.tasks(limit=1)


def test_tasks_returns_benchmark_tasks_with_stubbed_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With tau_bench.envs.get_env stubbed, tasks() yields BenchmarkTask records."""
    fake_task_a = {"task_id": "retail-0", "instruction": "Cancel order 42."}
    fake_task_b = {"task_id": "retail-1", "instruction": "Look up SKU 1."}
    fake_env = MagicMock()
    fake_env.tasks = [fake_task_a, fake_task_b]

    fake_envs_mod = MagicMock()
    fake_envs_mod.get_env = MagicMock(return_value=fake_env)
    fake_tau_bench_mod = MagicMock()

    monkeypatch.setitem(sys.modules, "tau_bench", fake_tau_bench_mod)
    monkeypatch.setitem(sys.modules, "tau_bench.envs", fake_envs_mod)

    adapter = TauBenchAdapter(env_name="retail")
    tasks = list(adapter.tasks(limit=2))
    assert len(tasks) == 2
    assert tasks[0].task_id == "retail-0"
    assert tasks[0].task_index == 0
    assert tasks[0].instruction == "Cancel order 42."
    assert tasks[0].payload is fake_task_a


def test_name_property() -> None:
    assert TauBenchAdapter(env_name="retail").name == "tau-retail"
    assert TauBenchAdapter(env_name="airline").name == "tau-airline"


async def test_run_task_calls_agent_solve_and_persists_trajectory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run_task glues tau-bench's agent.solve to our store via _convert_tau_traj."""
    from ariadne_eval.storage.duckdb_store import DuckDBStore

    # Stub env with a task list
    fake_env = MagicMock()
    fake_env.tasks = [{"task_id": "retail-0", "instruction": "do thing"}]
    fake_env.tools_info = []
    fake_env.wiki = ""

    # Stub the agent class — solve() returns a dict-shaped EnvRunResult
    fake_env_result = {
        "task_id": "retail-0",
        "reward": 1.0,
        "info": {},
        "traj": [
            {"role": "user", "content": "do thing"},
            {"role": "assistant", "content": "done."},
        ],
    }
    fake_agent_instance = MagicMock()
    fake_agent_instance.solve = MagicMock(return_value=fake_env_result)

    fake_tool_calling_module = MagicMock()
    fake_tool_calling_module.ToolCallingAgent = MagicMock(return_value=fake_agent_instance)
    fake_agents_mod = MagicMock()
    fake_agents_mod.tool_calling_agent = fake_tool_calling_module

    fake_envs_mod = MagicMock()
    fake_envs_mod.get_env = MagicMock(return_value=fake_env)
    fake_tau_bench_mod = MagicMock()

    monkeypatch.setitem(sys.modules, "tau_bench", fake_tau_bench_mod)
    monkeypatch.setitem(sys.modules, "tau_bench.envs", fake_envs_mod)
    monkeypatch.setitem(sys.modules, "tau_bench.agents", fake_agents_mod)

    store = DuckDBStore(path=tmp_path / "test.duckdb")
    adapter = TauBenchAdapter(env_name="retail")
    tasks = adapter.tasks(limit=1)

    result = await adapter.run_task(
        tasks[0],
        model="haiku-stub",
        provider="anthropic",
        store=store,
        seed=42,
    )
    assert result.success is True
    assert result.raw_score == 1.0
    # Trajectory was persisted
    traj, steps = await store.get_trajectory(result.trajectory_id)
    assert traj.task == "do thing"
    assert traj.final_answer == "done."
    assert len(steps) >= 2  # user + assistant
    await store.close()
