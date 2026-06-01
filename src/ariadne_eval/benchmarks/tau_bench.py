"""tau-bench adapter and trajectory converter.

The adapter class lazy-imports the ``tau_bench`` package; users who
don't install the ``[tau-bench]`` extra never trigger the import. The
converter below is pure-Python and does not depend on the tau_bench
package — it accepts the EnvRunResult shape as a dict so it can be
exercised from unit tests without the extra installed.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast

from ariadne_eval.benchmarks.base import BenchmarkRunResult, BenchmarkTask
from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import StepStatus, TrajectoryStatus
from ariadne_eval.core.trajectory import (
    JsonValue,
    LLMCallPayload,
    Message,
    Step,
    ToolCallPayload,
    Trajectory,
    UserInputPayload,
)
from ariadne_eval.storage.base import Store

__all__ = ["TauBenchAdapter", "_convert_tau_traj"]


_SUCCESS_THRESHOLD = 1.0 - 1e-6


def _convert_tau_traj(
    env_result: dict[str, object],
    *,
    instruction: str,
    model_id: str,
    agent_name: str,
    agent_version: str,
) -> tuple[Trajectory, list[Step]]:
    """Convert a tau-bench ``EnvRunResult``-shaped dict to ``(Trajectory, list[Step])``.

    Accepts a plain dict with keys ``task_id``, ``reward``, ``info``,
    ``traj`` (a list of message dicts). This lets the converter be
    tested without installing the ``tau_bench`` package.
    """
    traj_id = new_id()
    started = datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC)

    reward = float(cast(float, env_result["reward"]))
    task_id = str(env_result["task_id"])
    raw_messages = cast(list[dict[str, Any]], env_result["traj"])

    steps: list[Step] = []
    final_answer: str | None = None
    # tool_call_id → the child Step (so we can fill result once the tool-result message arrives)
    pending_tool_payloads: dict[str, Step] = {}
    msg_index = 0

    for raw_msg in raw_messages:
        role = raw_msg.get("role")
        content = raw_msg.get("content", "") or ""
        step_started = started + timedelta(seconds=msg_index)
        step_finished = step_started + timedelta(milliseconds=10)

        if role == "user":
            step = Step(
                id=new_id(),
                trajectory_id=traj_id,
                parent_step_id=None,
                name="user_input",
                started_at=step_started,
                finished_at=step_finished,
                status=StepStatus.SUCCEEDED,
                payload=UserInputPayload(message=str(content)),
            )
            steps.append(step)

        elif role == "assistant":
            tool_calls = raw_msg.get("tool_calls") or []
            llm_step = Step(
                id=new_id(),
                trajectory_id=traj_id,
                parent_step_id=None,
                name="llm_call",
                started_at=step_started,
                finished_at=step_finished,
                status=StepStatus.SUCCEEDED,
                payload=LLMCallPayload(
                    model_id=model_id,
                    prompt_messages=[Message(role="user", content=instruction)],
                    completion=str(content),
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=10.0,
                    cost_usd=0.0,
                ),
            )
            steps.append(llm_step)

            if not tool_calls:
                # Final-answer assistant turn (no tool calls)
                if content:
                    final_answer = str(content)
            else:
                # Emit one child ToolCall step per tool_call
                for tool_call in tool_calls:
                    fn = tool_call.get("function", {})
                    tool_name = str(fn.get("name", "unknown"))
                    arguments_raw = fn.get("arguments", "{}")
                    try:
                        arguments: dict[str, Any] = (
                            json.loads(arguments_raw)
                            if isinstance(arguments_raw, str)
                            else dict(arguments_raw)
                        )
                    except (json.JSONDecodeError, TypeError):
                        arguments = {"_raw": str(arguments_raw)}
                    tool_step = Step(
                        id=new_id(),
                        trajectory_id=traj_id,
                        parent_step_id=llm_step.id,
                        name=f"tool_{tool_name}",
                        started_at=step_started + timedelta(milliseconds=1),
                        finished_at=step_finished,
                        status=StepStatus.SUCCEEDED,
                        payload=ToolCallPayload(
                            tool_name=tool_name,
                            arguments=cast("dict[str, JsonValue]", arguments),
                            result=None,
                            latency_ms=10.0,
                        ),
                    )
                    steps.append(tool_step)
                    pending_tool_payloads[str(tool_call.get("id", ""))] = tool_step

        elif role == "tool":
            tool_call_id = str(raw_msg.get("tool_call_id", ""))
            if tool_call_id in pending_tool_payloads:
                child_step = pending_tool_payloads.pop(tool_call_id)
                # Use model_copy(update=...) rather than direct mutation to re-run field validators.
                new_payload = cast(ToolCallPayload, child_step.payload).model_copy(
                    update={"result": str(content)}
                )
                # Replace the step in `steps` with one carrying the updated payload
                idx = steps.index(child_step)
                steps[idx] = child_step.model_copy(update={"payload": new_payload})

        # All other roles (system, etc.) are ignored
        msg_index += 1

    final_status = (
        TrajectoryStatus.SUCCEEDED if reward >= _SUCCESS_THRESHOLD else TrajectoryStatus.FAILED
    )

    trajectory = Trajectory(
        id=traj_id,
        task=instruction,
        agent_name=agent_name,
        agent_version=agent_version,
        model_id=model_id,
        started_at=started,
        finished_at=started + timedelta(seconds=max(msg_index, 1)),
        final_status=final_status,
        final_answer=final_answer,
        metadata={
            "tau_bench_reward": reward,
            "tau_bench_task_id": task_id,
        },
    )

    return trajectory, steps


# ---------------------------------------------------------------------------
# Lazy-import helpers
# ---------------------------------------------------------------------------

_EXTRA_MSG = (
    "tau_bench is not installed. Install ariadne-eval with the [tau-bench] "
    "extra: pip install 'ariadne-eval[tau-bench]'"
)


def _import_tau_bench_envs() -> Any:
    """Lazy import of tau_bench.envs with an actionable error message."""
    try:
        return importlib.import_module("tau_bench.envs")
    except ImportError as exc:
        raise ImportError(_EXTRA_MSG) from exc


def _build_tau_bench_agent(*, kind: str, model: str, provider: str, env: Any) -> Any:
    """Construct the tau-bench agent class corresponding to ``kind``."""
    if kind == "tool-calling":
        agent_mod = importlib.import_module("tau_bench.agents.tool_calling_agent")
        return agent_mod.ToolCallingAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=model,
            provider=provider,
            temperature=0.0,
        )
    elif kind == "react":
        agent_mod = importlib.import_module("tau_bench.agents.chat_react_agent")
        return agent_mod.ChatReActAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=model,
            provider=provider,
            temperature=0.0,
        )
    elif kind == "few-shot-tool-calling":
        agent_mod = importlib.import_module("tau_bench.agents.few_shot_agent")
        return agent_mod.FewShotToolCallingAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=model,
            provider=provider,
            temperature=0.0,
        )
    else:
        raise ValueError(f"unknown agent_kind: {kind!r}")


def _as_dict(obj: Any) -> dict[str, Any]:
    """Coerce a tau-bench EnvRunResult (or similar) into a plain dict.

    tau-bench's types may be frozen dataclasses, Pydantic models, or
    namedtuples; this converter walks the common attributes our
    converter needs (``task_id``, ``reward``, ``info``, ``traj``).
    """
    if hasattr(obj, "model_dump"):
        return cast("dict[str, Any]", obj.model_dump())
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    if hasattr(obj, "_asdict"):
        return cast("dict[str, Any]", obj._asdict())
    raise TypeError(f"cannot coerce {type(obj).__name__} to dict")


# ---------------------------------------------------------------------------
# TauBenchAdapter
# ---------------------------------------------------------------------------


class TauBenchAdapter:
    """Concrete Benchmark over tau-bench's retail or airline domains.

    Lazy-imports tau_bench; the [tau-bench] extra must be installed for
    ``tasks()`` and ``run_task()`` to work.
    """

    name: str

    # Pinned in pyproject.toml's [tau-bench] extra; recorded in trajectory.agent_version.
    _TAU_BENCH_COMMIT = "59a200c6d575d595120f1cb70fea53cef0632f6b"

    def __init__(
        self,
        env_name: Literal["retail", "airline"],
        *,
        user_model: str = "groq/llama-3.3-70b-versatile",
        user_strategy: str = "llm",
        agent_kind: Literal["tool-calling", "react", "few-shot-tool-calling"] = "tool-calling",
    ) -> None:
        """Configure the adapter.

        ``user_model`` is the LLM that drives the simulated user inside
        tau-bench (a tau-bench native concept; defaults to Groq Llama
        because tau-bench's simulated user is the largest line-item
        across all cells).
        """
        self._env_name = env_name
        self._user_model = user_model
        self._user_strategy = user_strategy
        self._agent_kind = agent_kind
        self.name = f"tau-{env_name}"
        self._last_split: str = "test"

    def tasks(
        self,
        *,
        split: str = "test",
        limit: int | None = None,
    ) -> Sequence[BenchmarkTask]:
        """Return tau-bench's task list as ``BenchmarkTask`` records."""
        self._last_split = split
        envs = _import_tau_bench_envs()
        # Always pass task_index=0 to dodge tau-bench's off-by-one bug:
        # base.py uses ``random.randint(0, len(tasks))`` (inclusive on both
        # ends) when task_index is None, which can index past the list.
        env = envs.get_env(
            self._env_name,
            user_strategy=self._user_strategy,
            user_model=self._user_model,
            user_provider=self._user_model.split("/", 1)[0],
            task_split=split,
            task_index=0,
        )
        raw_tasks: list[Any] = list(env.tasks)
        if limit is not None:
            raw_tasks = raw_tasks[:limit]
        return [
            BenchmarkTask(
                task_id=str(raw.get("task_id", f"{self._env_name}-{i}"))
                if isinstance(raw, dict)
                else str(getattr(raw, "task_id", f"{self._env_name}-{i}")),
                task_index=i,
                instruction=str(raw.get("instruction", ""))
                if isinstance(raw, dict)
                else str(getattr(raw, "instruction", "")),
                payload=raw,
            )
            for i, raw in enumerate(raw_tasks)
        ]

    async def run_task(
        self,
        task: BenchmarkTask,
        model: str,
        provider: str,
        *,
        store: Store,
        seed: int = 42,
    ) -> BenchmarkRunResult:
        """Run ``task`` against ``model``, capture the trajectory, persist it."""
        # Note: ``seed`` is part of the Benchmark protocol but tau-bench's
        # agent.solve() doesn't accept a seed kwarg. Determinism is achieved
        # via temperature=0.0 on the LLMs; seed shows up only in the runner's
        # bootstrap CI computation downstream. Documented intentional discard.
        _ = seed  # consume the parameter without forwarding

        envs = _import_tau_bench_envs()
        # Pass task_index explicitly to dodge tau-bench's off-by-one bug;
        # we'll override per-task via agent.solve(task_index=...) anyway.
        env = envs.get_env(
            self._env_name,
            user_strategy=self._user_strategy,
            user_model=self._user_model,
            user_provider=self._user_model.split("/", 1)[0],
            task_split=self._last_split,
            task_index=task.task_index,
        )

        # YAML convention: ``model`` is a litellm alias like
        # "anthropic/claude-haiku-4-5-20251001" — provider prefix already
        # included. tau-bench's agent takes ``model`` and ``provider``
        # separately and re-composes the litellm alias internally, so
        # strip the prefix before handing it to tau-bench to avoid
        # double-prefixing ("anthropic/anthropic/…").
        bare_model = model.split("/", 1)[1] if "/" in model else model

        agent = _build_tau_bench_agent(
            kind=self._agent_kind,
            model=bare_model,
            provider=provider,
            env=env,
        )

        # tau-bench's agent.solve() is synchronous; run it off the event loop.
        loop = asyncio.get_running_loop()
        env_result = await loop.run_in_executor(
            None, lambda: agent.solve(env=env, task_index=task.task_index)
        )

        # env_result may be a tau_bench EnvRunResult (frozen dataclass) or
        # similar. Convert to a plain dict for our converter.
        result_dict = env_result if isinstance(env_result, dict) else _as_dict(env_result)
        # SolveResult uses 'messages'; our converter expects 'traj'. Also
        # SolveResult lacks task_id (it lives on the BenchmarkTask, not the
        # result), so we inject it here.
        if "traj" not in result_dict and "messages" in result_dict:
            result_dict["traj"] = result_dict["messages"]
        if "task_id" not in result_dict:
            result_dict["task_id"] = task.task_id

        traj, steps = _convert_tau_traj(
            result_dict,
            instruction=task.instruction,
            model_id=model,
            agent_name=f"tau-bench/{self._agent_kind}",
            agent_version=self._TAU_BENCH_COMMIT,
        )
        await store.save_trajectory(traj, steps)

        reward = float(result_dict.get("reward", 0.0))
        return BenchmarkRunResult(
            trajectory_id=traj.id,
            success=reward >= _SUCCESS_THRESHOLD,
            raw_score=reward,
            error=None,
        )
