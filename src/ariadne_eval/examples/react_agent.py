"""Reference ReAct agent for tracing demos and end-to-end testing.

Text-parsed ReAct loop: the LLM emits Thought/Action/Action Input lines
or a FINAL ANSWER line. The agent parses, looks up the tool, executes it,
appends an Observation, and loops. Uses ``start_trajectory`` and
``enable_litellm_autotrace`` for full tracing.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

from ariadne_eval.adapters.litellm import enable_litellm_autotrace
from ariadne_eval.core.status import TrajectoryStatus
from ariadne_eval.core.trajectory import JsonValue, StepError
from ariadne_eval.examples.tools import TOOLS, Tool
from ariadne_eval.tracing.context import start_trajectory
from ariadne_eval.tracing.decorator import record_tool_call, trace_step

if TYPE_CHECKING:
    from ariadne_eval.storage.base import Store

__all__ = [
    "ReactAgent",
    "ReactParseError",
    "StepLimitExhausted",
]


class ReactParseError(ValueError):
    """Raised when an assistant message does not match the expected format."""


class StepLimitExhausted(RuntimeError):
    """Raised when the loop exceeds ``max_steps`` without emitting FINAL ANSWER."""


_ACTION_RE = re.compile(r"^Action:\s*(\S+)\s*$", re.MULTILINE)
_INPUT_RE = re.compile(r"^Action Input:\s*(.+)$", re.MULTILINE)
_FINAL_RE = re.compile(r"^FINAL ANSWER:\s*(.+)", re.MULTILINE | re.DOTALL)


SYSTEM_PROMPT = """\
You are a ReAct agent. You have access to these tools:
- calculator(expression: str) -> float
- search(query: str) -> str

Use this format strictly:

Thought: <reasoning>
Action: <one of: calculator, search>
Action Input: <input to the tool>

The user will then provide:
Observation: <tool result>

Continue until you can answer, then emit exactly:
Thought: <final reasoning>
FINAL ANSWER: <the answer>

Task: {task}
"""


def _parse_assistant_text(text: str) -> tuple[str | None, str | None, str | None]:
    """Parse an assistant message; returns (action, action_input, final_answer).

    Exactly one of ``final_answer`` or ``(action, action_input)`` is set.
    Raises :class:`ReactParseError` on malformed input.
    """
    final_match = _FINAL_RE.search(text)
    if final_match is not None:
        return None, None, final_match.group(1).strip()

    action_match = _ACTION_RE.search(text)
    input_match = _INPUT_RE.search(text)
    if action_match is None or input_match is None:
        raise ReactParseError(
            f"could not parse Action / Action Input from assistant message: {text!r}"
        )
    return action_match.group(1).strip(), input_match.group(1).strip(), None


class ReactAgent:
    """A minimal reference ReAct agent for tracing demos."""

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        tools: dict[str, Tool] | None = None,
        max_steps: int = 10,
    ) -> None:
        """Build a ReactAgent.

        Args:
            model_id: LiteLLM model identifier.
            tools: Optional override of the default ``TOOLS`` registry.
            max_steps: Maximum tool-call rounds before raising
                :class:`StepLimitExhausted`.
        """
        self.model_id = model_id
        self.tools = tools if tools is not None else TOOLS
        self.max_steps = max_steps

    async def _call_llm(self, messages: list[dict[str, str]]) -> Any:
        """Call the LLM. Indirection so tests can stub without VCR."""
        import litellm  # lazy

        return await litellm.acompletion(model=self.model_id, messages=messages)

    async def arun(self, task: str, *, store: "Store | None" = None) -> str:
        """Run the loop until FINAL ANSWER or ``max_steps`` exhausted.

        Returns the final answer string. Raises :class:`StepLimitExhausted`
        if the loop hits ``max_steps`` first; :class:`ReactParseError` if the
        LLM emits malformed text or asks for an unknown tool.
        """
        enable_litellm_autotrace()

        async with start_trajectory(
            task,
            agent_name="react",
            agent_version="0.1",
            model_id=self.model_id,
            store=store,
        ) as traj:
            messages: list[dict[str, str]] = [
                {"role": "system", "content": SYSTEM_PROMPT.format(task=task)},
            ]
            for _ in range(self.max_steps):
                response = await self._call_llm(messages)
                text = response.choices[0].message.content or ""

                action, action_input, final = _parse_assistant_text(text)
                if final is not None:
                    traj.set_final_answer(final)
                    return final

                if action not in self.tools:
                    traj.set_final_status(TrajectoryStatus.FAILED)
                    raise ReactParseError(f"unknown tool: {action!r}")
                tool = self.tools[action]

                result = await self._run_tool(tool, action_input or "")

                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": f"Observation: {result}\n"})

            traj.set_final_status(TrajectoryStatus.FAILED)
            traj.add_metadata("reason", "step_limit_exhausted")
            raise StepLimitExhausted(
                f"agent exceeded max_steps={self.max_steps} without emitting FINAL ANSWER"
            )

    async def _run_tool(self, tool: Tool, action_input: str) -> JsonValue:
        """Execute a tool inside a @trace_step + record_tool_call wrapper."""

        @trace_step(f"tool_{tool.name}")
        async def _wrapped() -> JsonValue:
            t0 = time.perf_counter()
            try:
                result = tool.fn(action_input)
            except Exception as exc:
                await record_tool_call(
                    tool_name=tool.name,
                    arguments={"input": action_input},
                    result=None,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    error=StepError(type=type(exc).__name__, message=str(exc)),
                )
                raise
            await record_tool_call(
                tool_name=tool.name,
                arguments={"input": action_input},
                result=result,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            return result

        return await _wrapped()
