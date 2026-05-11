# Reference Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal reference ReAct agent under `src/ariadne_eval/examples/` and prove the whole tracing → storage chain works end-to-end via a hand-crafted VCR cassette, following the design at `docs/superpowers/specs/2026-05-11-reference-agent-design.md`.

**Architecture:** Two stub tools (`calculator` via AST-whitelisted arithmetic; `search` via dict lookup). `ReactAgent` text-parses the LLM's Thought/Action/Action Input lines and loops up to `max_steps` times. The agent uses `start_trajectory` + `enable_litellm_autotrace` for LLM tracing and wraps each tool call in `@trace_step` + `record_tool_call`. Integration test plays a hand-crafted cassette via pytest-recording with `record_mode="none"`.

**Tech Stack:** Python 3.11+, `litellm` (lazy-imported), `pytest-recording` / `vcrpy`, `pytest`, `pytest-asyncio` (auto). All pinned in `pyproject.toml`.

**Branch:** `phase-4-reference-agent` (already created on `main` after the Phase 3 merge; the spec is already committed there).

---

## Task 1: Test package markers

**Files:**
- Create: `tests/unit/examples/__init__.py`
- Create: `tests/integration/__init__.py` (if not present)
- Create: `tests/integration/cassettes/__init__.py` (vcrpy ignores it but pytest collection prefers it)

- [ ] **Step 1: Create the markers**

```bash
: > tests/unit/examples/__init__.py
: > tests/integration/__init__.py
mkdir -p tests/integration/cassettes
```

(Note: do NOT create `tests/integration/cassettes/__init__.py` — VCR will try to load it as a cassette.)

- [ ] **Step 2: Verify pytest still passes**

Run: `uv run pytest -m fast`
Expected: 154 passed (Phase 3 baseline).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/examples/__init__.py tests/integration/__init__.py
git commit -m "test: add package markers for tests/unit/examples and tests/integration"
```

---

## Task 2: Tools — calculator, search, _safe_compute

**Files:**
- Create: `src/ariadne_eval/examples/tools.py`
- Test: `tests/unit/examples/test_tools.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/examples/test_tools.py
"""Stub tools used by the reference ReAct agent."""

from __future__ import annotations

import pytest

from ariadne_eval.examples.tools import TOOLS, Tool, calculator, search


@pytest.mark.fast
def test_calculator_basic_arithmetic():
    assert calculator("17*23") == 391
    assert calculator("391/6") == pytest.approx(65.166666, abs=1e-4)
    assert calculator("2+3-1") == 4
    assert calculator("(2+3)*4") == 20
    assert calculator("2**10") == 1024


@pytest.mark.fast
def test_calculator_handles_unary_minus():
    assert calculator("-5+3") == -2


@pytest.mark.fast
@pytest.mark.parametrize(
    "unsafe",
    [
        "__import__('os')",
        "open('/etc/passwd')",
        "x + 1",            # name reference disallowed
        "[1, 2, 3]",        # list literal disallowed
        "1 if True else 0", # IfExp disallowed
        "lambda: 1",        # Lambda disallowed
    ],
)
def test_calculator_rejects_non_arithmetic(unsafe):
    with pytest.raises(ValueError):
        calculator(unsafe)


@pytest.mark.fast
def test_calculator_rejects_syntax_errors():
    with pytest.raises(ValueError):
        calculator("17 *")


@pytest.mark.fast
def test_search_known_query():
    out = search("banana")
    assert "6 letters" in out


@pytest.mark.fast
def test_search_unknown_query_returns_no_results():
    assert search("zzz_nonexistent") == "No results."


@pytest.mark.fast
def test_search_strips_and_lowercases():
    assert search("  Banana  ") == search("banana")


@pytest.mark.fast
def test_tools_registry_has_both_entries():
    assert set(TOOLS.keys()) == {"calculator", "search"}
    assert all(isinstance(t, Tool) for t in TOOLS.values())
    assert TOOLS["calculator"].name == "calculator"
    assert callable(TOOLS["calculator"].fn)
```

- [ ] **Step 2: Run test, expect fail**

Run: `uv run pytest tests/unit/examples/test_tools.py -v`
Expected: ImportError on `ariadne_eval.examples.tools`.

- [ ] **Step 3: Write the implementation**

```python
# src/ariadne_eval/examples/tools.py
"""Stub tools used by the reference ReAct agent.

The calculator parses its input via Python's ``ast`` module and walks the
syntax tree with a whitelisted visitor. Python's built-in expression
evaluator is never invoked. The search tool is a dict lookup against a
small fixed knowledge base.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from ariadne_eval.core.trajectory import JsonValue

__all__ = ["TOOLS", "Tool", "calculator", "search"]


@dataclass(frozen=True)
class Tool:
    """A tool the reference agent can call."""

    name: str
    description: str
    fn: Callable[[str], JsonValue]


_ALLOWED_BIN_OPS: Final = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.FloorDiv,
)
_ALLOWED_UNARY_OPS: Final = (ast.UAdd, ast.USub)


def _safe_compute(expression: str) -> float:
    """Parse and walk an arithmetic expression with an AST whitelist.

    Accepts numeric literals and the basic arithmetic operators. Rejects
    everything else with ``ValueError``.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"could not parse expression: {expression!r}") from exc

    def _walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY_OPS):
            operand = _walk(node.operand)
            return +operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BIN_OPS):
            left = _walk(node.left)
            right = _walk(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                return left / right
            if isinstance(op, ast.Mod):
                return left % right
            if isinstance(op, ast.Pow):
                return left**right
            if isinstance(op, ast.FloorDiv):
                return left // right
        raise ValueError(
            f"disallowed expression node: {type(node).__name__} in {expression!r}"
        )

    return _walk(tree)


def calculator(expression: str) -> float:
    """Evaluate a basic arithmetic expression safely."""
    return _safe_compute(expression)


_SEARCH_DB: Final[dict[str, str]] = {
    "banana": "Banana is a fruit. The word has 6 letters.",
    "ariadne": "Ariadne gave Theseus a thread to navigate the labyrinth.",
}


def search(query: str) -> str:
    """Return a fixed answer for a small set of demo queries."""
    return _SEARCH_DB.get(query.lower().strip(), "No results.")


TOOLS: Final[dict[str, Tool]] = {
    "calculator": Tool(
        name="calculator",
        description="calculator(expression: str) -> float — evaluate a basic arithmetic expression",
        fn=calculator,
    ),
    "search": Tool(
        name="search",
        description="search(query: str) -> str — search a small fixed knowledge base",
        fn=search,
    ),
}
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/unit/examples/test_tools.py -v`
Expected: every test passes (about 13 with parametrize expansion).

- [ ] **Step 5: Commit**

```bash
git add src/ariadne_eval/examples/tools.py tests/unit/examples/test_tools.py
git commit -m "feat(examples): add calculator and search stub tools"
```

---

## Task 3: ReactAgent — parser, errors, loop

**Files:**
- Create: `src/ariadne_eval/examples/react_agent.py`
- Test: `tests/unit/examples/test_react_agent.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/examples/test_react_agent.py
"""ReactAgent parser, error paths, and step-limit exhaustion.

These tests do not touch a real LLM — they patch ``ReactAgent._call_llm``
to inject canned responses.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ariadne_eval.examples.react_agent import (
    ReactAgent,
    ReactParseError,
    StepLimitExhausted,
    _parse_assistant_text,
)


@pytest.mark.fast
def test_parse_action_and_input():
    text = (
        "Thought: I need to compute 17*23.\n"
        "Action: calculator\n"
        "Action Input: 17*23\n"
    )
    action, action_input, final = _parse_assistant_text(text)
    assert action == "calculator"
    assert action_input == "17*23"
    assert final is None


@pytest.mark.fast
def test_parse_final_answer():
    text = "Thought: That's it.\nFINAL ANSWER: 42"
    action, action_input, final = _parse_assistant_text(text)
    assert action is None
    assert action_input is None
    assert final == "42"


@pytest.mark.fast
def test_parse_final_answer_multiline():
    text = "Thought: details\nFINAL ANSWER: 65.16\n(more reasoning)"
    _, _, final = _parse_assistant_text(text)
    assert final is not None
    assert final.startswith("65.16")


@pytest.mark.fast
def test_parse_malformed_raises():
    with pytest.raises(ReactParseError):
        _parse_assistant_text("just some random text with no markers")


@pytest.mark.fast
def test_parse_action_without_input_raises():
    with pytest.raises(ReactParseError):
        _parse_assistant_text("Action: calculator\n")


def _fake_response(content: str) -> SimpleNamespace:
    """Build a minimal ChatCompletion-shaped response for stubbing."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=10),
    )


@pytest.mark.fast
async def test_step_limit_exhaustion(monkeypatch):
    """When the LLM never emits FINAL ANSWER, we hit max_steps and raise."""
    agent = ReactAgent(model_id="gpt-4o-mini", max_steps=3)

    call_count = {"n": 0}

    async def _stub_call_llm(messages):
        call_count["n"] += 1
        return _fake_response(
            "Thought: keep going\nAction: calculator\nAction Input: 1+1\n"
        )

    monkeypatch.setattr(agent, "_call_llm", _stub_call_llm)

    with pytest.raises(StepLimitExhausted):
        await agent.arun("forever loop")

    # Should call exactly max_steps times before giving up
    assert call_count["n"] == 3


@pytest.mark.fast
async def test_arun_returns_final_answer(monkeypatch):
    agent = ReactAgent(model_id="gpt-4o-mini", max_steps=5)

    responses = iter(
        [
            "Thought: I'll search.\nAction: search\nAction Input: banana\n",
            "Thought: That has 6 letters.\nFINAL ANSWER: 6",
        ]
    )

    async def _stub_call_llm(messages):
        return _fake_response(next(responses))

    monkeypatch.setattr(agent, "_call_llm", _stub_call_llm)
    answer = await agent.arun("how many letters in banana")
    assert answer.strip() == "6"


@pytest.mark.fast
async def test_arun_persists_to_store(monkeypatch, tmp_path):
    from ariadne_eval.storage.duckdb_store import DuckDBStore

    agent = ReactAgent(model_id="gpt-4o-mini")
    responses = iter(
        [
            "Thought: simple.\nAction: calculator\nAction Input: 1+1\n",
            "Thought: done.\nFINAL ANSWER: 2",
        ]
    )

    async def _stub_call_llm(messages):
        return _fake_response(next(responses))

    monkeypatch.setattr(agent, "_call_llm", _stub_call_llm)

    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        await agent.arun("compute 1+1", store=store)
        listed = await store.list_trajectories()
        assert len(listed) == 1
        _, steps = await store.get_trajectory(listed[0].id)
        names = {s.name for s in steps}
        assert "tool_calculator" in names
        assert "calculator" in names  # the record_tool_call step
    finally:
        await store.close()
```

- [ ] **Step 2: Run tests, expect fail**

Run: `uv run pytest tests/unit/examples/test_react_agent.py -v`
Expected: ImportError on `ariadne_eval.examples.react_agent`.

- [ ] **Step 3: Write the implementation**

```python
# src/ariadne_eval/examples/react_agent.py
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
    """Raised when the agent loop exceeds ``max_steps`` without emitting FINAL ANSWER."""


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
    """Parse an assistant message. Returns (action, action_input, final_answer).

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
        if the loop hits ``max_steps`` first.
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

                # action and action_input are non-None here
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
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/unit/examples/test_react_agent.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ariadne_eval/examples/react_agent.py tests/unit/examples/test_react_agent.py
git commit -m "feat(examples): add ReactAgent with text-parsed ReAct loop"
```

---

## Task 4: Integration conftest with VCR config

**Files:**
- Create: `tests/integration/conftest.py`

- [ ] **Step 1: Write the conftest**

```python
# tests/integration/conftest.py
"""Shared fixtures for the integration test suite.

VCR config: redact auth headers, refuse to make real HTTP calls
(``record_mode='none'``), and match on method + URL only so the
cassette is robust to minor request-body differences.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, object]:
    """pytest-recording reads this fixture for the ``@pytest.mark.vcr`` config."""
    return {
        "filter_headers": [
            ("authorization", "REDACTED"),
            ("x-api-key", "REDACTED"),
            ("openai-organization", "REDACTED"),
            ("anthropic-version", None),
        ],
        "record_mode": "none",
        "match_on": ["method", "scheme", "host", "port", "path"],
    }
```

- [ ] **Step 2: Verify pytest still collects without errors**

Run: `uv run pytest -m fast --collect-only 2>&1 | tail -5`
Expected: no collection errors.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/conftest.py
git commit -m "test(integration): add VCR config with record_mode=none and header redaction"
```

---

## Task 5: Hand-craft the cassette

**Files:**
- Create: `tests/integration/cassettes/test_react_agent_traces_end_to_end.yaml`

- [ ] **Step 1: Create the cassette directory if missing**

```bash
mkdir -p tests/integration/cassettes
```

- [ ] **Step 2: Write the cassette**

Create `tests/integration/cassettes/test_react_agent_traces_end_to_end.yaml`:

```yaml
interactions:
  - request:
      body: ''
      headers:
        Authorization:
          - REDACTED
        Content-Type:
          - application/json
      method: POST
      uri: https://api.openai.com/v1/chat/completions
    response:
      body:
        string: '{"id":"chatcmpl-stub-1","object":"chat.completion","created":1700000000,"model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"Thought: I need to compute 17*23 first.\nAction: calculator\nAction Input: 17*23\n"},"finish_reason":"stop"}],"usage":{"prompt_tokens":50,"completion_tokens":20,"total_tokens":70}}'
      headers:
        Content-Type:
          - application/json
      status:
        code: 200
        message: OK
  - request:
      body: ''
      headers:
        Authorization:
          - REDACTED
        Content-Type:
          - application/json
      method: POST
      uri: https://api.openai.com/v1/chat/completions
    response:
      body:
        string: '{"id":"chatcmpl-stub-2","object":"chat.completion","created":1700000001,"model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"Thought: 17*23 is 391. The word ''banana'' has 6 letters. So I divide.\nAction: calculator\nAction Input: 391/6\n"},"finish_reason":"stop"}],"usage":{"prompt_tokens":75,"completion_tokens":25,"total_tokens":100}}'
      headers:
        Content-Type:
          - application/json
      status:
        code: 200
        message: OK
  - request:
      body: ''
      headers:
        Authorization:
          - REDACTED
        Content-Type:
          - application/json
      method: POST
      uri: https://api.openai.com/v1/chat/completions
    response:
      body:
        string: '{"id":"chatcmpl-stub-3","object":"chat.completion","created":1700000002,"model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"Thought: The result is 65.166666...\nFINAL ANSWER: 65.16666666666667"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":15,"total_tokens":115}}'
      headers:
        Content-Type:
          - application/json
      status:
        code: 200
        message: OK
version: 1
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration/cassettes/test_react_agent_traces_end_to_end.yaml
git commit -m "test(integration): hand-crafted VCR cassette for ReAct E2E test

3 turns: calculator(17*23) -> calculator(391/6) -> FINAL ANSWER 65.166...
Authorization header redacted; subsequent re-recordings will be cleaned by
the filter_headers config in conftest.py."
```

---

## Task 6: Integration test

**Files:**
- Create: `tests/integration/test_react_end_to_end.py`

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_react_end_to_end.py
"""End-to-end: ReactAgent + LiteLLM autotrace + DuckDBStore via VCR cassette."""

from __future__ import annotations

import pytest

from ariadne_eval.examples.react_agent import ReactAgent
from ariadne_eval.storage.duckdb_store import DuckDBStore


@pytest.mark.integration
@pytest.mark.vcr
async def test_react_agent_traces_end_to_end(tmp_path):
    """Full chain: agent loop -> litellm -> autotrace callback -> store."""
    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        agent = ReactAgent(model_id="gpt-4o-mini")
        answer = await agent.arun(
            "What is 17 * 23, and then divide by the number of letters in 'banana'?",
            store=store,
        )
        assert "65" in str(answer)

        listed = await store.list_trajectories()
        assert len(listed) == 1

        traj, steps = await store.get_trajectory(listed[0].id)
        assert traj.final_status.value == "succeeded"

        step_payload_types = {type(s.payload).__name__ for s in steps}
        # Three LLM calls (via autotrace), two tool wrappers (@trace_step),
        # two tool-call recordings.
        assert "LLMCallPayload" in step_payload_types
        assert "ToolCallPayload" in step_payload_types
        assert "InternalPayload" in step_payload_types
    finally:
        await store.close()
```

- [ ] **Step 2: Run the integration test**

Run: `uv run pytest -m integration tests/integration/test_react_end_to_end.py -v`
Expected: 1 passed. (No real network call; cassette serves the responses.)

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_react_end_to_end.py
git commit -m "test(integration): end-to-end ReAct agent test via VCR cassette"
```

---

## Task 7: Rewrite examples/01_quickstart

**Files:**
- Modify: `examples/01_quickstart/main.py`
- Modify: `examples/01_quickstart/README.md`

- [ ] **Step 1: Rewrite the main script**

Replace `examples/01_quickstart/main.py` with:

```python
"""Trace a real ReAct agent end-to-end.

Requires ``OPENAI_API_KEY`` in your environment. Run with:

    uv run python examples/01_quickstart/main.py

The trajectory is persisted to ``~/.ariadne/quickstart.duckdb``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ariadne_eval import DuckDBStore
from ariadne_eval.examples.react_agent import ReactAgent


async def main() -> None:
    store_path = Path("~/.ariadne/quickstart.duckdb").expanduser()
    store = DuckDBStore(path=store_path)
    try:
        agent = ReactAgent(model_id="gpt-4o-mini")
        answer = await agent.arun(
            "What is 17 * 23, and then divide by the number of letters in 'banana'?",
            store=store,
        )
        print(f"final answer: {answer}")
        print(f"trajectory persisted to: {store_path}")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Rewrite the README**

Replace `examples/01_quickstart/README.md` with:

````markdown
# 01 — Quickstart

A real ReAct agent traced end-to-end. The agent uses an LLM (default
gpt-4o-mini via litellm) and two stub tools (calculator + search).

## Prerequisites

Set `OPENAI_API_KEY` in your shell or in `.env`:

```bash
export OPENAI_API_KEY=sk-...
```

To use a different model (e.g. Anthropic Claude), set the matching env
var and edit `model_id="..."` in `main.py`. LiteLLM handles routing.

## Run

```bash
uv run python examples/01_quickstart/main.py
```

Expected output (the LLM's exact wording may vary):

```
final answer: 65.16666666666667
trajectory persisted to: /Users/.../.ariadne/quickstart.duckdb
```

## What it shows

- `start_trajectory(...)` opens an async tracing context.
- `enable_litellm_autotrace()` auto-records every `litellm.acompletion` call
  as an `llm_call` Step.
- `@trace_step("tool_calculator")` wraps each tool invocation as an
  `internal` Step (the structural step in the trace tree).
- `record_tool_call(...)` adds the typed `ToolCallPayload` as a child.
- The full trajectory is saved to DuckDB at context exit.

Once the replay UI ships (v0.0.9), point `ariadne ui` at the same
DuckDB file to drill into the trace.

## Re-running

Each run produces a new trajectory; the DuckDB file grows over time.
Delete `~/.ariadne/quickstart.duckdb` to start fresh.
````

- [ ] **Step 3: Verify the example imports cleanly**

Run: `uv run python -c "import examples.01_quickstart.main" 2>&1 | tail -2`

The exact command above won't work because of the directory name starting with a digit; instead verify the script can be parsed:

Run: `uv run python -c "import ast; ast.parse(open('examples/01_quickstart/main.py').read())"`
Expected: no output (parses cleanly).

- [ ] **Step 4: Commit**

```bash
git add examples/01_quickstart/main.py examples/01_quickstart/README.md
git commit -m "docs(examples): quickstart now uses the ReactAgent reference"
```

---

## Task 8: Wire up mkdocs include and rewrite docs/quickstart.md

**Files:**
- Modify: `mkdocs.yml`
- Modify: `docs/quickstart.md`

- [ ] **Step 1: Enable the snippets extension in mkdocs.yml**

Modify `mkdocs.yml` `markdown_extensions` section to add `pymdownx.snippets` with a base path:

```yaml
markdown_extensions:
  - admonition
  - attr_list
  - md_in_html
  - tables
  - toc:
      permalink: true
  - pymdownx.details
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - pymdownx.snippets:
      base_path:
        - .
      check_paths: true
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
```

(The only change is adding the `base_path` + `check_paths` to the existing `pymdownx.snippets` entry, or adding the entire entry if absent.)

- [ ] **Step 2: Rewrite docs/quickstart.md**

Replace the contents of `docs/quickstart.md` with:

````markdown
# Quickstart

A real ReAct agent, traced end-to-end. Takes ~5 seconds and a few cents
of OpenAI credit.

## Install

```bash
pip install ariadne-eval
```

## Set your API key

```bash
export OPENAI_API_KEY=sk-...
```

## Run

```bash
uv run python examples/01_quickstart/main.py
```

The example, verbatim from the repo:

```python
--8<-- "examples/01_quickstart/main.py"
```

You should see something like:

```
final answer: 65.16666666666667
trajectory persisted to: ~/.ariadne/quickstart.duckdb
```

## What just happened

The agent:

1. Asked gpt-4o-mini for the next action (LLM call #1, auto-traced).
2. Got `Action: calculator, Action Input: 17*23` back.
3. Ran the calculator tool inside `@trace_step("tool_calculator")` →
   `record_tool_call(...)`.
4. Sent the observation back to the LLM (call #2).
5. Looped one more time to produce the final answer.

The whole tree is saved as a single `Trajectory` in DuckDB with five
or so `Step` rows. Once the replay UI ships (v0.0.9), point
`ariadne ui` at the file to drill in.

## Next

- [Tracing concepts](concepts/tracing.md) — how `@trace_step`, recorders,
  and sampling fit together.
- [Storage](concepts/storage.md) — schema, JSONL portability, the limits.
- The repo's `examples/01_quickstart/README.md` — full prerequisites and
  troubleshooting.
````

- [ ] **Step 3: Verify docs build**

Run: `uv run mkdocs build --strict 2>&1 | tail -5`
Expected: clean build (the `--8<--` snippet should resolve to the example's contents).

- [ ] **Step 4: Commit**

```bash
git add mkdocs.yml docs/quickstart.md
git commit -m "docs(quickstart): pull example via mkdocs include so doc and code can't drift"
```

---

## Task 9: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add the entry**

Modify the `[Unreleased]` section in `CHANGELOG.md` to add (above the existing entries):

```markdown
### Added
- Reference ReAct agent (`ariadne_eval.examples.react_agent.ReactAgent`)
  with text-parsed ReAct loop, two stub tools (`calculator` via
  AST-whitelisted arithmetic, `search` via dict lookup), and
  `StepLimitExhausted` / `ReactParseError` errors. Used by
  `examples/01_quickstart/` and the new end-to-end integration test.
- End-to-end integration test via a hand-crafted VCR cassette
  (`tests/integration/test_react_end_to_end.py`) with
  `record_mode="none"` so CI never makes real HTTP calls. Auth headers
  redacted via the `vcr_config` fixture.
- mkdocs `pymdownx.snippets` extension wired up so docs can include
  files verbatim via `--8<--` syntax — the quickstart docs page now
  pulls `examples/01_quickstart/main.py` directly, so the example and
  the docs cannot drift.
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): unreleased entry for the reference agent"
```

---

## Task 10: Final verification + tag v0.0.5-alpha

**Files:** none (verification only).

- [ ] **Step 1: All fast tests pass**

Run: `uv run pytest -m fast`
Expected: every test green; total well above the Phase 3 baseline of 154.

- [ ] **Step 2: Integration test passes via cassette**

Run: `uv run pytest -m integration tests/integration/test_react_end_to_end.py -v`
Expected: 1 passed.

- [ ] **Step 3: Coverage on the new code**

Run: `uv run pytest -m fast --cov=src/ariadne_eval/examples --cov-report=term`
Expected: each file in `src/ariadne_eval/examples/` shows ≥90 % coverage.

(Examples are reference code, not core library code; ≥90 % is acceptable here vs the 95 % target for `core/`, `storage/`, `tracing/`.)

- [ ] **Step 4: mypy strict**

Run: `uv run mypy --strict`
Expected: `Success: no issues found`.

- [ ] **Step 5: ruff + format**

Run: `uv run ruff check && uv run ruff format --check`
Expected: both green.

- [ ] **Step 6: Pre-commit clean**

Run: `uv run pre-commit run --all-files`
Expected: all hooks pass.

- [ ] **Step 7: Docs build clean**

Run: `uv run mkdocs build --strict`
Expected: clean build, no warnings.

- [ ] **Step 8: Tag the phase**

```bash
git tag v0.0.5-alpha -m "Phase 4: reference agent + end-to-end wiring

Reference ReAct agent with text-parsed loop and two stub tools
(AST-whitelisted calculator + dict-lookup search). End-to-end
integration test via a hand-crafted VCR cassette with
record_mode='none' so CI never makes real network calls. Quickstart
example and docs now use the reference agent; docs pull the example
via mkdocs include so they cannot drift."
```

---

## Self-review

**Spec coverage check:**

| Spec section | Task |
|---|---|
| `tools.py` (`Tool`, `calculator`, `search`, `_safe_compute`) | Task 2 |
| `react_agent.py` (`ReactAgent`, `_parse_assistant_text`, errors) | Task 3 |
| Integration conftest with VCR config | Task 4 |
| Hand-crafted cassette | Task 5 |
| Integration test asserting trace shape | Task 6 |
| Rewrite `examples/01_quickstart/{main.py,README.md}` | Task 7 |
| `docs/quickstart.md` via mkdocs include | Task 8 |
| CHANGELOG entry | Task 9 |
| Final verification + alpha tag | Task 10 |

All sections covered.

**Type consistency check:** `ReactAgent`, `ReactParseError`,
`StepLimitExhausted`, `_parse_assistant_text`, `Tool`, `TOOLS`,
`_safe_compute`, `_call_llm`, `_run_tool`, `enable_litellm_autotrace`,
`@trace_step`, `record_tool_call`, `start_trajectory` are referenced
consistently across tasks 2–7.

**Placeholder scan:** no TBD / TODO / "implement later" markers.
