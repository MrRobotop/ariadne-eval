# Phase 4 Design: Reference Agent and End-to-End Wiring

**Status:** Approved (2026-05-11) · **Phase:** 4 · **Target version:** 0.0.5

## Goal

Ship a minimal reference ReAct agent that exercises the entire tracing -> storage ->
retrieval chain end-to-end. The agent is both:

- A working library example users can import (`ariadne_eval.examples.react_agent.ReactAgent`).
- The substrate for an integration test that proves the whole stack works in CI
  without any external API key, via a hand-crafted VCR cassette.

The reference is intentionally minimal — it's a *tracing demo*, not an
agent-quality demo. Text-parsed ReAct (Thought / Action / Action Input /
Observation), two stub tools, no tool schemas. Real production-grade agents
are a downstream concern.

## Scope

In scope: `ReactAgent` class, two stub tools (`calculator`, `search`), the
integration test with a hand-crafted cassette, a rewritten quickstart example
and docs page that use the reference agent.

Out of scope: structured tool calls / OpenAI function calling, real internet
search, multi-agent orchestration, prompt optimization. All deferred to a
later phase or out of charter.

## Architecture

| File | Responsibility |
|---|---|
| `src/ariadne_eval/examples/tools.py` | `Tool` dataclass, `calculator`, `search`, `TOOLS` registry, `_safe_compute` AST visitor. |
| `src/ariadne_eval/examples/react_agent.py` | `ReactAgent` class, system prompt, regex parser, `ReactParseError`. |
| `tests/integration/conftest.py` | VCR config (`filter_headers`, `record_mode="none"`). |
| `tests/integration/test_react_end_to_end.py` | The single integration test, marked `@pytest.mark.integration` + `@pytest.mark.vcr`. |
| `tests/integration/cassettes/test_react_end_to_end.yaml` | Hand-crafted 3-turn ReAct conversation. |
| `tests/unit/examples/test_tools.py` | Unit tests for `calculator` (rejects unsafe) and `search`. |
| `tests/unit/examples/test_react_agent.py` | Unit tests for the parser and max-steps exhaustion path (uses an in-process stub LLM, no cassette). |
| `examples/01_quickstart/main.py` | (rewrite) imports `ReactAgent`, runs the canonical task. |
| `examples/01_quickstart/README.md` | (rewrite) how to run, what to expect. |
| `docs/quickstart.md` | (rewrite) mirrors the example via mkdocs include. |

## ReAct prompt format

```
You have access to these tools:
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
```

Text-parsed for three reasons: (1) it matches the original ReAct paper, (2) it
is dramatically easier to hand-craft cassettes in this format than as
structured function-call JSON, (3) tracing — not the prompt engineering — is
what this phase showcases.

## Parser

Regex-based:

```python
_ACTION_RE = re.compile(r"^Action:\s*(\w+)\s*$", re.MULTILINE)
_INPUT_RE  = re.compile(r"^Action Input:\s*(.+)$", re.MULTILINE)
_FINAL_RE  = re.compile(r"^FINAL ANSWER:\s*(.+)$", re.MULTILINE | re.DOTALL)
```

Extracts the first `Action:` block or the `FINAL ANSWER:` line. Unparseable
responses raise `ReactParseError`; the agent catches the error, marks the
trajectory `FAILED`, and re-raises so callers see the failure.

## Tools

The calculator parses its input through Python's `ast` module and walks the
syntax tree with a whitelisted visitor. It never invokes Python's built-in
interpreter — only numeric literals and the four arithmetic operators plus
power and modulo are accepted.

The `Tool` dataclass is frozen and holds three fields: name, description, and
a `Callable[[str], JsonValue]` function. A module-level `TOOLS` dict maps
tool names to `Tool` instances.

`search` is a dict lookup against a small fixed knowledge base — banana,
ariadne. Unknown queries return "No results.".

### `_safe_compute`

Uses `ast.parse(expression, mode="eval")` to *parse* (not run) the expression.
The resulting tree is walked by a `NodeVisitor` subclass that accepts:

- `ast.Constant` (numeric literals)
- `ast.BinOp` with `Add` / `Sub` / `Mult` / `Div` / `Mod` / `Pow` / `FloorDiv`
- `ast.UnaryOp` with `UAdd` / `USub`
- `ast.Expression` (the wrapper)

Anything else — `Name`, `Call`, `Attribute`, `Subscript` — raises
`ValueError("disallowed expression: ...")`. This matters even for a demo: an
LLM that's asked to "calculate the result" may return arbitrary Python code,
and a calculator that runs LLM-emitted source is the kind of unforced error
that ends up in a security advisory.

## ReactAgent

```python
class ReactAgent:
    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        tools: dict[str, Tool] | None = None,
        max_steps: int = 10,
    ) -> None: ...

    async def arun(
        self,
        task: str,
        *,
        store: Store | None = None,
    ) -> str:
        """Run the loop until FINAL ANSWER or max_steps. Return the answer."""
```

Loop:

1. `enable_litellm_autotrace()` — idempotent; safe to call every run.
2. `async with start_trajectory(task, agent_name="react", agent_version="0.1",
   model_id=self.model_id, store=store) as traj:`.
3. Build `messages = [{"role": "system", "content": SYSTEM_PROMPT.format(task=task)}]`.
4. For step in `range(self.max_steps)`:
   - `response = await litellm.acompletion(model=self.model_id, messages=messages)`.
     The autotrace callback fires synchronously, recording an `llm_call` step.
   - `text = response.choices[0].message.content or ""`.
   - If `FINAL ANSWER:` matches: extract the answer, `traj.set_final_answer(...)`, return.
   - Else parse `Action:` + `Action Input:`. If parse fails: mark `FAILED`, raise.
   - Look up the tool. If unknown: mark `FAILED`, raise.
   - Wrap the tool call in `@trace_step(f"tool_{action}")` and inside that
     function call `record_tool_call(...)` with the typed payload. If the
     tool raises, record a failed `record_tool_call(..., error=...)` and
     re-raise.
   - Append the assistant message and an `Observation: <result>` user message to `messages`. Continue.
5. If we exit the loop without `FINAL ANSWER`: `traj.set_final_status(FAILED)`, `traj.add_metadata("reason", "step_limit_exhausted")`, raise `StepLimitExhausted`.

## Cassette + VCR configuration

`tests/integration/conftest.py`:

```python
import pytest


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [
            ("authorization", "REDACTED"),
            ("x-api-key", "REDACTED"),
            ("openai-organization", "REDACTED"),
            ("anthropic-version", None),  # drop the header entirely
        ],
        "record_mode": "none",
        "match_on": ["method", "scheme", "host", "port", "path"],
    }
```

`record_mode="none"` means the test will fail loudly if it tries to make a
real HTTP call (e.g. someone deleted the cassette). That's the safety we want
in CI — no silent recordings, no accidental API spend.

`match_on` excludes the request body so the cassette is robust to small
prompt differences (we hand-crafted it; small drift in the system prompt is
OK).

### Hand-crafted cassette

Three responses:

1. `"Thought: I need to compute 17*23 first.\nAction: calculator\nAction Input: 17*23"`
2. `"Thought: 17*23 is 391. The word 'banana' has 6 letters. So I divide.\nAction: calculator\nAction Input: 391/6"`
3. `"Thought: The result is 65.166...\nFINAL ANSWER: 65.16666666666667"`

That produces 3 LLM calls + 2 tool calls = 5 traced events when assembled
into a trajectory.

The cassette file's `Authorization` header is pre-redacted to
`Bearer REDACTED`. The `filter_headers` config keeps any future re-recording
clean too.

## Integration test

```python
@pytest.mark.integration
@pytest.mark.vcr
async def test_react_agent_traces_end_to_end(tmp_path):
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
        assert "LLMCallPayload" in step_payload_types
        assert "ToolCallPayload" in step_payload_types
        assert "InternalPayload" in step_payload_types  # the @trace_step wrapper
    finally:
        await store.close()
```

The pytest-recording plugin automatically loads the cassette named after the
test function. No path wiring needed beyond the conftest fixture.

## Unit tests

`tests/unit/examples/test_tools.py`:

- `calculator("17*23")` returns `391`.
- `calculator("391/6")` returns `~65.166...`.
- `calculator("__import__('os')")` raises `ValueError` (Name + Call disallowed).
- `calculator("open('/etc/passwd')")` raises `ValueError`.
- `search("banana")` returns the canned answer.
- `search("unknown")` returns "No results.".

`tests/unit/examples/test_react_agent.py`:

- Parser extracts action / action_input from a typical assistant message.
- Parser detects `FINAL ANSWER:` and extracts the answer.
- Parser raises `ReactParseError` on malformed text.
- An in-process stub LLM (a callable that always returns a non-FINAL response)
  triggers `StepLimitExhausted` after `max_steps`.

The stub LLM is injected via a `_call_llm` indirection on `ReactAgent` — the
default implementation calls `litellm.acompletion`; tests patch it to a
function that returns a canned `ChatCompletion`-shaped object.

## Examples and docs

`examples/01_quickstart/main.py` becomes a short script that imports
`ReactAgent`, instantiates a `DuckDBStore`, calls `agent.arun(...)`, and
prints the final answer. Full code in the file itself; the docs page pulls
it via mkdocs include so the two cannot drift.

`docs/quickstart.md` uses the mkdocs include syntax to pull
`examples/01_quickstart/main.py` verbatim. The doc cannot drift from the
code.

## Public API additions

Minimal:

```python
# ariadne_eval/__init__.py adds nothing for Phase 4.
# The reference agent is accessed via:
from ariadne_eval.examples.react_agent import ReactAgent
```

The reference is intentionally not re-exported at the package root — it's an
example, not part of the production API surface.

## Edge cases handled

1. **LLM returns malformed text.** `ReactParseError` raised; trajectory
   `FAILED`; exception propagates so the caller knows.
2. **LLM asks for an unknown tool.** Same path — `ReactParseError` covers it
   (with a clear message).
3. **Tool raises.** Captured by the inner try/except;
   `record_tool_call(..., error=...)` records a `FAILED` ToolCall step;
   exception re-raised so the agent's `@trace_step` also catches it.
4. **Loop exceeds `max_steps`.** `StepLimitExhausted`; trajectory `FAILED`
   with metadata `{"reason": "step_limit_exhausted"}`.
5. **Cassette missing in CI.** `record_mode="none"` raises VCR's
   `CannotOverwriteExistingCassetteException`. Tests fail loudly, no silent
   re-recording.
6. **Cassette contains a real API key.** Pre-redacted in the committed file;
   `filter_headers` config catches accidental future re-recordings.
7. **`OPENAI_API_KEY` set in CI.** Has no effect — `record_mode="none"`
   never makes a network call; cassette is replayed.

## Documentation

- `examples/01_quickstart/main.py` is the canonical "how to use this thing"
  artifact.
- `examples/01_quickstart/README.md` documents the prerequisites
  (`OPENAI_API_KEY`) and the expected output.
- `docs/quickstart.md` reuses the example via mkdocs include.
- `docs/concepts/tracing.md` (existing) gets a "see also the reference agent"
  pointer.
- `CHANGELOG.md [Unreleased]` records the reference agent and the
  integration test.

## Deferred

- Structured tool calls (`tools=` argument to `litellm.acompletion`).
  Text-parsed is fine for v0.0.5; modern function-calling lands when a real
  consumer asks.
- Real-API recording mode. Hand-crafted cassette ships with v0.0.5; the
  README documents how to re-record with `pytest --record-mode=rewrite`
  plus a real key when we want to refresh.
- Streaming responses.
- Multi-agent / sub-agent composition.
