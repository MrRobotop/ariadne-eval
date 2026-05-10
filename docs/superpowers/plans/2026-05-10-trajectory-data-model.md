# Trajectory Data Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Pydantic v2 trajectory data model under `src/ariadne_eval/core/`
following the design in `docs/superpowers/specs/2026-05-10-trajectory-data-model-design.md`.

**Architecture:** Three modules — `core.status` (enums), `core.ids` (ULID
generation), `core.trajectory` (all data classes). Discriminated `Payload`
union over four `step_type` values; tree-shaped via string `parent_step_id`
references; ULID-based time-sortable IDs; tz-aware UTC datetimes only;
truncation only on `completion` and `result`; opt-in redact hook.

**Tech Stack:** Python 3.11+, Pydantic v2, python-ulid, pytest, hypothesis,
mypy --strict, ruff. All already pinned in `pyproject.toml`.

**Branch:** `phase-1-trajectory-model` (already created, contains the spec doc).

---

## Task 1: Pytest package markers

These two empty `__init__.py` files prevent pytest from collecting two
files with the same basename under different paths and crashing with
"import file mismatch". Doing this first means every later task can run
its tests without surprise.

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/core/__init__.py`
- Create: `tests/property/__init__.py`

- [ ] **Step 1: Create the four empty package markers**

```bash
: > tests/__init__.py
: > tests/unit/__init__.py
: > tests/unit/core/__init__.py
: > tests/property/__init__.py
```

- [ ] **Step 2: Verify pytest still passes**

Run: `uv run pytest -m fast`
Expected: `4 passed` (the existing smoke tests, undisturbed).

- [ ] **Step 3: Commit**

```bash
git add tests/__init__.py tests/unit/__init__.py tests/unit/core/__init__.py tests/property/__init__.py
git commit -m "test: add package markers for tests/unit/core and tests/property"
```

---

## Task 2: Status enums (RED → GREEN)

**Files:**
- Create: `src/ariadne_eval/core/status.py`
- Test: `tests/unit/core/test_status.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/core/test_status.py
"""Status enums are part of the public API; their string values must not
change without a major version bump. These tests pin the values."""

import pytest

from ariadne_eval.core.status import StepStatus, TrajectoryStatus


@pytest.mark.fast
def test_step_status_string_values():
    assert StepStatus.PENDING.value == "pending"
    assert StepStatus.RUNNING.value == "running"
    assert StepStatus.SUCCEEDED.value == "succeeded"
    assert StepStatus.FAILED.value == "failed"
    assert StepStatus.SKIPPED.value == "skipped"


@pytest.mark.fast
def test_trajectory_status_string_values():
    assert TrajectoryStatus.RUNNING.value == "running"
    assert TrajectoryStatus.SUCCEEDED.value == "succeeded"
    assert TrajectoryStatus.FAILED.value == "failed"
    assert TrajectoryStatus.ABORTED.value == "aborted"


@pytest.mark.fast
def test_status_enums_are_str_enums():
    """StrEnum members must compare equal to their string value (cf. JSON
    serialization in Pydantic v2)."""
    assert StepStatus.SUCCEEDED == "succeeded"
    assert TrajectoryStatus.ABORTED == "aborted"


@pytest.mark.fast
def test_step_status_full_membership():
    assert {s.value for s in StepStatus} == {
        "pending", "running", "succeeded", "failed", "skipped"
    }


@pytest.mark.fast
def test_trajectory_status_full_membership():
    assert {s.value for s in TrajectoryStatus} == {
        "running", "succeeded", "failed", "aborted"
    }
```

- [ ] **Step 2: Run test, expect fail**

Run: `uv run pytest tests/unit/core/test_status.py -v`
Expected: ImportError / ModuleNotFoundError on `ariadne_eval.core.status`.

- [ ] **Step 3: Write the implementation**

```python
# src/ariadne_eval/core/status.py
"""Status enums for trajectories and steps.

These are part of the public API. Renaming a member or changing a string
value is a major version bump.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["StepStatus", "TrajectoryStatus"]


class StepStatus(StrEnum):
    """Lifecycle status of a single ``Step`` in a trajectory."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class TrajectoryStatus(StrEnum):
    """Lifecycle status of an end-to-end ``Trajectory``."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"
```

- [ ] **Step 4: Run test, expect pass**

Run: `uv run pytest tests/unit/core/test_status.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ariadne_eval/core/status.py tests/unit/core/test_status.py
git commit -m "feat(core): add StepStatus and TrajectoryStatus enums"
```

---

## Task 3: ID module (RED → GREEN)

**Files:**
- Create: `src/ariadne_eval/core/ids.py`
- Test: `tests/unit/core/test_ids.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/core/test_ids.py
"""ULID-based IDs: time-sortable, unique, validatable."""

import time

import pytest

from ariadne_eval.core.ids import is_valid_id, new_id


@pytest.mark.fast
def test_new_id_returns_26_char_string():
    out = new_id()
    assert isinstance(out, str)
    assert len(out) == 26


@pytest.mark.fast
def test_new_id_is_unique_over_10k_samples():
    seen = {new_id() for _ in range(10_000)}
    assert len(seen) == 10_000


@pytest.mark.fast
def test_new_id_is_time_sortable_across_milliseconds():
    """ULIDs encode the timestamp in their first 10 chars; IDs minted
    in monotonically later milliseconds must sort lexicographically
    later."""
    earlier = new_id()
    time.sleep(0.005)  # 5ms — guaranteed boundary across ms
    later = new_id()
    assert earlier < later


@pytest.mark.fast
def test_is_valid_id_accepts_freshly_minted_id():
    assert is_valid_id(new_id()) is True


@pytest.mark.fast
@pytest.mark.parametrize(
    "bad",
    [
        "",
        "too-short",
        "x" * 25,                              # 25 chars
        "x" * 27,                              # 27 chars
        "01ARZ3NDEKTSV4RRFFQ69G5FA!",          # invalid char
        "01ARZ3NDEKTSV4RRFFQ69G5FAU",          # contains 'U' — Crockford excludes I,L,O,U
    ],
)
def test_is_valid_id_rejects_malformed(bad):
    assert is_valid_id(bad) is False


@pytest.mark.fast
def test_is_valid_id_handles_non_string():
    """Defensive: callers pass us values from JSON; non-strings are False, not raise."""
    assert is_valid_id(None) is False  # type: ignore[arg-type]
    assert is_valid_id(123) is False  # type: ignore[arg-type]
```

- [ ] **Step 2: Run test, expect fail**

Run: `uv run pytest tests/unit/core/test_ids.py -v`
Expected: ImportError on `ariadne_eval.core.ids`.

- [ ] **Step 3: Write the implementation**

```python
# src/ariadne_eval/core/ids.py
"""ULID-based time-sortable string IDs.

We use ``python-ulid`` (already pinned in pyproject.toml). ULIDs are 26-char
Crockford base32 strings, sortable lexicographically by the millisecond at
which they were minted. Crockford base32 excludes ``I``, ``L``, ``O``, and
``U`` to avoid visual ambiguity.
"""

from __future__ import annotations

from typing import Any, Final

from ulid import ULID

__all__ = ["is_valid_id", "new_id"]


_VALID_CHARS: Final[frozenset[str]] = frozenset(
    "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
)
"""Crockford base32 alphabet: digits + uppercase letters minus I, L, O, U."""


def new_id() -> str:
    """Mint a new time-sortable string ID."""
    return str(ULID())


def is_valid_id(value: Any) -> bool:
    """Return ``True`` iff ``value`` is a syntactically valid ULID string.

    Does not raise on non-string input; callers can pass JSON-decoded
    values directly without a type-check.
    """
    if not isinstance(value, str):
        return False
    if len(value) != 26:
        return False
    return all(ch in _VALID_CHARS for ch in value)
```

- [ ] **Step 4: Run test, expect pass**

Run: `uv run pytest tests/unit/core/test_ids.py -v`
Expected: 8 passed (4 base + 1 unique + 1 sortable + 6 parametrized rejection cases counted as one + 1 non-string).

(If `python-ulid` exposes the import as `from ulid import ULID`, the import
already works because the dep is pinned. No extra install step.)

- [ ] **Step 5: Commit**

```bash
git add src/ariadne_eval/core/ids.py tests/unit/core/test_ids.py
git commit -m "feat(core): add ULID-based new_id() and is_valid_id()"
```

---

## Task 4: JsonValue, TextBlock, ToolCallRef, Message

**Files:**
- Create: `src/ariadne_eval/core/trajectory.py` (initial; extended in later tasks)
- Test: `tests/unit/core/test_message.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/core/test_message.py
"""Message + content-block tests.

These pin the wire shape that the tracing layer (Phase 3) and the judge
(Phase 6) will rely on.
"""

import pytest
from pydantic import ValidationError

from ariadne_eval.core.trajectory import (
    Message,
    TextBlock,
    ToolCallRef,
)


@pytest.mark.fast
def test_text_block_round_trip():
    blk = TextBlock(text="hello")
    dumped = blk.model_dump()
    assert dumped == {"type": "text", "text": "hello"}
    assert TextBlock.model_validate(dumped) == blk


@pytest.mark.fast
def test_message_with_string_content():
    m = Message(role="user", content="hi there")
    assert m.role == "user"
    assert m.content == "hi there"
    assert m.tool_calls == []
    assert m.tool_call_id is None


@pytest.mark.fast
def test_message_with_block_list_content():
    m = Message(role="assistant", content=[TextBlock(text="hello")])
    dumped = m.model_dump()
    assert dumped["content"] == [{"type": "text", "text": "hello"}]


@pytest.mark.fast
def test_message_rejects_unknown_role():
    with pytest.raises(ValidationError):
        Message(role="banana", content="x")  # type: ignore[arg-type]


@pytest.mark.fast
def test_message_tool_call_id_only_with_tool_role():
    """tool_call_id is meaningful only when role == 'tool'."""
    Message(role="tool", content="result", tool_call_id="call_abc")  # ok
    with pytest.raises(ValidationError) as exc:
        Message(role="user", content="x", tool_call_id="call_abc")
    assert "tool_call_id" in str(exc.value)


@pytest.mark.fast
def test_tool_call_ref_round_trip():
    ref = ToolCallRef(id="call_abc", name="search", arguments={"q": "ariadne"})
    assert ref.model_dump() == {
        "id": "call_abc",
        "name": "search",
        "arguments": {"q": "ariadne"},
    }


@pytest.mark.fast
def test_message_with_assistant_tool_calls():
    m = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCallRef(id="call_1", name="search", arguments={"q": "x"}),
        ],
    )
    assert len(m.tool_calls) == 1
    assert m.tool_calls[0].name == "search"
```

- [ ] **Step 2: Run test, expect fail**

Run: `uv run pytest tests/unit/core/test_message.py -v`
Expected: ImportError on `ariadne_eval.core.trajectory`.

- [ ] **Step 3: Write the implementation**

```python
# src/ariadne_eval/core/trajectory.py
"""Trajectory data model: typed Pydantic v2 records for an agent run.

This file is the schema that every later phase reads. See
``docs/superpowers/specs/2026-05-10-trajectory-data-model-design.md`` for
the design rationale.
"""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "ContentBlock",
    "JsonValue",
    "Message",
    "TextBlock",
    "ToolCallRef",
]


# Recursive JSON-compatible value type. ``mypy`` accepts the forward
# string reference; Pydantic resolves it via ``model_rebuild`` if needed.
JsonValue = Union[
    str,
    int,
    float,
    bool,
    None,
    list["JsonValue"],
    dict[str, "JsonValue"],
]


class TextBlock(BaseModel):
    """A plain-text content block. Multimodal variants land in a future minor."""

    type: Literal["text"] = "text"
    text: str


# v0.0.2 ships text-only. Adding image / audio variants later is non-breaking
# because the field is already a ``BaseModel`` reference, not a bare ``str``.
ContentBlock = TextBlock


class ToolCallRef(BaseModel):
    """A tool-use directive emitted by an assistant message.

    Mirrors the shape used by Anthropic / OpenAI / litellm. The ``id`` is the
    provider-issued correlation token; tool-result messages reference it via
    ``Message.tool_call_id``.
    """

    id: str
    name: str
    arguments: dict[str, "JsonValue"]


class Message(BaseModel):
    """A single chat-completion message (system / user / assistant / tool)."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentBlock]
    tool_calls: list[ToolCallRef] = Field(default_factory=list)
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _tool_call_id_only_when_tool_role(self) -> "Message":
        if self.tool_call_id is not None and self.role != "tool":
            raise ValueError(
                "tool_call_id is only valid when role == 'tool'; "
                f"got role={self.role!r}"
            )
        return self
```

- [ ] **Step 4: Run test, expect pass**

Run: `uv run pytest tests/unit/core/test_message.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ariadne_eval/core/trajectory.py tests/unit/core/test_message.py
git commit -m "feat(core): add Message, TextBlock, ToolCallRef, JsonValue"
```

---

## Task 5: Payload variants and discriminated union

**Files:**
- Modify: `src/ariadne_eval/core/trajectory.py` (append payload classes)
- Test: `tests/unit/core/test_trajectory.py` (new file; just payload tests for now)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/core/test_trajectory.py
"""Payloads, Step, Trajectory, validators, truncation.

This file accumulates as later tasks add more behaviour. Each task adds
its tests to this file rather than spawning a new one.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from ariadne_eval.core.trajectory import (
    InternalPayload,
    LLMCallPayload,
    Message,
    Payload,
    ToolCallPayload,
    UserInputPayload,
)


@pytest.mark.fast
def test_llm_call_payload_minimal():
    p = LLMCallPayload(
        model_id="claude-sonnet",
        prompt_messages=[Message(role="user", content="hi")],
        completion="hello",
        input_tokens=10,
        output_tokens=2,
        cost_usd=0.0001,
        latency_ms=42.0,
    )
    assert p.step_type == "llm_call"
    assert p.completion_truncated is False
    assert p.tool_calls_emitted == []


@pytest.mark.fast
def test_tool_call_payload_minimal():
    p = ToolCallPayload(
        tool_name="search",
        arguments={"q": "ariadne"},
        result={"hits": 3},
        latency_ms=12.0,
    )
    assert p.step_type == "tool_call"
    assert p.result_truncated is False


@pytest.mark.fast
def test_user_input_payload_minimal():
    p = UserInputPayload(message="please continue")
    assert p.step_type == "user_input"
    assert p.channel is None


@pytest.mark.fast
def test_internal_payload_minimal():
    p = InternalPayload(kind="branch", data={"reason": "retry"})
    assert p.step_type == "internal"


@pytest.mark.fast
def test_payload_discriminator_resolves_correctly():
    """The discriminator must select the right variant on deserialization."""
    adapter = TypeAdapter(Payload)

    raw = {
        "step_type": "tool_call",
        "tool_name": "search",
        "arguments": {"q": "x"},
        "result": None,
        "latency_ms": 1.0,
    }
    inst = adapter.validate_python(raw)
    assert isinstance(inst, ToolCallPayload)


@pytest.mark.fast
def test_payload_discriminator_rejects_unknown_step_type():
    adapter = TypeAdapter(Payload)
    with pytest.raises(ValidationError):
        adapter.validate_python({"step_type": "made_up", "x": 1})
```

- [ ] **Step 2: Run test, expect fail**

Run: `uv run pytest tests/unit/core/test_trajectory.py -v`
Expected: ImportError on the payload classes.

- [ ] **Step 3: Append the payload classes to `trajectory.py`**

Append to `src/ariadne_eval/core/trajectory.py`:

```python
# === append below the existing Message class ===

from typing import Annotated  # add to existing typing import line


class LLMCallPayload(BaseModel):
    """Payload for an LLM call step."""

    step_type: Literal["llm_call"] = "llm_call"
    model_id: str
    prompt_messages: list[Message]
    completion: str
    completion_truncated: bool = False
    input_tokens: int
    output_tokens: int
    cost_usd: float
    temperature: float | None = None
    latency_ms: float
    ttft_ms: float | None = None
    tool_calls_emitted: list[str] = Field(default_factory=list)


class ToolCallPayload(BaseModel):
    """Payload for a tool execution step."""

    step_type: Literal["tool_call"] = "tool_call"
    tool_name: str
    arguments: dict[str, "JsonValue"]
    result: "JsonValue" = None
    result_truncated: bool = False
    latency_ms: float


class UserInputPayload(BaseModel):
    """Payload for an externally-supplied user input step."""

    step_type: Literal["user_input"] = "user_input"
    message: str
    channel: str | None = None


class InternalPayload(BaseModel):
    """Payload for an agent-internal step (branching, planning, bookkeeping)."""

    step_type: Literal["internal"] = "internal"
    kind: str
    data: "JsonValue" = None


Payload = Annotated[
    LLMCallPayload | ToolCallPayload | UserInputPayload | InternalPayload,
    Field(discriminator="step_type"),
]
```

Update the `__all__` list at the top to include the new symbols:

```python
__all__ = [
    "ContentBlock",
    "InternalPayload",
    "JsonValue",
    "LLMCallPayload",
    "Message",
    "Payload",
    "TextBlock",
    "ToolCallPayload",
    "ToolCallRef",
    "UserInputPayload",
]
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/unit/core -v`
Expected: previous tests still pass + 6 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ariadne_eval/core/trajectory.py tests/unit/core/test_trajectory.py
git commit -m "feat(core): add Payload variants and discriminated union"
```

---

## Task 6: StepError model

**Files:**
- Modify: `src/ariadne_eval/core/trajectory.py` (append StepError)
- Modify: `tests/unit/core/test_trajectory.py` (append error tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/core/test_trajectory.py`:

```python
from ariadne_eval.core.trajectory import StepError


@pytest.mark.fast
def test_step_error_minimal():
    e = StepError(type="TimeoutError", message="timed out after 30s")
    assert e.traceback is None


@pytest.mark.fast
def test_step_error_with_traceback():
    e = StepError(
        type="ValueError",
        message="bad input",
        traceback="Traceback (most recent call last):\n  ...\nValueError: bad input",
    )
    assert "Traceback" in (e.traceback or "")


@pytest.mark.fast
def test_step_error_round_trip():
    e = StepError(type="X", message="y")
    assert StepError.model_validate(e.model_dump()) == e
```

- [ ] **Step 2: Run test, expect fail**

Run: `uv run pytest tests/unit/core/test_trajectory.py -v`
Expected: ImportError on `StepError`.

- [ ] **Step 3: Append `StepError` to `trajectory.py`**

```python
class StepError(BaseModel):
    """Structured error attached to a failed step.

    ``traceback`` is opt-in (default ``None``). Tracing infrastructure
    populates it only when ``capture_tracebacks=True`` is set on the
    enclosing trajectory; production traces stay light by default.
    """

    type: str
    message: str
    traceback: str | None = None
```

Update `__all__` to add `"StepError"`.

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/unit/core -v`
Expected: 3 new + all previous pass.

- [ ] **Step 5: Commit**

```bash
git add src/ariadne_eval/core/trajectory.py tests/unit/core/test_trajectory.py
git commit -m "feat(core): add StepError model"
```

---

## Task 7: Step model with validators

**Files:**
- Modify: `src/ariadne_eval/core/trajectory.py` (append Step)
- Modify: `tests/unit/core/test_trajectory.py` (append Step tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/core/test_trajectory.py`:

```python
from datetime import UTC, datetime, timezone

from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import StepStatus
from ariadne_eval.core.trajectory import Step


def _ll() -> LLMCallPayload:
    return LLMCallPayload(
        model_id="m",
        prompt_messages=[Message(role="user", content="hi")],
        completion="hello",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        latency_ms=1.0,
    )


@pytest.mark.fast
def test_step_minimal_succeeded():
    sid, tid = new_id(), new_id()
    s = Step(
        id=sid,
        trajectory_id=tid,
        parent_step_id=None,
        name="ask llm",
        started_at=datetime.now(tz=UTC),
        finished_at=datetime.now(tz=UTC),
        status=StepStatus.SUCCEEDED,
        payload=_ll(),
    )
    assert s.error is None
    assert s.metadata == {}


@pytest.mark.fast
def test_step_rejects_naive_started_at():
    with pytest.raises(ValidationError) as exc:
        Step(
            id=new_id(),
            trajectory_id=new_id(),
            parent_step_id=None,
            name="x",
            started_at=datetime(2026, 5, 10, 12, 0, 0),  # naive
            finished_at=None,
            status=StepStatus.RUNNING,
            payload=_ll(),
        )
    assert "tz-aware" in str(exc.value).lower() or "timezone" in str(exc.value).lower()


@pytest.mark.fast
def test_step_rejects_naive_finished_at():
    with pytest.raises(ValidationError):
        Step(
            id=new_id(),
            trajectory_id=new_id(),
            parent_step_id=None,
            name="x",
            started_at=datetime.now(tz=UTC),
            finished_at=datetime(2026, 5, 10, 12, 0, 0),  # naive
            status=StepStatus.SUCCEEDED,
            payload=_ll(),
        )


@pytest.mark.fast
def test_step_rejects_self_parent():
    sid = new_id()
    with pytest.raises(ValidationError) as exc:
        Step(
            id=sid,
            trajectory_id=new_id(),
            parent_step_id=sid,
            name="x",
            started_at=datetime.now(tz=UTC),
            finished_at=None,
            status=StepStatus.RUNNING,
            payload=_ll(),
        )
    assert "self" in str(exc.value).lower() or "parent_step_id" in str(exc.value)


@pytest.mark.fast
def test_failed_step_requires_error():
    with pytest.raises(ValidationError) as exc:
        Step(
            id=new_id(),
            trajectory_id=new_id(),
            parent_step_id=None,
            name="x",
            started_at=datetime.now(tz=UTC),
            finished_at=datetime.now(tz=UTC),
            status=StepStatus.FAILED,
            payload=_ll(),
            error=None,  # <-- not allowed when status == failed
        )
    assert "error" in str(exc.value).lower()


@pytest.mark.fast
def test_step_round_trip_json():
    s = Step(
        id=new_id(),
        trajectory_id=new_id(),
        parent_step_id=None,
        name="x",
        started_at=datetime.now(tz=UTC),
        finished_at=None,
        status=StepStatus.RUNNING,
        payload=_ll(),
    )
    dumped = s.model_dump_json()
    rehydrated = Step.model_validate_json(dumped)
    assert rehydrated == s


@pytest.mark.fast
def test_step_id_must_be_valid_ulid():
    with pytest.raises(ValidationError):
        Step(
            id="not-a-ulid",
            trajectory_id=new_id(),
            parent_step_id=None,
            name="x",
            started_at=datetime.now(tz=UTC),
            finished_at=None,
            status=StepStatus.RUNNING,
            payload=_ll(),
        )
```

- [ ] **Step 2: Run tests, expect fail**

Run: `uv run pytest tests/unit/core/test_trajectory.py -v`
Expected: ImportError on `Step` (and the new validators).

- [ ] **Step 3: Append `Step` to `trajectory.py`**

Add the import for `datetime` at the top:

```python
from datetime import datetime
```

Then append:

```python
from ariadne_eval.core.ids import is_valid_id
from ariadne_eval.core.status import StepStatus, TrajectoryStatus

from pydantic import field_validator


def _require_tz_aware(dt: datetime | None, *, field: str) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        raise ValueError(f"{field} must be tz-aware (got naive datetime)")
    return dt


def _require_valid_id(value: str, *, field: str) -> str:
    if not is_valid_id(value):
        raise ValueError(f"{field} is not a valid ULID: {value!r}")
    return value


class Step(BaseModel):
    """One node in the trajectory tree."""

    id: str
    trajectory_id: str
    parent_step_id: str | None
    name: str
    started_at: datetime
    finished_at: datetime | None = None
    status: StepStatus
    payload: Payload
    error: StepError | None = None
    metadata: dict[str, "JsonValue"] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        return _require_valid_id(v, field="id")

    @field_validator("trajectory_id")
    @classmethod
    def _validate_trajectory_id(cls, v: str) -> str:
        return _require_valid_id(v, field="trajectory_id")

    @field_validator("parent_step_id")
    @classmethod
    def _validate_parent(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _require_valid_id(v, field="parent_step_id")

    @field_validator("started_at")
    @classmethod
    def _validate_started(cls, v: datetime) -> datetime:
        out = _require_tz_aware(v, field="started_at")
        assert out is not None  # mypy: started_at is non-None
        return out

    @field_validator("finished_at")
    @classmethod
    def _validate_finished(cls, v: datetime | None) -> datetime | None:
        return _require_tz_aware(v, field="finished_at")

    @model_validator(mode="after")
    def _no_self_parent(self) -> "Step":
        if self.parent_step_id is not None and self.parent_step_id == self.id:
            raise ValueError("parent_step_id cannot equal id (self-parenting)")
        return self

    @model_validator(mode="after")
    def _failed_requires_error(self) -> "Step":
        if self.status == StepStatus.FAILED and self.error is None:
            raise ValueError("status=failed requires error to be set")
        return self
```

Update `__all__` to add `"Step"`.

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/unit/core -v`
Expected: 7 new tests pass; previous tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/ariadne_eval/core/trajectory.py tests/unit/core/test_trajectory.py
git commit -m "feat(core): add Step model with id / datetime / status validators"
```

---

## Task 8: Trajectory model

**Files:**
- Modify: `src/ariadne_eval/core/trajectory.py` (append Trajectory)
- Modify: `tests/unit/core/test_trajectory.py` (append Trajectory tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/core/test_trajectory.py`:

```python
from ariadne_eval.core.trajectory import Trajectory


def _traj_minimal(**overrides) -> Trajectory:
    defaults = dict(
        id=new_id(),
        task="compute 2+2",
        agent_name="react",
        agent_version="0.1",
        model_id="claude-sonnet",
        started_at=datetime.now(tz=UTC),
        finished_at=None,
        final_status=TrajectoryStatus.RUNNING,
    )
    defaults.update(overrides)
    return Trajectory(**defaults)


@pytest.mark.fast
def test_trajectory_minimal():
    t = _traj_minimal()
    assert t.schema_version == 1
    assert t.metadata == {}
    assert t.final_answer is None
    assert t.root_step_id is None


@pytest.mark.fast
def test_trajectory_round_trip_json():
    t = _traj_minimal(finished_at=datetime.now(tz=UTC), final_status=TrajectoryStatus.SUCCEEDED, final_answer="42")
    rehydrated = Trajectory.model_validate_json(t.model_dump_json())
    assert rehydrated == t


@pytest.mark.fast
def test_trajectory_rejects_naive_started_at():
    with pytest.raises(ValidationError):
        _traj_minimal(started_at=datetime(2026, 5, 10, 12, 0, 0))


@pytest.mark.fast
def test_trajectory_id_must_be_valid_ulid():
    with pytest.raises(ValidationError):
        _traj_minimal(id="nope")


@pytest.mark.fast
def test_trajectory_root_step_id_validated_when_set():
    with pytest.raises(ValidationError):
        _traj_minimal(root_step_id="not-a-ulid")
```

- [ ] **Step 2: Run tests, expect fail**

Run: `uv run pytest tests/unit/core/test_trajectory.py -v`
Expected: ImportError on `Trajectory`.

- [ ] **Step 3: Append `Trajectory` to `trajectory.py`**

```python
class Trajectory(BaseModel):
    """An end-to-end agent run.

    The trajectory itself owns light metadata; the actual tree of steps is
    stored separately (each ``Step`` carries a ``trajectory_id`` foreign
    key). ``root_step_id`` is the entry point into the tree.
    """

    schema_version: int = 1
    id: str
    task: str
    agent_name: str
    agent_version: str
    model_id: str
    started_at: datetime
    finished_at: datetime | None = None
    final_status: TrajectoryStatus
    final_answer: "JsonValue" = None
    root_step_id: str | None = None
    metadata: dict[str, "JsonValue"] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        return _require_valid_id(v, field="id")

    @field_validator("root_step_id")
    @classmethod
    def _validate_root(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _require_valid_id(v, field="root_step_id")

    @field_validator("started_at")
    @classmethod
    def _validate_started(cls, v: datetime) -> datetime:
        out = _require_tz_aware(v, field="started_at")
        assert out is not None
        return out

    @field_validator("finished_at")
    @classmethod
    def _validate_finished(cls, v: datetime | None) -> datetime | None:
        return _require_tz_aware(v, field="finished_at")
```

Update `__all__` to add `"Trajectory"` and `"TrajectoryStatus"` and `"StepStatus"`
(re-exporting the enums from this module gives users one import path).

Add at the bottom of the module:

```python
# Re-export status enums so users can do `from ariadne_eval.core.trajectory import StepStatus`.
__all__ += ["StepStatus", "TrajectoryStatus"]
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/unit/core -v`
Expected: 5 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ariadne_eval/core/trajectory.py tests/unit/core/test_trajectory.py
git commit -m "feat(core): add Trajectory model with schema_version"
```

---

## Task 9: Truncation behaviour

**Files:**
- Modify: `src/ariadne_eval/core/trajectory.py`
- Modify: `tests/unit/core/test_trajectory.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/core/test_trajectory.py`:

```python
from ariadne_eval.core.trajectory import MAX_FIELD_CHARS


@pytest.mark.fast
def test_completion_under_limit_not_truncated():
    p = LLMCallPayload(
        model_id="m",
        prompt_messages=[Message(role="user", content="hi")],
        completion="x" * 100,
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        latency_ms=1.0,
    )
    assert p.completion == "x" * 100
    assert p.completion_truncated is False


@pytest.mark.fast
def test_completion_over_limit_truncated():
    too_long = "x" * (MAX_FIELD_CHARS + 100)
    p = LLMCallPayload(
        model_id="m",
        prompt_messages=[Message(role="user", content="hi")],
        completion=too_long,
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        latency_ms=1.0,
    )
    assert len(p.completion) == MAX_FIELD_CHARS
    assert p.completion_truncated is True


@pytest.mark.fast
def test_tool_result_under_limit_not_truncated():
    p = ToolCallPayload(
        tool_name="t",
        arguments={"x": 1},
        result={"a": "b"},
        latency_ms=1.0,
    )
    assert p.result == {"a": "b"}
    assert p.result_truncated is False


@pytest.mark.fast
def test_tool_result_over_limit_truncated_to_string():
    """When a structured result exceeds the cap, it is replaced with the
    JSON-prefix string and ``result_truncated`` is set."""
    huge = {"data": "x" * (MAX_FIELD_CHARS + 100)}
    p = ToolCallPayload(
        tool_name="t",
        arguments={"x": 1},
        result=huge,
        latency_ms=1.0,
    )
    assert p.result_truncated is True
    assert isinstance(p.result, str)
    assert len(p.result) == MAX_FIELD_CHARS
```

- [ ] **Step 2: Run tests, expect fail**

Run: `uv run pytest tests/unit/core/test_trajectory.py -v -k truncat`
Expected: import error on `MAX_FIELD_CHARS` or assertion failures.

- [ ] **Step 3: Add truncation to `trajectory.py`**

Near the top of `trajectory.py`, after the `JsonValue` definition:

```python
import json
from typing import Final

MAX_FIELD_CHARS: Final[int] = 65_536
"""Per-field truncation cap. See spec for rationale."""


def _truncate_str(value: str) -> tuple[str, bool]:
    """Return ``(possibly-truncated, was_truncated)``."""
    if len(value) > MAX_FIELD_CHARS:
        return value[:MAX_FIELD_CHARS], True
    return value, False


def _truncate_json_value(value: "JsonValue") -> tuple["JsonValue", bool]:
    """Return ``(possibly-degraded, was_truncated)``.

    If the JSON-serialized form fits, the original structure is preserved.
    Otherwise the value is replaced with the JSON-prefix string (a structural
    degradation that is documented in the spec).
    """
    serialized = json.dumps(value, default=str)
    if len(serialized) <= MAX_FIELD_CHARS:
        return value, False
    return serialized[:MAX_FIELD_CHARS], True
```

Update `LLMCallPayload` to apply truncation:

```python
class LLMCallPayload(BaseModel):
    step_type: Literal["llm_call"] = "llm_call"
    model_id: str
    prompt_messages: list[Message]
    completion: str
    completion_truncated: bool = False
    input_tokens: int
    output_tokens: int
    cost_usd: float
    temperature: float | None = None
    latency_ms: float
    ttft_ms: float | None = None
    tool_calls_emitted: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _truncate_completion(self) -> "LLMCallPayload":
        # Idempotent: re-running on an already-truncated value is a no-op.
        new_completion, was = _truncate_str(self.completion)
        if was:
            object.__setattr__(self, "completion", new_completion)
            object.__setattr__(self, "completion_truncated", True)
        return self
```

Update `ToolCallPayload` similarly:

```python
class ToolCallPayload(BaseModel):
    step_type: Literal["tool_call"] = "tool_call"
    tool_name: str
    arguments: dict[str, "JsonValue"]
    result: "JsonValue" = None
    result_truncated: bool = False
    latency_ms: float

    @model_validator(mode="after")
    def _truncate_result(self) -> "ToolCallPayload":
        new_result, was = _truncate_json_value(self.result)
        if was:
            object.__setattr__(self, "result", new_result)
            object.__setattr__(self, "result_truncated", True)
        return self
```

Update `__all__` to add `"MAX_FIELD_CHARS"`.

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/unit/core -v`
Expected: 4 new tests pass; previous tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/ariadne_eval/core/trajectory.py tests/unit/core/test_trajectory.py
git commit -m "feat(core): truncate completion and tool result above 64K chars"
```

---

## Task 10: Redact hook

**Files:**
- Modify: `src/ariadne_eval/core/trajectory.py` (add `Trajectory.redact`)
- Create: `tests/unit/core/test_redact.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/core/test_redact.py
"""Opt-in redact hook on Trajectory."""

from datetime import UTC, datetime

import pytest

from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import TrajectoryStatus
from ariadne_eval.core.trajectory import Trajectory


def _t() -> Trajectory:
    return Trajectory(
        id=new_id(),
        task="compute 2+2",
        agent_name="react",
        agent_version="0.1",
        model_id="claude-sonnet",
        started_at=datetime.now(tz=UTC),
        finished_at=None,
        final_status=TrajectoryStatus.RUNNING,
        metadata={"user_email": "alice@example.com"},
    )


@pytest.mark.fast
def test_redact_with_noop_returns_equal_copy():
    t = _t()
    redacted = t.redact(lambda x: x)
    assert redacted == t
    assert redacted is not t  # returns a new instance


@pytest.mark.fast
def test_redact_with_user_function_modifies_metadata():
    t = _t()

    def scrub(traj: Trajectory) -> Trajectory:
        new_meta = {**traj.metadata, "user_email": "[REDACTED]"}
        return traj.model_copy(update={"metadata": new_meta})

    redacted = t.redact(scrub)
    assert redacted.metadata["user_email"] == "[REDACTED]"
    assert t.metadata["user_email"] == "alice@example.com"  # original untouched
```

- [ ] **Step 2: Run test, expect fail**

Run: `uv run pytest tests/unit/core/test_redact.py -v`
Expected: AttributeError — `Trajectory.redact` not defined.

- [ ] **Step 3: Add the method**

Append inside `Trajectory` class in `trajectory.py`:

```python
    def redact(
        self,
        redactor: "Callable[[Trajectory], Trajectory]",
    ) -> "Trajectory":
        """Apply a user-supplied redactor and return a new Trajectory.

        The default behaviour of ``ariadne-eval`` is to preserve raw
        payloads. Privacy-sensitive consumers opt in by calling this hook
        with their own redactor. The hook never mutates the original.
        """
        return redactor(self.model_copy(deep=True))
```

Add at the top of the file (with other imports):

```python
from collections.abc import Callable
```

- [ ] **Step 4: Run test, expect pass**

Run: `uv run pytest tests/unit/core/test_redact.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ariadne_eval/core/trajectory.py tests/unit/core/test_redact.py
git commit -m "feat(core): add opt-in Trajectory.redact hook"
```

---

## Task 11: Property-based round-trip test

**Files:**
- Create: `tests/property/test_trajectory_roundtrip.py`

- [ ] **Step 1: Write the property test**

```python
# tests/property/test_trajectory_roundtrip.py
"""Property-based round-trip: any Trajectory we can construct must serialize
and deserialize to an equal value."""

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import StepStatus, TrajectoryStatus
from ariadne_eval.core.trajectory import (
    InternalPayload,
    LLMCallPayload,
    Message,
    Step,
    ToolCallPayload,
    Trajectory,
    UserInputPayload,
)


_BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


@st.composite
def _datetimes(draw):
    seconds = draw(st.integers(min_value=0, max_value=10_000_000))
    return _BASE_TIME + timedelta(seconds=seconds)


@st.composite
def _llm_payloads(draw):
    return LLMCallPayload(
        model_id=draw(st.sampled_from(["claude-sonnet", "gpt-4o", "haiku"])),
        prompt_messages=[Message(role="user", content=draw(st.text(max_size=64)))],
        completion=draw(st.text(max_size=128)),
        input_tokens=draw(st.integers(min_value=0, max_value=10_000)),
        output_tokens=draw(st.integers(min_value=0, max_value=10_000)),
        cost_usd=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        latency_ms=draw(st.floats(min_value=0.0, max_value=5_000.0, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def _tool_payloads(draw):
    return ToolCallPayload(
        tool_name=draw(st.sampled_from(["search", "calculator", "fetch"])),
        arguments={"q": draw(st.text(max_size=32))},
        result=draw(st.one_of(st.none(), st.integers(), st.text(max_size=64))),
        latency_ms=draw(st.floats(min_value=0.0, max_value=5_000.0, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def _user_payloads(draw):
    return UserInputPayload(message=draw(st.text(max_size=64)))


@st.composite
def _internal_payloads(draw):
    return InternalPayload(kind=draw(st.text(min_size=1, max_size=16, alphabet="abcdef")))


_payloads = st.one_of(_llm_payloads(), _tool_payloads(), _user_payloads(), _internal_payloads())


@st.composite
def _steps(draw):
    started = draw(_datetimes())
    return Step(
        id=new_id(),
        trajectory_id=new_id(),
        parent_step_id=None,
        name=draw(st.text(min_size=1, max_size=24)),
        started_at=started,
        finished_at=started + timedelta(milliseconds=draw(st.integers(min_value=0, max_value=10_000))),
        status=StepStatus.SUCCEEDED,
        payload=draw(_payloads),
    )


@st.composite
def _trajectories(draw):
    started = draw(_datetimes())
    return Trajectory(
        id=new_id(),
        task=draw(st.text(min_size=1, max_size=64)),
        agent_name=draw(st.sampled_from(["react", "tool-use"])),
        agent_version=draw(st.sampled_from(["0.1", "0.2"])),
        model_id=draw(st.sampled_from(["claude-sonnet", "gpt-4o"])),
        started_at=started,
        finished_at=started + timedelta(seconds=draw(st.integers(min_value=0, max_value=600))),
        final_status=draw(st.sampled_from(list(TrajectoryStatus))),
        final_answer=draw(st.one_of(st.none(), st.text(max_size=64))),
        metadata={"k": draw(st.text(max_size=16))},
    )


@pytest.mark.fast
@given(t=_trajectories())
@settings(max_examples=200, deadline=None)
def test_trajectory_json_round_trip(t):
    rehydrated = Trajectory.model_validate_json(t.model_dump_json())
    assert rehydrated == t


@pytest.mark.fast
@given(s=_steps())
@settings(max_examples=200, deadline=None)
def test_step_json_round_trip(s):
    rehydrated = Step.model_validate_json(s.model_dump_json())
    assert rehydrated == s
```

- [ ] **Step 2: Run test, expect pass**

Run: `uv run pytest tests/property -v`
Expected: 2 tests pass; each runs 200 hypothesis examples.

- [ ] **Step 3: Commit**

```bash
git add tests/property/test_trajectory_roundtrip.py
git commit -m "test(core): hypothesis round-trip for Trajectory and Step"
```

---

## Task 12: Public API surface

**Files:**
- Modify: `src/ariadne_eval/__init__.py`
- Modify: `tests/unit/test_smoke.py` (extend)

- [ ] **Step 1: Extend the smoke test**

Append to `tests/unit/test_smoke.py`:

```python
@pytest.mark.fast
def test_public_api_exports_core_types():
    """Pin the public surface so accidental removals are caught early."""
    import ariadne_eval

    expected = {
        "__version__",
        "Trajectory", "Step", "Message", "ContentBlock", "TextBlock",
        "ToolCallRef",
        "LLMCallPayload", "ToolCallPayload", "UserInputPayload", "InternalPayload",
        "StepError", "StepStatus", "TrajectoryStatus", "JsonValue",
        "new_id", "is_valid_id",
    }
    missing = expected - set(ariadne_eval.__all__)
    assert not missing, f"Missing from public API: {missing}"
    for name in expected:
        assert hasattr(ariadne_eval, name), f"ariadne_eval.{name} not importable"
```

- [ ] **Step 2: Run test, expect fail**

Run: `uv run pytest tests/unit/test_smoke.py::test_public_api_exports_core_types -v`
Expected: assertion error listing missing symbols.

- [ ] **Step 3: Update `src/ariadne_eval/__init__.py`**

```python
"""ariadne-eval: trajectory-level observability and evaluation for LLM agents.

The public API is intentionally small. Every symbol re-exported here is part
of the supported surface; everything else is private and may change without
warning. See ``docs/reference/`` for the full reference.
"""

from __future__ import annotations

from ariadne_eval._version import __version__
from ariadne_eval.core.ids import is_valid_id, new_id
from ariadne_eval.core.status import StepStatus, TrajectoryStatus
from ariadne_eval.core.trajectory import (
    ContentBlock,
    InternalPayload,
    JsonValue,
    LLMCallPayload,
    Message,
    Step,
    StepError,
    TextBlock,
    ToolCallPayload,
    ToolCallRef,
    Trajectory,
    UserInputPayload,
)

__all__ = [
    "__version__",
    "ContentBlock",
    "InternalPayload",
    "JsonValue",
    "LLMCallPayload",
    "Message",
    "Step",
    "StepError",
    "StepStatus",
    "TextBlock",
    "ToolCallPayload",
    "ToolCallRef",
    "Trajectory",
    "TrajectoryStatus",
    "UserInputPayload",
    "is_valid_id",
    "new_id",
]
```

- [ ] **Step 4: Run all tests, expect pass**

Run: `uv run pytest -m fast` (default selection, no path)
Expected: every test passes including the smoke API test and all phase-1 tests.

- [ ] **Step 5: Run mypy**

Run: `uv run mypy --strict`
Expected: `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add src/ariadne_eval/__init__.py tests/unit/test_smoke.py
git commit -m "feat: re-export trajectory data model from the package root"
```

---

## Task 13: Concept documentation

**Files:**
- Create: `docs/concepts/trajectory.md`
- Modify: `mkdocs.yml` (add to nav)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write the concept doc**

```markdown
# The trajectory model

A *trajectory* is one end-to-end agent run on one task. ariadne-eval
represents it as a tree of typed steps:

```
Trajectory(id=01J..., task="compute 17*23 / len('banana')")
└── Step name="ask_llm" payload=LLMCallPayload
    ├── Step name="calculator" payload=ToolCallPayload(args={"expr":"17*23"})
    │   └── result: 391
    └── Step name="ask_llm" payload=LLMCallPayload
        └── final_answer: "65.166..."
```

Every node is a `Step` with a string ID, a `parent_step_id` (null at the
root), a status, timestamps, and a typed `payload` chosen from a
discriminated union over four `step_type` values:

| `step_type` | Payload | When to use |
|---|---|---|
| `llm_call` | `LLMCallPayload` | Any single chat-completion call. |
| `tool_call` | `ToolCallPayload` | A tool execution (function call, API hit, etc.). |
| `user_input` | `UserInputPayload` | An external prompt mid-run. |
| `internal` | `InternalPayload` | Bookkeeping, planning, or branching. |

The trajectory itself owns light metadata (`task`, `agent_name`,
`agent_version`, `model_id`, `final_status`, `final_answer`,
`schema_version`) and delegates the actual run history to its tree of
steps.

## Design decisions

- **Tree, not DAG.** Re-entry is rare; when needed, model it with explicit
  `internal` "branch" steps. v0.1.x will not introduce DAG semantics.
- **String parent references.** `parent_step_id` is a string ID, not an
  embedded `Step`, so JSON serialization is acyclic by construction.
- **ULID IDs.** Time-sortable, lexicographic ordering matches construction
  order to the millisecond.
- **tz-aware UTC datetimes.** Naive datetimes are rejected at construction.
- **Truncation.** The two payload fields with explicit `*_truncated` flags
  (`completion` and `result`) are capped at 64K characters. Other fields
  are stored as-is; consumer code that worries about size opens its own
  cap.
- **Opt-in redaction.** Raw prompts and completions are kept unless the
  user explicitly applies a `redact()` hook.

## Public types

```python
from ariadne_eval import (
    Trajectory, Step, Message,
    LLMCallPayload, ToolCallPayload,
    UserInputPayload, InternalPayload,
    StepError, StepStatus, TrajectoryStatus,
    new_id,
)
```

See [API Reference](../reference/index.md) for full field-by-field
documentation generated from the source.
```

(The triple-backtick-fenced ASCII tree may need an outer backtick wrapper —
include the inner block exactly as shown so mkdocs renders it inside a
plain text block.)

- [ ] **Step 2: Update `mkdocs.yml` nav**

Modify the `Concepts` section in `mkdocs.yml`:

```yaml
  - Concepts:
      - concepts/index.md
      - Trajectory model: concepts/trajectory.md
```

- [ ] **Step 3: Append CHANGELOG entry**

Modify `CHANGELOG.md` under `## [Unreleased]`:

```markdown
## [Unreleased]

### Added
- Core trajectory data model: `Trajectory`, `Step`, `Message`, four payload
  variants (`LLMCallPayload`, `ToolCallPayload`, `UserInputPayload`,
  `InternalPayload`), `StepError`, `StepStatus`, `TrajectoryStatus`,
  `JsonValue`, `new_id`, `is_valid_id`. Validators: tz-aware datetimes,
  ULID format, no self-parenting, failed-step requires error. Truncation on
  `completion` and `result` above 64K chars. Opt-in `Trajectory.redact()`
  hook. Hypothesis round-trip property tests.
```

- [ ] **Step 4: Verify docs build**

Run: `uv run mkdocs build --strict`
Expected: clean build (no warnings).

- [ ] **Step 5: Commit**

```bash
git add docs/concepts/trajectory.md mkdocs.yml CHANGELOG.md
git commit -m "docs(concepts): add trajectory model concept page"
```

---

## Task 14: Final verification

**Files:** none (verification only).

- [ ] **Step 1: All tests pass**

Run: `uv run pytest -m fast`
Expected: every test passes; total > 30; no warnings (filterwarnings is `error`).

- [ ] **Step 2: Coverage > 95% on touched files**

Run: `uv run pytest -m fast --cov=src/ariadne_eval/core --cov-report=term-missing`
Expected: every file in `src/ariadne_eval/core/` shows ≥ 95 %.

- [ ] **Step 3: mypy strict**

Run: `uv run mypy --strict`
Expected: `Success: no issues found`.

- [ ] **Step 4: ruff clean**

Run: `uv run ruff check && uv run ruff format --check`
Expected: both green.

- [ ] **Step 5: Pre-commit clean**

Run: `uv run pre-commit run --all-files`
Expected: all hooks pass.

- [ ] **Step 6: Tag the phase**

```bash
git tag v0.0.2-alpha -m "Phase 1: trajectory data model"
```

(Push of the tag to GitHub is deferred to whenever the user pushes the
branch upstream.)

---

## Self-review

**Spec coverage check:**

| Spec section | Task |
|---|---|
| `core.ids` module | Task 3 |
| `core.status` module | Task 2 |
| `core.trajectory` module | Tasks 4–10 |
| `Message` typed model | Task 4 |
| Discriminated `Payload` union | Task 5 |
| `StepError` model | Task 6 |
| `Step` with tz-aware / ULID / self-parent / error-required validators | Task 7 |
| `Trajectory` with `schema_version` | Task 8 |
| Truncation on `completion` and `result` | Task 9 |
| Opt-in `redact()` hook | Task 10 |
| Property test (Hypothesis ≥ 200 examples) | Task 11 |
| Public API re-exports | Task 12 |
| Concept doc with diagram | Task 13 |

All sections covered.

**Type consistency check:** `LLMCallPayload`, `ToolCallPayload`, etc. are
referenced by name across tasks 4–9 and 11–12; spelling matches throughout.

**Placeholder scan:** no TBD / TODO / "implement later" markers in this plan.
