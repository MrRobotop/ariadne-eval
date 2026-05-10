# Phase 1 Design: Core Trajectory Data Model

**Status:** Approved (2026-05-10) · **Phase:** 1 · **Target version:** 0.0.2

## Goal

Define the Pydantic v2 models that represent an end-to-end agent run as a tree
of typed steps. Every downstream component — storage, tracing, metrics,
judges, UI — reads and writes these types. Getting the shape right here is the
single most important design decision in the project.

## Scope

In scope: data classes, ID generation, status enums, JSON round-trip,
truncation, naive-datetime rejection, an opt-in redact hook, and the test
suite that proves all of the above.

Out of scope: storage, tracing decorators, metrics, the UI. Those are later
phases that consume the types defined here.

## Architecture

Three modules under `src/ariadne_eval/core/`:

| Module | Public surface | Concern |
|---|---|---|
| `core.ids` | `new_id()`, `is_valid_id()` | Wraps `python-ulid`; provides time-sortable string IDs. |
| `core.status` | `StepStatus`, `TrajectoryStatus` | Enums with stable string values (part of the public API). |
| `core.trajectory` | `Trajectory`, `Step`, `Message`, `ContentBlock`, payload variants, `StepError`, `JsonValue` | Pydantic v2 models with a discriminated `Payload` union over `step_type`. |

The trajectory is a **tree, not a DAG**, in v0.1. `Step.parent_step_id: str | None`
keeps serialization acyclic; readers reconstruct the tree from parent links.
Re-entry / branching is a future concern modelled with explicit `internal`
"branch" steps when needed.

## Type definitions

```python
JsonValue = (
    str
    | int
    | float
    | bool
    | None
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


# v0.0.2 ships a single ContentBlock variant. When multimodal consumers
# appear, this becomes:
#     ContentBlock = Annotated[
#         TextBlock | ImageBlock | AudioBlock,
#         Field(discriminator="type"),
#     ]
# Adding variants is non-breaking; renaming or removing TextBlock is.
ContentBlock = TextBlock


class ToolCallRef(BaseModel):
    """A tool-use directive emitted by an LLM (mirrors Anthropic / OpenAI shape)."""

    id: str
    name: str
    arguments: dict[str, JsonValue]


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentBlock]
    tool_calls: list[ToolCallRef] = []
    tool_call_id: str | None = None  # set when role == "tool"


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
    ttft_ms: float | None = None  # streaming first-token offset
    tool_calls_emitted: list[str] = []  # child ToolCall step IDs


class ToolCallPayload(BaseModel):
    step_type: Literal["tool_call"] = "tool_call"
    tool_name: str
    arguments: dict[str, JsonValue]
    result: JsonValue = None
    result_truncated: bool = False
    latency_ms: float


class UserInputPayload(BaseModel):
    step_type: Literal["user_input"] = "user_input"
    message: str
    channel: str | None = None


class InternalPayload(BaseModel):
    step_type: Literal["internal"] = "internal"
    kind: str
    data: JsonValue = None


Payload = Annotated[
    LLMCallPayload | ToolCallPayload | UserInputPayload | InternalPayload,
    Field(discriminator="step_type"),
]


class StepError(BaseModel):
    type: str
    message: str
    traceback: str | None = None  # opt-in capture; default off in production


class Step(BaseModel):
    id: str
    trajectory_id: str
    parent_step_id: str | None
    name: str
    started_at: datetime  # tz-aware UTC, validated
    finished_at: datetime | None = None
    status: StepStatus
    payload: Payload
    error: StepError | None = None
    metadata: dict[str, JsonValue] = {}


class Trajectory(BaseModel):
    schema_version: int = 1
    id: str
    task: str
    agent_name: str
    agent_version: str
    model_id: str
    started_at: datetime
    finished_at: datetime | None = None
    final_status: TrajectoryStatus
    final_answer: JsonValue = None
    root_step_id: str | None = None
    metadata: dict[str, JsonValue] = {}
```

### Status enums

```python
class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class TrajectoryStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"
```

String values are part of the public API. Renaming any of them is a major
version bump.

## Cross-cutting behaviour

### Truncation

A module-level constant:

```python
MAX_FIELD_CHARS = 65_536  # configurable per-trajectory in a later phase
```

Truncation is applied at construction time, only on the two fields that carry
their own `*_truncated` flag:

- `LLMCallPayload.completion`. If `len(s) > MAX_FIELD_CHARS`, the string is
  replaced with `s[:MAX_FIELD_CHARS]` and `completion_truncated` is set
  to `True`.
- `ToolCallPayload.result`. If `len(json.dumps(value, default=str))
  > MAX_FIELD_CHARS`, the value is replaced with the prefix of its JSON
  serialization (a `str`) and `result_truncated` is set to `True`. Note that
  truncation degrades a structured `JsonValue` to a string — this is an
  intentional trade-off for v0.0.2.

`prompt_messages` are stored as-is in v0.0.2. They have no per-message
truncation flag, and most production prompts are well under the cap. If a
real producer regularly emits oversized prompts, we add a per-message flag
in a future minor release (non-breaking addition).

Truncation is measured by character count: `len(str)` for the completion case
and `len(json.dumps(...))` for the result case. Deterministic and easy to
explain; encoding-agnostic.

### Datetime tz-awareness

Every datetime field is wrapped in a Pydantic validator that raises
`ValueError("datetime must be tz-aware UTC")` if `dt.tzinfo is None`. Internal
code uses `datetime.now(tz=timezone.utc)` exclusively.

Serialization is ISO 8601 with offset (`2026-05-10T18:21:00+00:00`).

### Redact hook

```python
def redact(self, redactor: Callable[[Trajectory], Trajectory]) -> Trajectory:
    """Return a new trajectory with the user-supplied redactor applied."""
```

Default behaviour: nothing is auto-redacted. Privacy-sensitive fields
(`prompt_messages`, `completion`, `result`, `metadata`) remain raw unless the
user explicitly invokes `redact()`. This is the conservative default per the
project's "never log raw prompts by default" rule — opt-in over opt-out.

### IDs

`core.ids.new_id() -> str` returns `str(ulid.ULID())`. ULIDs are 26-char
Crockford-base32 strings, time-sortable to millisecond precision. Their
sortability lets us property-test that ID order matches `started_at` order
within a trajectory.

`core.ids.is_valid_id(s: str) -> bool` validates the format (length 26,
character set) without raising. Used in field validators on `Step.id`,
`Step.trajectory_id`, `Step.parent_step_id`, and `Trajectory.id`.

## Edge cases handled at construction

1. **Tool call failure** — when `Step.status == "failed"`, `Step.error` MUST be
   populated; `ToolCallPayload.result` MAY be `None`. Enforced by a
   model-level validator on `Step`.
2. **Streaming responses** — `ttft_ms` is the offset from `started_at` of the
   first token. `completion` is the final assembled text; partial chunks are
   not stored.
3. **Naive datetimes** — rejected with a clear error message naming the field.
4. **Cycles** — impossible by construction; `parent_step_id` is a string
   reference, never an embedded `Step`.
5. **Self-parent** — rejected: `Step.parent_step_id` cannot equal `Step.id`.

## Public API surface

`ariadne_eval/__init__.py` re-exports:

```python
__all__ = [
    "__version__",
    "Trajectory",
    "Step",
    "Message",
    "ContentBlock",
    "TextBlock",
    "ToolCallRef",
    "LLMCallPayload",
    "ToolCallPayload",
    "UserInputPayload",
    "InternalPayload",
    "StepError",
    "StepStatus",
    "TrajectoryStatus",
    "JsonValue",
    "new_id",
    "is_valid_id",
]
```

Anything not in `__all__` is private and may change without warning.

## Testing strategy (TDD)

Tests are written **before** implementation, RED → GREEN → REFACTOR. Six
files:

| File | Coverage |
|---|---|
| `tests/unit/core/test_ids.py` | ULID validity, monotonic sort within ms, uniqueness over 10k samples, `is_valid_id` true/false table. |
| `tests/unit/core/test_status.py` | Stable string values; importable from public API. |
| `tests/unit/core/test_message.py` | Role/content variants; tool_call_id only valid with role="tool" (model-level validator). |
| `tests/unit/core/test_trajectory.py` | Round-trip JSON each model w/ and w/o optionals; discriminated-union resolution; naive-datetime rejection; truncation behaviour; failed-step / error-required interaction; self-parent rejection. |
| `tests/unit/core/test_redact.py` | Default noop preserves equality; user redactor applied to all known sensitive fields. |
| `tests/property/test_trajectory_roundtrip.py` | Hypothesis: generate random trajectories, serialize, deserialize, deep equality. ≥200 examples; `deadline=None` to tolerate slow validators. |

**Coverage target:** >95 % on touched files (measured by `pytest --cov`).

**Type strictness:** `mypy --strict` clean on `src/ariadne_eval/core/`.

## Documentation

- `docs/concepts/trajectory.md` — narrative explainer with an ASCII tree
  diagram of a small example trajectory. Targets an end-user reading the docs
  for the first time. Generated as part of this phase, not deferred.
- API reference auto-generated by `mkdocstrings` from the public docstrings.

## Out of scope

- Storage. Phase 2.
- Tracing decorators. Phase 3.
- Streaming partial chunks. Future phase if a use case appears.
- Multimodal content. The `ContentBlock` discriminator leaves room; new
  variants are non-breaking additions.
- Cost recomputation from token counts. v0.0.2 trusts whatever litellm
  returned; recomputation is a Phase 5 concern.
