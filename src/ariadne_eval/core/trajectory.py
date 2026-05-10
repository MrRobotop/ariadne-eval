"""Trajectory data model: typed Pydantic v2 records for an agent run.

This file is the schema that every later phase reads. See
``docs/superpowers/specs/2026-05-10-trajectory-data-model-design.md`` for
the design rationale.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Final, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import TypeAliasType

from ariadne_eval.core.ids import is_valid_id
from ariadne_eval.core.status import StepStatus, TrajectoryStatus

__all__ = [
    "MAX_FIELD_CHARS",
    "ContentBlock",
    "InternalPayload",
    "JsonValue",
    "LLMCallPayload",
    "Message",
    "Payload",
    "Step",
    "StepError",
    "StepStatus",
    "TextBlock",
    "ToolCallPayload",
    "ToolCallRef",
    "Trajectory",
    "TrajectoryStatus",
    "UserInputPayload",
]


MAX_FIELD_CHARS: Final[int] = 65_536
"""Per-field truncation cap, applied to ``LLMCallPayload.completion`` and
``ToolCallPayload.result`` only. See the design spec for rationale."""


def _truncate_str(value: str) -> tuple[str, bool]:
    """Return ``(possibly-truncated, was_truncated)``."""
    if len(value) > MAX_FIELD_CHARS:
        return value[:MAX_FIELD_CHARS], True
    return value, False


def _truncate_json_value(value: object) -> tuple[object, bool]:
    """Return ``(possibly-degraded, was_truncated)``.

    If the JSON-serialized form fits, the original structure is preserved.
    Otherwise the value is replaced with the JSON-prefix string (a structural
    degradation that is documented in the spec).
    """
    serialized = json.dumps(value, default=str)
    if len(serialized) <= MAX_FIELD_CHARS:
        return value, False
    return serialized[:MAX_FIELD_CHARS], True


# Recursive JSON-compatible value type. Using ``TypeAliasType`` (PEP 695 /
# typing_extensions backport) is the Pydantic-recommended way to declare a
# recursive alias: it gives Pydantic a stable handle for the recursion so
# the schema generator does not unfold forever.
JsonValue = TypeAliasType(
    "JsonValue",
    "str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]",
)


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
    arguments: dict[str, JsonValue]


class Message(BaseModel):
    """A single chat-completion message (system / user / assistant / tool)."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentBlock]
    tool_calls: list[ToolCallRef] = Field(default_factory=list)
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _tool_call_id_only_when_tool_role(self) -> Message:
        if self.tool_call_id is not None and self.role != "tool":
            raise ValueError(
                f"tool_call_id is only valid when role == 'tool'; got role={self.role!r}"
            )
        return self


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

    @model_validator(mode="after")
    def _truncate_completion(self) -> LLMCallPayload:
        new_completion, was = _truncate_str(self.completion)
        if was:
            object.__setattr__(self, "completion", new_completion)
            object.__setattr__(self, "completion_truncated", True)
        return self


class ToolCallPayload(BaseModel):
    """Payload for a tool execution step."""

    step_type: Literal["tool_call"] = "tool_call"
    tool_name: str
    arguments: dict[str, JsonValue]
    result: JsonValue = None
    result_truncated: bool = False
    latency_ms: float

    @model_validator(mode="after")
    def _truncate_result(self) -> ToolCallPayload:
        new_result, was = _truncate_json_value(self.result)
        if was:
            object.__setattr__(self, "result", new_result)
            object.__setattr__(self, "result_truncated", True)
        return self


class UserInputPayload(BaseModel):
    """Payload for an externally-supplied user input step."""

    step_type: Literal["user_input"] = "user_input"
    message: str
    channel: str | None = None


class InternalPayload(BaseModel):
    """Payload for an agent-internal step (branching, planning, bookkeeping)."""

    step_type: Literal["internal"] = "internal"
    kind: str
    data: JsonValue = None


Payload = Annotated[
    LLMCallPayload | ToolCallPayload | UserInputPayload | InternalPayload,
    Field(discriminator="step_type"),
]


class StepError(BaseModel):
    """Structured error attached to a failed step.

    ``traceback`` is opt-in (default ``None``). Tracing infrastructure
    populates it only when ``capture_tracebacks=True`` is set on the
    enclosing trajectory; production traces stay light by default.
    """

    type: str
    message: str
    traceback: str | None = None


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
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

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
        if v.tzinfo is None:
            raise ValueError("started_at must be tz-aware (got naive datetime)")
        return v

    @field_validator("finished_at")
    @classmethod
    def _validate_finished(cls, v: datetime | None) -> datetime | None:
        return _require_tz_aware(v, field="finished_at")

    @model_validator(mode="after")
    def _no_self_parent(self) -> Step:
        if self.parent_step_id is not None and self.parent_step_id == self.id:
            raise ValueError("parent_step_id cannot equal id (self-parenting)")
        return self

    @model_validator(mode="after")
    def _failed_requires_error(self) -> Step:
        if self.status == StepStatus.FAILED and self.error is None:
            raise ValueError("status=failed requires error to be set")
        return self


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
    final_answer: JsonValue = None
    root_step_id: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

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
        if v.tzinfo is None:
            raise ValueError("started_at must be tz-aware (got naive datetime)")
        return v

    @field_validator("finished_at")
    @classmethod
    def _validate_finished(cls, v: datetime | None) -> datetime | None:
        return _require_tz_aware(v, field="finished_at")

    def redact(
        self,
        redactor: Callable[[Trajectory], Trajectory],
    ) -> Trajectory:
        """Apply a user-supplied redactor and return a new Trajectory.

        The default behaviour of ``ariadne-eval`` is to preserve raw
        payloads. Privacy-sensitive consumers opt in by calling this hook
        with their own redactor. The hook never mutates the original.
        """
        return redactor(self.model_copy(deep=True))
