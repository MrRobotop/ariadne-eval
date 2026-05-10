"""Trajectory data model: typed Pydantic v2 records for an agent run.

This file is the schema that every later phase reads. See
``docs/superpowers/specs/2026-05-10-trajectory-data-model-design.md`` for
the design rationale.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypeAliasType

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
