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
    "__version__",
    "is_valid_id",
    "new_id",
]
