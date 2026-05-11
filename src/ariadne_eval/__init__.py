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
from ariadne_eval.storage.base import (
    MetadataTooLargeError,
    Store,
    StoreError,
    TrajectoryNotFoundError,
)
from ariadne_eval.storage.duckdb_store import DuckDBStore
from ariadne_eval.storage.jsonl_store import export_jsonl, import_jsonl

__all__ = [
    "ContentBlock",
    "DuckDBStore",
    "InternalPayload",
    "JsonValue",
    "LLMCallPayload",
    "Message",
    "MetadataTooLargeError",
    "Step",
    "StepError",
    "StepStatus",
    "Store",
    "StoreError",
    "TextBlock",
    "ToolCallPayload",
    "ToolCallRef",
    "Trajectory",
    "TrajectoryNotFoundError",
    "TrajectoryStatus",
    "UserInputPayload",
    "__version__",
    "export_jsonl",
    "import_jsonl",
    "is_valid_id",
    "new_id",
]
