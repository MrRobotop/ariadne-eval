"""ariadne-eval: trajectory-level observability and evaluation for LLM agents.

The public API is intentionally small. Every symbol re-exported here is part
of the supported surface; everything else is private and may change without
warning. See ``docs/reference/`` for the full reference.
"""

from __future__ import annotations

from ariadne_eval._version import __version__
from ariadne_eval.adapters.litellm import enable_litellm_autotrace
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
from ariadne_eval.eval.case import Case, ExpectedTool
from ariadne_eval.eval.errors import (
    BootstrapInsufficientDataWarning,
    MissingReferenceError,
)
from ariadne_eval.eval.metrics.base import Metric, MetricResult
from ariadne_eval.eval.metrics.efficiency import StepEfficiency
from ariadne_eval.eval.metrics.final_answer import FinalAnswerMatch
from ariadne_eval.eval.metrics.tool_accuracy import ToolAccuracy
from ariadne_eval.eval.runner import EvalReport, Runner
from ariadne_eval.eval.stats.bootstrap import BootstrapCI, bootstrap_mean_ci
from ariadne_eval.storage.base import (
    MetadataTooLargeError,
    Store,
    StoreError,
    TrajectoryNotFoundError,
)
from ariadne_eval.storage.duckdb_store import DuckDBStore
from ariadne_eval.storage.jsonl_store import export_jsonl, import_jsonl
from ariadne_eval.tracing._fail_mode import FailMode, UnattachedTracingWarning
from ariadne_eval.tracing.context import (
    TrajectoryHandle,
    current_step,
    current_trajectory,
    start_trajectory,
)
from ariadne_eval.tracing.decorator import (
    record_llm_call,
    record_tool_call,
    trace_step,
)
from ariadne_eval.tracing.sampler import (
    AlwaysSampler,
    RateSampler,
    Sampler,
    TaskFilterSampler,
)

__all__ = [
    "AlwaysSampler",
    "BootstrapCI",
    "BootstrapInsufficientDataWarning",
    "Case",
    "ContentBlock",
    "DuckDBStore",
    "EvalReport",
    "ExpectedTool",
    "FailMode",
    "FinalAnswerMatch",
    "InternalPayload",
    "JsonValue",
    "LLMCallPayload",
    "Message",
    "MetadataTooLargeError",
    "Metric",
    "MetricResult",
    "MissingReferenceError",
    "RateSampler",
    "Runner",
    "Sampler",
    "Step",
    "StepEfficiency",
    "StepError",
    "StepStatus",
    "Store",
    "StoreError",
    "TaskFilterSampler",
    "TextBlock",
    "ToolAccuracy",
    "ToolCallPayload",
    "ToolCallRef",
    "Trajectory",
    "TrajectoryHandle",
    "TrajectoryNotFoundError",
    "TrajectoryStatus",
    "UnattachedTracingWarning",
    "UserInputPayload",
    "__version__",
    "bootstrap_mean_ci",
    "current_step",
    "current_trajectory",
    "enable_litellm_autotrace",
    "export_jsonl",
    "import_jsonl",
    "is_valid_id",
    "new_id",
    "record_llm_call",
    "record_tool_call",
    "start_trajectory",
    "trace_step",
]
