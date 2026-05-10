# The trajectory model

A *trajectory* is one end-to-end agent run on one task. ariadne-eval
represents it as a tree of typed steps:

```text
Trajectory(id=01J..., task="compute 17*23 / len('banana')")
└── Step name="ask_llm" payload=LLMCallPayload
    ├── Step name="calculator" payload=ToolCallPayload(args={"expr":"17*23"})
    │   └── result: 391
    └── Step name="ask_llm" payload=LLMCallPayload
        └── final_answer: "65.166..."
```

Every node is a `Step` with a string ID, a `parent_step_id` (`None` at the
root), a status, timestamps, and a typed `payload` chosen from a discriminated
union over four `step_type` values:

| `step_type` | Payload | When to use |
|---|---|---|
| `llm_call` | `LLMCallPayload` | Any single chat-completion call. |
| `tool_call` | `ToolCallPayload` | A tool execution (function call, API hit, etc.). |
| `user_input` | `UserInputPayload` | An external prompt mid-run. |
| `internal` | `InternalPayload` | Bookkeeping, planning, or branching. |

The trajectory itself owns light metadata (`task`, `agent_name`,
`agent_version`, `model_id`, `final_status`, `final_answer`,
`schema_version`) and delegates the actual run history to its tree of steps.

## Design decisions

- **Tree, not DAG.** Re-entry is rare; when needed, model it with explicit
  `internal` "branch" steps. v0.1.x will not introduce DAG semantics.
- **String parent references.** `parent_step_id` is a string ID, not an
  embedded `Step`, so JSON serialization is acyclic by construction.
- **ULID IDs.** Time-sortable, lexicographic ordering matches construction
  order to the millisecond. Crockford base32 alphabet (no I, L, O, U).
- **tz-aware UTC datetimes.** Naive datetimes are rejected at construction.
- **Truncation.** The two payload fields with explicit `*_truncated` flags
  (`completion` and `result`) are capped at 64K characters. Other fields are
  stored as-is; consumer code that worries about size opens its own cap.
- **Opt-in redaction.** Raw prompts and completions are kept unless the user
  explicitly applies a `redact()` hook on the `Trajectory`.

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
