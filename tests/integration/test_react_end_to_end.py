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
        assert "LLMCallPayload" in step_payload_types
        assert "ToolCallPayload" in step_payload_types
        assert "InternalPayload" in step_payload_types
    finally:
        await store.close()
