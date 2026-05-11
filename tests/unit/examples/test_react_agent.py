"""ReactAgent parser, error paths, and step-limit exhaustion.

These tests do not touch a real LLM — they patch ``ReactAgent._call_llm``
to inject canned responses.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ariadne_eval.examples.react_agent import (
    ReactAgent,
    ReactParseError,
    StepLimitExhausted,
    _parse_assistant_text,
)


@pytest.mark.fast
def test_parse_action_and_input():
    text = (
        "Thought: I need to compute 17*23.\n"
        "Action: calculator\n"
        "Action Input: 17*23\n"
    )
    action, action_input, final = _parse_assistant_text(text)
    assert action == "calculator"
    assert action_input == "17*23"
    assert final is None


@pytest.mark.fast
def test_parse_final_answer():
    text = "Thought: That's it.\nFINAL ANSWER: 42"
    action, action_input, final = _parse_assistant_text(text)
    assert action is None
    assert action_input is None
    assert final == "42"


@pytest.mark.fast
def test_parse_final_answer_multiline():
    text = "Thought: details\nFINAL ANSWER: 65.16\n(more reasoning)"
    _, _, final = _parse_assistant_text(text)
    assert final is not None
    assert final.startswith("65.16")


@pytest.mark.fast
def test_parse_malformed_raises():
    with pytest.raises(ReactParseError):
        _parse_assistant_text("just some random text with no markers")


@pytest.mark.fast
def test_parse_action_without_input_raises():
    with pytest.raises(ReactParseError):
        _parse_assistant_text("Action: calculator\n")


def _fake_response(content: str) -> SimpleNamespace:
    """Build a minimal ChatCompletion-shaped response for stubbing."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=10),
    )


@pytest.mark.fast
async def test_step_limit_exhaustion(monkeypatch):
    """When the LLM never emits FINAL ANSWER, we hit max_steps and raise."""
    agent = ReactAgent(model_id="gpt-4o-mini", max_steps=3)

    call_count = {"n": 0}

    async def _stub_call_llm(messages):
        call_count["n"] += 1
        return _fake_response(
            "Thought: keep going\nAction: calculator\nAction Input: 1+1\n"
        )

    monkeypatch.setattr(agent, "_call_llm", _stub_call_llm)

    with pytest.raises(StepLimitExhausted):
        await agent.arun("forever loop")

    assert call_count["n"] == 3


@pytest.mark.fast
async def test_arun_returns_final_answer(monkeypatch):
    agent = ReactAgent(model_id="gpt-4o-mini", max_steps=5)

    responses = iter(
        [
            "Thought: I'll search.\nAction: search\nAction Input: banana\n",
            "Thought: That has 6 letters.\nFINAL ANSWER: 6",
        ]
    )

    async def _stub_call_llm(messages):
        return _fake_response(next(responses))

    monkeypatch.setattr(agent, "_call_llm", _stub_call_llm)
    answer = await agent.arun("how many letters in banana")
    assert answer.strip() == "6"


@pytest.mark.fast
async def test_arun_persists_to_store(monkeypatch, tmp_path):
    from ariadne_eval.storage.duckdb_store import DuckDBStore

    agent = ReactAgent(model_id="gpt-4o-mini")
    responses = iter(
        [
            "Thought: simple.\nAction: calculator\nAction Input: 1+1\n",
            "Thought: done.\nFINAL ANSWER: 2",
        ]
    )

    async def _stub_call_llm(messages):
        return _fake_response(next(responses))

    monkeypatch.setattr(agent, "_call_llm", _stub_call_llm)

    store = DuckDBStore(path=tmp_path / "s.duckdb")
    try:
        await agent.arun("compute 1+1", store=store)
        listed = await store.list_trajectories()
        assert len(listed) == 1
        _, steps = await store.get_trajectory(listed[0].id)
        names = {s.name for s in steps}
        assert "tool_calculator" in names
        assert "calculator" in names
    finally:
        await store.close()
