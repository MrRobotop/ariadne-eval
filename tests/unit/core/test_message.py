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
        Message(role="banana", content="x")


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
