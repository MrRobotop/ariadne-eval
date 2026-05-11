"""Any tree of @trace_step calls produces a Trajectory whose tree matches."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ariadne_eval.tracing.context import start_trajectory
from ariadne_eval.tracing.decorator import trace_step


@dataclass
class _Node:
    name: str
    children: list[_Node]


def _trees(max_depth: int, max_breadth: int) -> st.SearchStrategy[_Node]:
    name_strat = st.text(min_size=1, max_size=6, alphabet="abcdef")

    def _inner(depth: int) -> st.SearchStrategy[_Node]:
        if depth == 0:
            return st.builds(_Node, name=name_strat, children=st.just([]))
        return st.builds(
            _Node,
            name=name_strat,
            children=st.lists(_inner(depth - 1), max_size=max_breadth),
        )

    return _inner(max_depth)


async def _run_tree(node: _Node) -> None:
    """Run the node and its subtree, each wrapped in @trace_step."""

    @trace_step(node.name)
    async def body() -> None:
        for child in node.children:
            await _run_tree(child)

    await body()


def _expected_parents(node: _Node, parent: str | None = None) -> list[tuple[str, str | None]]:
    """Return a list of (name, parent_name) pairs for the expected tree."""
    out = [(node.name, parent)]
    for child in node.children:
        out.extend(_expected_parents(child, parent=node.name))
    return out


@pytest.mark.fast
@given(tree=_trees(max_depth=3, max_breadth=3))
@settings(max_examples=30, deadline=None)
def test_call_tree_matches_trajectory_tree(tree):
    async def run():
        async with start_trajectory("t", agent_name="a", agent_version="0.1", model_id="m") as traj:
            await _run_tree(tree)
        return traj

    traj = asyncio.run(run())

    id_to_step = {s.id: s for s in traj._steps}
    actual_parent_names: list[tuple[str, str | None]] = []
    for s in traj._steps:
        parent_name = id_to_step[s.parent_step_id].name if s.parent_step_id else None
        actual_parent_names.append((s.name, parent_name))

    expected = _expected_parents(tree)

    # Multiset comparison: tolerates traversal order differences.
    # Key maps None → "" so sorted() can compare across (name, None) tuples.
    def _key(pair: tuple[str, str | None]) -> tuple[str, str]:
        return (pair[0], pair[1] or "")

    assert sorted(actual_parent_names, key=_key) == sorted(expected, key=_key)
