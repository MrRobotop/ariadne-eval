"""Trace a real ReAct agent end-to-end.

Requires ``OPENAI_API_KEY`` in your environment. Run with:

    uv run python examples/01_quickstart/main.py

The trajectory is persisted to ``~/.ariadne/quickstart.duckdb``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ariadne_eval import DuckDBStore
from ariadne_eval.examples.react_agent import ReactAgent


async def main() -> None:
    store_path = Path("~/.ariadne/quickstart.duckdb").expanduser()
    store = DuckDBStore(path=store_path)
    try:
        agent = ReactAgent(model_id="gpt-4o-mini")
        answer = await agent.arun(
            "What is 17 * 23, and then divide by the number of letters in 'banana'?",
            store=store,
        )
        print(f"final answer: {answer}")
        print(f"trajectory persisted to: {store_path}")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
