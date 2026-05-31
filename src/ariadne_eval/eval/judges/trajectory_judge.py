"""LLM-as-judge over a full trajectory, defaulting to litellm.acompletion."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast

from ariadne_eval.core.trajectory import LLMCallPayload, Step, Trajectory
from ariadne_eval.eval.case import Case
from ariadne_eval.eval.judges.base import JudgeVerdict
from ariadne_eval.eval.judges.prompts import (
    PLAN_QUALITY_SYSTEM,
    PLAN_QUALITY_USER_TEMPLATE,
    parse_plan_quality_verdict,
)

__all__ = ["TrajectoryJudge"]


_NO_PLAN_SENTINEL = "(no plan recorded)"


def _extract_plan(steps: list[Step]) -> str | None:
    """Return the completion text of the first LLMCallPayload step in started_at order.

    Single source of truth for "what's the plan" — used by both
    ``TrajectoryJudge`` and ``PlanQuality`` so the rule stays consistent.
    """
    llm_steps = sorted(
        (s for s in steps if isinstance(s.payload, LLMCallPayload)),
        key=lambda s: s.started_at,
    )
    if not llm_steps:
        return None
    payload = cast(LLMCallPayload, llm_steps[0].payload)
    return payload.completion


_Client = Callable[..., Awaitable[str]]


async def _litellm_default_client(
    *, model: str, messages: list[dict[str, str]], temperature: float
) -> str:
    """Default judge client: calls litellm.acompletion and returns the text."""
    import litellm

    response = await litellm.acompletion(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return cast(str, response.choices[0].message.content)


class TrajectoryJudge:
    """LLM-as-judge over a full trajectory, using a configurable prompt."""

    name: str

    def __init__(
        self,
        model: str,
        *,
        system_prompt: str = PLAN_QUALITY_SYSTEM,
        user_prompt_template: str = PLAN_QUALITY_USER_TEMPLATE,
        response_parser: Callable[[str], JudgeVerdict] = parse_plan_quality_verdict,
        client: _Client | None = None,
        temperature: float = 0.0,
        name: str = "trajectory_judge",
    ) -> None:
        """Initialise with a model name and (optionally) an injected async client."""
        self._model = model
        self._system_prompt = system_prompt
        self._user_prompt_template = user_prompt_template
        self._response_parser = response_parser
        self._client: _Client = client if client is not None else _litellm_default_client
        self._temperature = temperature
        self.name = name

    async def judge(
        self,
        trajectory: Trajectory,
        steps: list[Step],
        case: Case | None,
    ) -> JudgeVerdict:
        """Render prompts, call the client, parse the verdict."""
        plan_text = _extract_plan(steps) or _NO_PLAN_SENTINEL
        user_msg = self._user_prompt_template.format(task=trajectory.task, plan=plan_text)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_msg},
        ]
        completion = await self._client(
            model=self._model, messages=messages, temperature=self._temperature
        )
        return self._response_parser(completion)
