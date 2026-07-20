"""ReAct-style agent controller.

Identical across A0/A1/A2 — the ONLY thing that changes between configs is the
persistence layer, which supplies `injected_context` and receives the recorded
episode. The controller is deliberately dumb about learning.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from providers import ModelProvider
from schemas import Episode, Step
from schemas.episode import Cost

if TYPE_CHECKING:  # avoid a runtime agent<->harness import cycle; types only
    from harness.environment import Environment, Variant

SYSTEM_PROMPT = (
    "You are a QA engineering agent. You solve testing and debugging tasks by "
    "calling tools. Think briefly, then act. When you believe the task is done, "
    "call the appropriate tool to submit your solution."
)


class AgentController:
    def __init__(self, provider: ModelProvider, max_steps: int = 4) -> None:
        self.provider = provider
        self.max_steps = max_steps

    def run(
        self,
        env: Environment,
        variant: Variant,
        injected_context: str,
        seed: int = 0,
    ) -> Episode:
        obs = env.reset(variant)
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        user = ""
        if injected_context:
            user += f"Relevant prior experience:\n{injected_context}\n\n"
        user += f"Task: {variant.goal}\n\n{obs.text}"
        messages.append({"role": "user", "content": user})

        steps: list[Step] = []
        total_cost = Cost()
        final_answer: Any = None
        tools = env.tool_schemas()

        for i in range(self.max_steps):
            resp = self.provider.complete(messages, tools=tools)
            total_cost.tokens += resp.usage.total_tokens
            total_cost.latency_ms += resp.usage.latency_ms

            if not resp.tool_calls:
                final_answer = resp.text
                break

            tc = resp.tool_calls[0]
            total_cost.tool_calls += 1
            step_obs = env.step(tc.name, tc.arguments)
            steps.append(Step(
                i=i, action=tc.name, tool=tc.name, tool_args=tc.arguments,
                observation=step_obs.text[:1000],
                thought_summary=(resp.text or "")[:300],
            ))
            messages.append({"role": "assistant", "content": resp.text or f"[call {tc.name}]"})
            messages.append({"role": "user", "content": f"Observation: {step_obs.text[:1000]}"})
            if step_obs.done:
                final_answer = tc.arguments
                break

        return Episode(
            episode_id=f"ep_{uuid.uuid4().hex[:8]}",
            task_family=variant.family,
            task_variant_id=variant.variant_id,
            seed=seed,
            goal=variant.goal,
            initial_state=dict(variant.spec),
            steps=steps,
            final_answer=final_answer,
            cost=total_cost,
        )
