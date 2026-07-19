"""Structured episode — the recorder's output.

Recorder principle (proposal 8.1): store auditable SUMMARIES, decision traces,
tool calls and rationales sufficient for debugging and learning. Never store
raw hidden chain-of-thought.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .reward import RewardVector


class Step(BaseModel):
    i: int
    action: str
    tool: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    observation: str = ""
    thought_summary: str = ""          # summary, not verbatim CoT


class Cost(BaseModel):
    tokens: int = 0
    tool_calls: int = 0
    latency_ms: float = 0.0


class Episode(BaseModel):
    episode_id: str
    task_family: str
    task_variant_id: str
    seed: int = 0
    goal: str = ""
    initial_state: dict[str, Any] = Field(default_factory=dict)
    plan: list[str] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    final_answer: Any = None
    outcome: RewardVector | None = None
    cost: Cost = Field(default_factory=Cost)
    safety_events: list[str] = Field(default_factory=list)

    def failure_signature(self) -> str:
        """A coarse signature used to cluster episodes and detect REPEATED
        failures. Real families should override with a domain-specific
        signature (e.g. failing assertion + root-cause bucket)."""
        ok = bool(self.outcome and self.outcome.task_success >= 1.0)
        first_bad = next((s.action for s in self.steps if "error" in s.observation.lower()), "none")
        return f"{self.task_family}|success={ok}|first_bad={first_bad}"
