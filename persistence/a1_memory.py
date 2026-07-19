"""A1 Memory-only — the Reflexion/ExpeL/MemGPT class of system.

Retrieves raw past episodes + free-text reflections and injects them. It does
NOT diagnose, validate, compile, or update policy. Expected to help modestly
then plateau — the baseline the Experience Engine must beat.
"""
from __future__ import annotations

from typing import Any

from schemas import Episode
from .store import ExperienceStore


class MemoryOnly:
    name = "a1"

    def __init__(self, store: ExperienceStore, k: int = 3) -> None:
        self.store = store
        self.k = k

    def retrieve(self, context: dict[str, Any]) -> str:
        query = context.get("goal", "") + " " + context.get("family", "")
        hits = self.store.search_episodes(query, k=self.k)
        if not hits:
            return ""
        lines = []
        for e in hits:
            verdict = "succeeded" if (e.outcome and e.outcome.task_success >= 1.0) else "failed"
            reflection = self._reflect(e)
            lines.append(f"- Past attempt ({verdict}): {reflection}")
        return "\n".join(lines)

    def record(self, episode: Episode) -> None:
        self.store.add_episode(episode)

    def consolidate(self, replay_fn=None) -> None:
        return None  # memory-only does no offline learning

    @staticmethod
    def _reflect(e: Episode) -> str:
        """Naive free-text reflection — deliberately shallow (no causal
        diagnosis, no validation). That shallowness is the point of A1."""
        if e.outcome and e.outcome.task_success >= 1.0:
            return f"solving '{e.goal}' worked via {len(e.steps)} step(s)."
        last = e.steps[-1].observation if e.steps else "no progress"
        return f"attempt at '{e.goal}' failed; last observation: {last[:160]}"
