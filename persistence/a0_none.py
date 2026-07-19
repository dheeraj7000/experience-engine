"""A0 Baseline — stateless. Records episodes for measurement only; injects
nothing, learns nothing. This is 'today's default agent'."""
from __future__ import annotations

from typing import Any

from schemas import Episode
from .store import ExperienceStore


class NoPersistence:
    name = "a0"

    def __init__(self, store: ExperienceStore) -> None:
        self.store = store

    def retrieve(self, context: dict[str, Any]) -> str:
        return ""

    def record(self, episode: Episode) -> None:
        self.store.add_episode(episode)   # kept for metrics, never retrieved

    def consolidate(self, replay_fn=None) -> None:
        return None
