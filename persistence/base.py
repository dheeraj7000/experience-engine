"""The single seam that defines A0 vs A1 vs A2.

Same agent, same tools, same environment. ONLY this object changes between
the three experimental configurations, which is what makes the comparison
clean and ablations a config flag rather than a fork.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from schemas import Episode


@runtime_checkable
class PersistenceLayer(Protocol):
    name: str

    def retrieve(self, context: dict[str, Any]) -> str:
        """Text injected into the agent prompt for this task context."""
        ...

    def record(self, episode: Episode) -> None:
        """Persist a completed (graded) episode."""
        ...

    def consolidate(self, replay_fn=None) -> None:
        """Offline learning. No-op for A0/A1; the MVES loop for A2.

        `replay_fn(experience) -> bool` lets the validator test a candidate on
        held-out variants before promotion (supplied by the Runner)."""
        ...
