"""Environment + task abstractions.

An Environment wraps a task backend. Crucially, `grade()` is where the reward
signal is *produced by execution* (run the tests), not asked of a model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from schemas import RewardVector


@dataclass
class Variant:
    """One concrete task instance within a family."""
    variant_id: str
    family: str
    goal: str
    spec: dict[str, Any] = field(default_factory=dict)   # backend-specific
    heldout: bool = False


@dataclass
class Observation:
    text: str
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Environment(Protocol):
    def reset(self, variant: Variant) -> Observation: ...
    def step(self, action_name: str, args: dict[str, Any]) -> Observation: ...
    def grade(self) -> RewardVector: ...
    def tool_schemas(self) -> list[dict]: ...
    def context(self) -> dict[str, Any]:
        """Structured context used for experience/policy matching."""
        ...
