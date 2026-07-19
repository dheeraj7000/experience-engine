"""Experience object — the inducer's output (proposal section 9).

Key property: a lesson is CONDITIONAL (context_conditions) and CONFIDENCE-
WEIGHTED, with an explicit rollback condition. This is what separates an
"experience" from a raw reflection.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ValidationStatus(str, Enum):
    candidate = "candidate"       # just induced, not yet tested
    provisional = "provisional"   # low-risk, single supporting episode
    validated = "validated"       # passed replay validation
    active = "active"             # promoted, influences behavior
    deprecated = "deprecated"     # superseded / stale
    rejected = "rejected"         # failed validation


class ExperienceObject(BaseModel):
    experience_id: str
    source_episodes: list[str] = Field(default_factory=list)
    task_family: str = ""
    context_conditions: dict[str, Any] = Field(default_factory=dict)
    root_cause: str = ""
    lesson: str = ""
    recommended_policy: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    evidence_count: int = 0
    contradictions: int = 0
    validation_status: ValidationStatus = ValidationStatus.candidate
    rollback_condition: str = ""
    last_validated: str | None = None

    def matches(self, context: dict[str, Any]) -> bool:
        """True if every context condition is satisfied by `context`."""
        for k, v in self.context_conditions.items():
            if str(context.get(k)) != str(v):
                return False
        return True
