"""Policy object — a persistent, scoped behavioral rule (proposal 8.9).

Minimal Phase-1 form. Full priority ordering / versioning / provenance is a
Phase-3 concern; the fields are present now so the store schema is stable.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .experience import ValidationStatus


class PolicyObject(BaseModel):
    policy_id: str
    scope: str = ""                               # task family or domain
    trigger_conditions: list[str] = Field(default_factory=list)
    behavior: str = ""                            # what to do when triggered
    priority: int = 50
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    supporting_experiences: list[str] = Field(default_factory=list)
    validation_status: ValidationStatus = ValidationStatus.active
    rollback_condition: str = ""

    def as_directive(self) -> str:
        return f"[{self.scope}] When {', '.join(self.trigger_conditions) or 'applicable'}: {self.behavior}"
