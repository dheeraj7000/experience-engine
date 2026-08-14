"""Skill object — a compiled, reusable workflow (proposal 8.8).

A skill is induced from repeated successful behaviors across episodes. It
captures the WHAT (workflow steps), the WHEN (preconditions), and the
verification criteria (postconditions). Skills sit between raw experiences
(lessons/advice) and policies (behavioral rules): a skill is the executable
procedure that a policy might route to.

Lifecycle: candidate → validated → active → deprecated
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .experience import ValidationStatus


class SkillObject(BaseModel):
    skill_id: str
    name: str = ""
    description: str = ""
    scope: str = ""                               # task family or domain
    preconditions: list[str] = Field(default_factory=list)
    workflow: list[str] = Field(default_factory=list)  # ordered steps
    postconditions: list[str] = Field(default_factory=list)
    source_experiences: list[str] = Field(default_factory=list)
    source_episodes: list[str] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    evidence_count: int = 0
    version: int = 1
    validation_status: ValidationStatus = ValidationStatus.candidate
    last_validated: str | None = None

    def matches_context(self, context: dict[str, Any]) -> bool:
        """True if this skill's scope applies to the given context."""
        if not self.scope:
            return True  # global skill
        return context.get("family", "") == self.scope

    def as_instruction(self) -> str:
        """Format the skill as an injectable instruction block."""
        steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(self.workflow))
        pre = ", ".join(self.preconditions) if self.preconditions else "applicable"
        return (
            f"[Skill: {self.name}] When {pre}:\n{steps}"
        )
