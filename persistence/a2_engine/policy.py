"""Policy Manager (minimal, proposal 8.9).

Promotes a validated experience into an active, scoped behavioral policy.
Full priority ordering / conflict resolution / versioning is Phase 3.
"""
from __future__ import annotations

import uuid

from schemas import ExperienceObject, PolicyObject
from schemas.experience import ValidationStatus


class PolicyManager:
    def promote(self, experience: ExperienceObject) -> PolicyObject:
        return PolicyObject(
            policy_id=f"pol_{uuid.uuid4().hex[:8]}",
            scope=experience.task_family,
            trigger_conditions=[f"{k}={v}" for k, v in experience.context_conditions.items()]
                               or ["applicable"],
            behavior=experience.recommended_policy or experience.lesson,
            priority=50 + int(experience.confidence * 40),
            confidence=experience.confidence,
            supporting_experiences=[experience.experience_id],
            validation_status=ValidationStatus.active,
            rollback_condition=experience.rollback_condition,
        )
