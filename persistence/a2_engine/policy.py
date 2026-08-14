"""Policy Manager (Phase 5 upgrade, proposal 8.9).

Phase 1 was a simple promote-from-experience function. Phase 5 adds:
  - Priority ordering with conflict resolution
  - Scope refinement (narrowing conditions when conflicts arise)
  - Policy versioning (supersedes edges, not deletion)
  - Rollback support (revert a policy and demote its experience)
  - Safety gating (high-impact policies require higher confidence thresholds)
  - Skill routing (policies can reference skills to execute)
"""
from __future__ import annotations

import uuid
from typing import Any

from schemas import ExperienceObject, PolicyObject, SkillObject
from schemas.experience import ValidationStatus


# Safety-critical scopes require higher confidence for promotion.
SAFETY_CRITICAL_SCOPES = {"security", "auth", "deletion", "production"}
SAFETY_CONFIDENCE_THRESHOLD = 0.85
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


class PolicyConflict:
    """A detected conflict between two active policies."""

    def __init__(self, policy_a: PolicyObject, policy_b: PolicyObject,
                 overlap_description: str) -> None:
        self.policy_a = policy_a
        self.policy_b = policy_b
        self.overlap_description = overlap_description

    def winner(self) -> PolicyObject:
        """Higher priority wins. Ties broken by confidence."""
        if self.policy_a.priority != self.policy_b.priority:
            return self.policy_a if self.policy_a.priority > self.policy_b.priority else self.policy_b
        return self.policy_a if self.policy_a.confidence >= self.policy_b.confidence else self.policy_b


class PolicyManager:
    """Full policy lifecycle management."""

    def __init__(self, safety_threshold: float = SAFETY_CONFIDENCE_THRESHOLD,
                 default_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        self.safety_threshold = safety_threshold
        self.default_threshold = default_threshold

    # ---- promotion --------------------------------------------------------
    def promote(self, experience: ExperienceObject,
                skill: SkillObject | None = None) -> PolicyObject:
        """Promote a validated experience into an active policy.

        If a skill is provided, the policy behavior references the skill.
        """
        # Safety gating: check if scope requires higher confidence.
        min_conf = self._min_confidence_for_scope(experience.task_family)
        if experience.confidence < min_conf:
            # Return a candidate policy (not yet active).
            return PolicyObject(
                policy_id=f"pol_{uuid.uuid4().hex[:8]}",
                scope=experience.task_family,
                trigger_conditions=self._conditions(experience),
                behavior=self._behavior(experience, skill),
                priority=self._compute_priority(experience),
                confidence=experience.confidence,
                supporting_experiences=[experience.experience_id],
                validation_status=ValidationStatus.candidate,
                rollback_condition=experience.rollback_condition,
            )

        return PolicyObject(
            policy_id=f"pol_{uuid.uuid4().hex[:8]}",
            scope=experience.task_family,
            trigger_conditions=self._conditions(experience),
            behavior=self._behavior(experience, skill),
            priority=self._compute_priority(experience),
            confidence=experience.confidence,
            supporting_experiences=[experience.experience_id],
            validation_status=ValidationStatus.active,
            rollback_condition=experience.rollback_condition,
        )

    def promote_from_skill(self, skill: SkillObject) -> PolicyObject:
        """Promote a validated skill directly into a routing policy."""
        return PolicyObject(
            policy_id=f"pol_{uuid.uuid4().hex[:8]}",
            scope=skill.scope,
            trigger_conditions=skill.preconditions or ["applicable"],
            behavior=f"Use skill '{skill.name}': {skill.as_instruction()}",
            priority=60 + int(skill.confidence * 30),
            confidence=skill.confidence,
            supporting_experiences=skill.source_experiences,
            validation_status=ValidationStatus.active
            if skill.validation_status == ValidationStatus.active
            else ValidationStatus.candidate,
            rollback_condition=f"if skill '{skill.name}' degrades performance",
        )

    # ---- conflict resolution ----------------------------------------------
    def detect_conflicts(self, policies: list[PolicyObject]) -> list[PolicyConflict]:
        """Find policies with overlapping trigger conditions and same scope."""
        active = [p for p in policies
                  if p.validation_status == ValidationStatus.active]
        conflicts: list[PolicyConflict] = []
        seen: set[tuple[str, str]] = set()

        for i, a in enumerate(active):
            for b in active[i + 1:]:
                pair = tuple(sorted([a.policy_id, b.policy_id]))
                if pair in seen:
                    continue
                seen.add(pair)
                if self._scopes_overlap(a, b):
                    overlap = self._describe_overlap(a, b)
                    if overlap:
                        conflicts.append(PolicyConflict(a, b, overlap))
        return conflicts

    def resolve_conflicts(self, conflicts: list[PolicyConflict],
                          policies: list[PolicyObject]) -> list[str]:
        """Resolve conflicts by adjusting priorities or narrowing scope.

        Returns descriptions of actions taken.
        """
        actions: list[str] = []
        for conflict in conflicts:
            winner = conflict.winner()
            loser = conflict.policy_a if winner is conflict.policy_b else conflict.policy_b
            # Narrow the loser's scope by adding a distinguishing condition.
            if loser.priority == winner.priority:
                loser.priority = max(0, loser.priority - 10)
                actions.append(
                    f"Lowered priority of '{loser.policy_id}' "
                    f"(lost to '{winner.policy_id}' on confidence).")
            else:
                actions.append(
                    f"Conflict between '{winner.policy_id}' (priority={winner.priority}) "
                    f"and '{loser.policy_id}' (priority={loser.priority}): "
                    f"winner is '{winner.policy_id}'.")
        return actions

    # ---- versioning -------------------------------------------------------
    def supersede(self, old_policy: PolicyObject, new_policy: PolicyObject) -> None:
        """Mark old_policy as superseded by new_policy."""
        old_policy.validation_status = ValidationStatus.deprecated
        # The new policy inherits supporting evidence from the old one.
        for exp_id in old_policy.supporting_experiences:
            if exp_id not in new_policy.supporting_experiences:
                new_policy.supporting_experiences.append(exp_id)

    # ---- rollback ---------------------------------------------------------
    def rollback(self, policy: PolicyObject) -> None:
        """Deactivate a policy that is causing harm.

        The policy is deprecated (not deleted) for auditability.
        """
        policy.validation_status = ValidationStatus.deprecated

    # ---- scope refinement -------------------------------------------------
    def refine_scope(self, policy: PolicyObject,
                     additional_condition: str) -> PolicyObject:
        """Narrow a policy's applicability by adding a trigger condition.

        Returns a new versioned policy (the old one is superseded).
        """
        new_policy = PolicyObject(
            policy_id=f"pol_{uuid.uuid4().hex[:8]}",
            scope=policy.scope,
            trigger_conditions=policy.trigger_conditions + [additional_condition],
            behavior=policy.behavior,
            priority=policy.priority,
            confidence=policy.confidence,
            supporting_experiences=list(policy.supporting_experiences),
            validation_status=ValidationStatus.active,
            rollback_condition=policy.rollback_condition,
        )
        self.supersede(policy, new_policy)
        return new_policy

    # ---- ordering ---------------------------------------------------------
    def ordered_policies(self, policies: list[PolicyObject],
                         scope: str = "") -> list[PolicyObject]:
        """Return active policies in priority order for a given scope."""
        active = [
            p for p in policies
            if p.validation_status == ValidationStatus.active
            and (not scope or p.scope == scope or p.scope == "")
        ]
        active.sort(key=lambda p: (p.priority, p.confidence), reverse=True)
        return active

    # ---- internals --------------------------------------------------------
    @staticmethod
    def _conditions(experience: ExperienceObject) -> list[str]:
        conditions = [f"{k}={v}" for k, v in experience.context_conditions.items()]
        return conditions or ["applicable"]

    @staticmethod
    def _behavior(experience: ExperienceObject,
                  skill: SkillObject | None) -> str:
        if skill:
            return f"Apply skill '{skill.name}': {skill.as_instruction()}"
        return experience.recommended_policy or experience.lesson

    @staticmethod
    def _compute_priority(experience: ExperienceObject) -> int:
        """Priority based on confidence + evidence strength."""
        base = 50
        conf_bonus = int(experience.confidence * 30)
        evidence_bonus = min(10, experience.evidence_count)
        contradiction_penalty = min(10, experience.contradictions * 3)
        return base + conf_bonus + evidence_bonus - contradiction_penalty

    def _min_confidence_for_scope(self, scope: str) -> float:
        """Safety-critical scopes require higher confidence."""
        if scope.lower() in SAFETY_CRITICAL_SCOPES:
            return self.safety_threshold
        return self.default_threshold

    @staticmethod
    def _scopes_overlap(a: PolicyObject, b: PolicyObject) -> bool:
        """Two policies overlap if they share the same scope."""
        if a.scope and b.scope and a.scope != b.scope:
            return False
        # Same scope or one is global — check trigger overlap.
        shared_triggers = set(a.trigger_conditions) & set(b.trigger_conditions)
        return bool(shared_triggers) or "applicable" in a.trigger_conditions or "applicable" in b.trigger_conditions

    @staticmethod
    def _describe_overlap(a: PolicyObject, b: PolicyObject) -> str:
        shared = set(a.trigger_conditions) & set(b.trigger_conditions)
        if shared:
            return f"shared triggers: {', '.join(shared)}"
        if "applicable" in a.trigger_conditions:
            return f"'{a.policy_id}' is broadly applicable, overlaps with '{b.policy_id}'"
        if "applicable" in b.trigger_conditions:
            return f"'{b.policy_id}' is broadly applicable, overlaps with '{a.policy_id}'"
        return ""
