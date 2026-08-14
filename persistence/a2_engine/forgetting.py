"""Forgetting Manager (Phase 6, proposal 8.11).

Agents should forget — otherwise experience bloats into millions of stale rules,
slows retrieval, causes contradictions, and overfits to old environments.

Forgetting ≠ deletion. It means:
  - Demote (reduce priority/confidence)
  - Archive (remove from active retrieval but keep for audit)
  - Merge (combine near-duplicate experiences)
  - Require revalidation (flag as stale)
  - Supersede (link to newer replacement)

Mechanisms:
  1. Time/evidence decay — confidence decays without reinforcement
  2. Staleness detection — experiences not retrieved or validated recently
  3. Contradiction-driven — highly contradicted experiences get demoted
  4. Revalidation scheduling — periodic re-test on held-out
  5. Evidence-weighted retention — more evidence = slower decay
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from schemas import ExperienceObject, PolicyObject, SkillObject
from schemas.experience import ValidationStatus


@dataclass
class DecayConfig:
    """Configuration for the forgetting manager."""
    # Confidence decays by this fraction each consolidation cycle if not reinforced.
    base_decay_rate: float = 0.02
    # Minimum confidence before archiving (below this = deprecated).
    archive_threshold: float = 0.15
    # Experiences with contradictions >= this get extra decay.
    contradiction_decay_multiplier: float = 2.0
    # Evidence count that halves the decay rate (more evidence = more durable).
    evidence_halflife: int = 5
    # Maximum cycles without reinforcement before staleness flag.
    staleness_cycles: int = 10
    # Revalidation interval (cycles between forced revalidation).
    revalidation_interval: int = 20


@dataclass
class ForgettingAction:
    """A record of what the forgetting manager did."""
    target_id: str
    target_type: str  # "experience", "policy", "skill"
    action: str       # "decayed", "archived", "flagged_stale", "revalidation_due"
    old_confidence: float
    new_confidence: float
    reason: str


class ForgettingManager:
    """Manages experience lifecycle: decay, staleness, archival.

    Called during consolidation (offline). Never during retrieve (online).
    """

    def __init__(self, config: DecayConfig | None = None) -> None:
        self.config = config or DecayConfig()
        # Track cycles since last reinforcement for each item.
        self._cycles_since_reinforced: dict[str, int] = {}
        self._reinforced_this_cycle: set[str] = set()
        self._total_cycles: int = 0

    def tick(self) -> None:
        """Advance the cycle counter. Called once per consolidation pass."""
        self._total_cycles += 1
        for key in list(self._cycles_since_reinforced):
            if key not in self._reinforced_this_cycle:
                self._cycles_since_reinforced[key] += 1
        self._reinforced_this_cycle.clear()

    def mark_reinforced(self, item_id: str) -> None:
        """Mark an item as recently reinforced (resets decay clock)."""
        self._cycles_since_reinforced[item_id] = 0
        self._reinforced_this_cycle.add(item_id)

    def apply_decay(self, experiences: list[ExperienceObject],
                    policies: list[PolicyObject],
                    skills: list[SkillObject] | None = None
                    ) -> list[ForgettingAction]:
        """Apply one cycle of decay to all active items. Returns actions taken."""
        actions: list[ForgettingAction] = []
        self.tick()

        for exp in experiences:
            if exp.validation_status in (ValidationStatus.deprecated,
                                         ValidationStatus.rejected):
                continue
            action = self._decay_experience(exp)
            if action:
                actions.append(action)

        for pol in policies:
            if pol.validation_status == ValidationStatus.deprecated:
                continue
            action = self._decay_policy(pol)
            if action:
                actions.append(action)

        if skills:
            for skill in skills:
                if skill.validation_status == ValidationStatus.deprecated:
                    continue
                action = self._decay_skill(skill)
                if action:
                    actions.append(action)

        return actions

    def stale_items(self, experiences: list[ExperienceObject]) -> list[str]:
        """Return IDs of experiences that are stale (need revalidation)."""
        stale: list[str] = []
        for exp in experiences:
            if exp.validation_status in (ValidationStatus.deprecated,
                                         ValidationStatus.rejected):
                continue
            cycles = self._cycles_since_reinforced.get(exp.experience_id, 0)
            if cycles >= self.config.staleness_cycles:
                stale.append(exp.experience_id)
        return stale

    def items_due_revalidation(self, experiences: list[ExperienceObject]
                               ) -> list[str]:
        """Return IDs of experiences due for periodic revalidation."""
        due: list[str] = []
        for exp in experiences:
            if exp.validation_status not in (ValidationStatus.active,
                                             ValidationStatus.validated):
                continue
            cycles = self._cycles_since_reinforced.get(exp.experience_id, 0)
            if cycles > 0 and cycles % self.config.revalidation_interval == 0:
                due.append(exp.experience_id)
        return due

    def _decay_experience(self, exp: ExperienceObject) -> ForgettingAction | None:
        """Apply decay to a single experience."""
        item_id = exp.experience_id
        if item_id not in self._cycles_since_reinforced:
            self._cycles_since_reinforced[item_id] = 0
            return None  # first cycle, no decay yet

        cycles = self._cycles_since_reinforced[item_id]
        if cycles == 0:
            return None  # just reinforced

        old_conf = exp.confidence
        decay = self._compute_decay(exp.evidence_count, exp.contradictions)
        exp.confidence = max(0.0, round(exp.confidence - decay, 4))

        # Archive if below threshold.
        if exp.confidence < self.config.archive_threshold:
            exp.validation_status = ValidationStatus.deprecated
            return ForgettingAction(
                target_id=item_id, target_type="experience",
                action="archived", old_confidence=old_conf,
                new_confidence=exp.confidence,
                reason=f"confidence decayed below {self.config.archive_threshold} "
                       f"after {cycles} cycles without reinforcement",
            )

        if decay > 0:
            return ForgettingAction(
                target_id=item_id, target_type="experience",
                action="decayed", old_confidence=old_conf,
                new_confidence=exp.confidence,
                reason=f"decay={decay:.4f} (cycles={cycles}, "
                       f"evidence={exp.evidence_count}, "
                       f"contradictions={exp.contradictions})",
            )
        return None

    def _decay_policy(self, pol: PolicyObject) -> ForgettingAction | None:
        """Apply decay to a policy (slower than experiences — policies are promoted)."""
        item_id = pol.policy_id
        if item_id not in self._cycles_since_reinforced:
            self._cycles_since_reinforced[item_id] = 0
            return None

        cycles = self._cycles_since_reinforced[item_id]
        if cycles == 0:
            return None

        old_conf = pol.confidence
        # Policies decay at half the rate of experiences (they're already validated).
        decay = self._compute_decay(5, 0) * 0.5  # conservative
        pol.confidence = max(0.0, round(pol.confidence - decay, 4))

        if pol.confidence < self.config.archive_threshold:
            pol.validation_status = ValidationStatus.deprecated
            return ForgettingAction(
                target_id=item_id, target_type="policy",
                action="archived", old_confidence=old_conf,
                new_confidence=pol.confidence,
                reason=f"policy confidence decayed below threshold after {cycles} cycles",
            )

        if decay > 0:
            return ForgettingAction(
                target_id=item_id, target_type="policy",
                action="decayed", old_confidence=old_conf,
                new_confidence=pol.confidence,
                reason=f"policy decay (cycles={cycles})",
            )
        return None

    def _decay_skill(self, skill: SkillObject) -> ForgettingAction | None:
        """Apply decay to a skill."""
        item_id = skill.skill_id
        if item_id not in self._cycles_since_reinforced:
            self._cycles_since_reinforced[item_id] = 0
            return None

        cycles = self._cycles_since_reinforced[item_id]
        if cycles == 0:
            return None

        old_conf = skill.confidence
        decay = self._compute_decay(skill.evidence_count, 0)
        skill.confidence = max(0.0, round(skill.confidence - decay, 4))

        if skill.confidence < self.config.archive_threshold:
            skill.validation_status = ValidationStatus.deprecated
            return ForgettingAction(
                target_id=item_id, target_type="skill",
                action="archived", old_confidence=old_conf,
                new_confidence=skill.confidence,
                reason=f"skill confidence decayed below threshold",
            )

        if decay > 0:
            return ForgettingAction(
                target_id=item_id, target_type="skill",
                action="decayed", old_confidence=old_conf,
                new_confidence=skill.confidence,
                reason=f"skill decay (cycles={cycles}, evidence={skill.evidence_count})",
            )
        return None

    def _compute_decay(self, evidence_count: int, contradictions: int) -> float:
        """Compute decay amount. More evidence = slower decay. More contradictions = faster."""
        base = self.config.base_decay_rate
        # Evidence halves the decay rate (more evidence = more durable).
        evidence_factor = 1.0 / (1.0 + evidence_count / self.config.evidence_halflife)
        # Contradictions accelerate decay.
        contradiction_factor = 1.0 + (
            contradictions * (self.config.contradiction_decay_multiplier - 1.0)
        )
        return round(base * evidence_factor * contradiction_factor, 6)
