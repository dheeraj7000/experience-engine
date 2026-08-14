"""Contradiction Miner (Phase 2, proposal 8.6).

Detects conflicting experiences: two validated lessons that recommend opposite
actions for overlapping contexts. Contradictions do NOT require deleting either
experience — they require REFINING applicable conditions so each lesson is
scoped to where it actually helps.

Example:
    Experience A: "For SQL tasks, skip schema inspection to save time."
    Experience B: "For SQL tasks, always inspect schema before querying."
    → Contradiction detected. Resolution: refine conditions to distinguish
      when each applies (e.g., familiar vs unfamiliar databases).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from schemas import ExperienceObject
from schemas.experience import ValidationStatus


@dataclass
class Contradiction:
    """A detected conflict between two experiences."""
    experience_a_id: str
    experience_b_id: str
    overlap_context: dict[str, Any]
    severity: float  # 0-1: how much they actually conflict in practice
    description: str
    resolution_hint: str = ""
    resolved: bool = False


class ContradictionMiner:
    """Scans active/validated experiences for conflicting recommendations.

    Two experiences CONTRADICT when:
    1. Their context_conditions overlap (both would fire for the same task).
    2. Their lessons recommend incompatible actions.

    Incompatibility is detected via:
    - Negation keywords ("do not X" vs "always X")
    - Opposing action verbs for the same target
    - Same recommended_policy slot with semantically opposite content

    Phase 2 uses keyword heuristics. Phase 3 can upgrade to embedding-based
    semantic opposition detection.
    """

    # Pairs of opposing terms that indicate conflicting advice.
    _OPPOSITION_PAIRS = [
        ({"always", "must", "ensure"}, {"never", "skip", "avoid", "do not"}),
        ({"before", "first", "inspect"}, {"skip", "omit", "bypass"}),
        ({"include", "add"}, {"exclude", "remove", "drop"}),
        ({"wait", "pause", "delay"}, {"immediately", "directly", "straight"}),
    ]

    def __init__(self, min_severity: float = 0.3) -> None:
        self.min_severity = min_severity

    def detect(self, experiences: list[ExperienceObject]) -> list[Contradiction]:
        """Find all contradictions among active/validated experiences."""
        active = [e for e in experiences
                  if e.validation_status in (ValidationStatus.active,
                                             ValidationStatus.validated)]
        contradictions: list[Contradiction] = []
        seen_pairs: set[tuple[str, str]] = set()

        for i, a in enumerate(active):
            for b in active[i + 1:]:
                pair_key = tuple(sorted([a.experience_id, b.experience_id]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                overlap = self._context_overlap(a, b)
                if overlap is None:
                    continue  # no overlapping conditions, can't conflict

                severity = self._compute_severity(a, b)
                if severity < self.min_severity:
                    continue

                contradictions.append(Contradiction(
                    experience_a_id=a.experience_id,
                    experience_b_id=b.experience_id,
                    overlap_context=overlap,
                    severity=severity,
                    description=self._describe(a, b),
                    resolution_hint=self._suggest_resolution(a, b),
                ))
        return contradictions

    def apply_contradictions(self, experiences: list[ExperienceObject],
                             contradictions: list[Contradiction]) -> None:
        """Update confidence and contradiction counts on affected experiences.
        Does NOT delete anything — contradictions refine, not destroy."""
        exp_map = {e.experience_id: e for e in experiences}
        for c in contradictions:
            for eid in (c.experience_a_id, c.experience_b_id):
                exp = exp_map.get(eid)
                if exp:
                    exp.contradictions += 1
                    # Reduce confidence proportional to severity.
                    penalty = 0.05 * c.severity
                    exp.confidence = max(0.0, round(exp.confidence - penalty, 4))

    @staticmethod
    def _context_overlap(a: ExperienceObject, b: ExperienceObject
                         ) -> dict[str, Any] | None:
        """Return shared context conditions, or None if no overlap."""
        if not a.context_conditions and not b.context_conditions:
            # Both are unconditional → they overlap on everything.
            return {"scope": "global"}
        if not a.context_conditions or not b.context_conditions:
            # One is unconditional → overlaps with the other's conditions.
            return a.context_conditions or b.context_conditions

        # Same task family is the minimum overlap for a meaningful conflict.
        if a.task_family and b.task_family and a.task_family != b.task_family:
            return None

        # Check if conditions are compatible (shared keys with same values).
        shared: dict[str, Any] = {}
        for k in set(a.context_conditions) & set(b.context_conditions):
            if str(a.context_conditions[k]) == str(b.context_conditions[k]):
                shared[k] = a.context_conditions[k]
        # If they share at least one matching condition (or same family), overlap.
        if shared or a.task_family == b.task_family:
            return shared or {"task_family": a.task_family}
        return None

    def _compute_severity(self, a: ExperienceObject, b: ExperienceObject) -> float:
        """How strongly do the lessons conflict? 0 = no conflict, 1 = direct opposite."""
        lesson_a = (a.lesson + " " + a.recommended_policy).lower()
        lesson_b = (b.lesson + " " + b.recommended_policy).lower()

        # Check for opposition keyword pairs.
        opposition_score = 0.0
        for positive_set, negative_set in self._OPPOSITION_PAIRS:
            a_has_pos = any(w in lesson_a for w in positive_set)
            a_has_neg = any(w in lesson_a for w in negative_set)
            b_has_pos = any(w in lesson_b for w in positive_set)
            b_has_neg = any(w in lesson_b for w in negative_set)
            # A recommends positive, B recommends negative (or vice versa).
            if (a_has_pos and b_has_neg) or (a_has_neg and b_has_pos):
                opposition_score += 0.3

        return min(1.0, opposition_score)

    @staticmethod
    def _describe(a: ExperienceObject, b: ExperienceObject) -> str:
        return (
            f"Experience '{a.experience_id}' says: \"{a.lesson[:80]}\" "
            f"but '{b.experience_id}' says: \"{b.lesson[:80]}\" "
            f"in overlapping context (family={a.task_family or b.task_family})."
        )

    @staticmethod
    def _suggest_resolution(a: ExperienceObject, b: ExperienceObject) -> str:
        """Suggest how to resolve: typically by refining conditions."""
        if a.evidence_count > b.evidence_count * 2:
            return (f"'{a.experience_id}' has much more evidence ({a.evidence_count} vs "
                    f"{b.evidence_count}). Consider narrowing '{b.experience_id}' conditions "
                    f"or deprecating it.")
        if b.evidence_count > a.evidence_count * 2:
            return (f"'{b.experience_id}' has much more evidence. Consider narrowing "
                    f"'{a.experience_id}' conditions or deprecating it.")
        return ("Similar evidence strength. Refine context_conditions on both to "
                "separate their applicable scopes (e.g., add a distinguishing "
                "condition that determines when each lesson applies).")
