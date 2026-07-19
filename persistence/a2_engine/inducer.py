"""Experience Inducer (proposal 8.4).

Turns a diagnosed cluster into a CONDITIONAL experience object: context +
lesson + recommended policy + rollback condition. The lesson must be scoped,
not global advice ('always inspect everything' is a bad rule).
"""
from __future__ import annotations

import uuid
from typing import Any

from schemas import Episode, ExperienceObject
from .confidence import compute_confidence


class ExperienceInducer:
    def induce(self, episodes: list[Episode], diagnosis: dict[str, Any],
               context: dict[str, Any]) -> ExperienceObject:
        family = episodes[0].task_family if episodes else ""
        successes = [e for e in episodes if e.outcome and e.outcome.task_success >= 1.0]
        consistency = self._consistency(episodes)
        confidence = compute_confidence(
            evidence_count=len(episodes),
            consistency=consistency,
            contradictions=0,
        )
        lesson = self._lesson(diagnosis, successes)
        return ExperienceObject(
            experience_id=f"exp_{uuid.uuid4().hex[:8]}",
            source_episodes=[e.episode_id for e in episodes],
            task_family=family,
            context_conditions=self._conditions(context),
            root_cause=diagnosis.get("root_cause", ""),
            lesson=lesson,
            recommended_policy=diagnosis.get("counterfactual_repair", lesson),
            confidence=confidence,
            evidence_count=len(episodes),
            contradictions=0,
            rollback_condition="if injecting this lesson does not improve replay outcome",
        )

    @staticmethod
    def _consistency(episodes: list[Episode]) -> float:
        if not episodes:
            return 0.0
        succ = sum(1 for e in episodes if e.outcome and e.outcome.task_success >= 1.0)
        frac = succ / len(episodes)
        return max(frac, 1.0 - frac)  # agreement in either direction

    @staticmethod
    def _conditions(context: dict[str, Any]) -> dict[str, Any]:
        # Keep only stable, matchable keys (avoid over-specific conditions).
        keep = ("family", "failure_type", "signal", "file_type", "data_volume")
        return {k: context[k] for k in keep if k in context}

    @staticmethod
    def _lesson(diagnosis: dict, successes: list[Episode]) -> str:
        repair = diagnosis.get("counterfactual_repair") or diagnosis.get("root_cause", "")
        if successes:
            return f"When this context recurs, {repair} (observed to succeed in prior episodes)."
        return f"When this context recurs, {repair}."
