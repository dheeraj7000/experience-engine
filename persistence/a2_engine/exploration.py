"""Exploration Manager (Phase 7, proposal 8.13 / APEX-inspired).

Long-term agents should generate safe learning opportunities rather than only
passively learning from assigned tasks. Without exploration, the agent suffers
from "exploration collapse" — behavior concentrates around familiar high-reward
routines and performance plateaus on novel task variants.

The Exploration Manager:
  1. Identifies underexplored task regions (weak families, low-coverage variants)
  2. Measures strategy diversity (are we always using the same approach?)
  3. Suggests practice tasks targeting weak areas
  4. Compares alternative policies on held-out variants
  5. Tracks exploration/exploitation balance

All exploration happens in SANDBOX — never on production tasks. Generated
practice tasks use the same TaskFamily interface and execution-grounded grading.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from schemas import Episode, ExperienceObject, SkillObject
from schemas.experience import ValidationStatus


@dataclass
class FamilyProfile:
    """Performance profile for a task family."""
    family_id: str
    total_episodes: int = 0
    successes: int = 0
    failures: int = 0
    success_rate: float = 0.0
    unique_variants_seen: int = 0
    unique_strategies: int = 0       # distinct action sequences observed
    active_experiences: int = 0
    active_skills: int = 0
    exploration_score: float = 0.0   # 0 = well-explored, 1 = needs exploration


@dataclass
class ExplorationSuggestion:
    """A suggestion for what to explore next."""
    family_id: str
    reason: str
    priority: float  # 0-1, higher = more urgent
    suggested_action: str  # "practice", "try_alternative", "expand_variants"
    metadata: dict[str, Any] = field(default_factory=dict)


class ExplorationManager:
    """Identifies weak areas and suggests exploration targets.

    Operates on the offline path — analyzes accumulated episodes to find
    gaps in coverage and suggest where the agent should practice.
    """

    def __init__(self, diversity_threshold: float = 0.3,
                 min_episodes_for_analysis: int = 5) -> None:
        self.diversity_threshold = diversity_threshold
        self.min_episodes = min_episodes_for_analysis

    def profile_families(self, episodes: list[Episode],
                         experiences: list[ExperienceObject],
                         skills: list[SkillObject] | None = None
                         ) -> dict[str, FamilyProfile]:
        """Build performance profiles for each task family."""
        profiles: dict[str, FamilyProfile] = {}

        # Group episodes by family.
        by_family: dict[str, list[Episode]] = defaultdict(list)
        for ep in episodes:
            by_family[ep.task_family].append(ep)

        for family_id, eps in by_family.items():
            successes = sum(1 for e in eps
                           if e.outcome and e.outcome.task_success >= 1.0)
            variants = {e.task_variant_id for e in eps}
            strategies = self._count_strategies(eps)

            # Count active experiences/skills for this family.
            active_exps = sum(1 for e in experiences
                             if e.task_family == family_id
                             and e.validation_status == ValidationStatus.active)
            active_skls = sum(1 for s in (skills or [])
                             if s.scope == family_id
                             and s.validation_status == ValidationStatus.active)

            success_rate = successes / len(eps) if eps else 0.0
            exploration_score = self._compute_exploration_score(
                success_rate, len(eps), len(variants), strategies)

            profiles[family_id] = FamilyProfile(
                family_id=family_id,
                total_episodes=len(eps),
                successes=successes,
                failures=len(eps) - successes,
                success_rate=round(success_rate, 4),
                unique_variants_seen=len(variants),
                unique_strategies=strategies,
                active_experiences=active_exps,
                active_skills=active_skls,
                exploration_score=round(exploration_score, 4),
            )

        return profiles

    def suggest_exploration(self, profiles: dict[str, FamilyProfile],
                            max_suggestions: int = 5
                            ) -> list[ExplorationSuggestion]:
        """Generate prioritized exploration suggestions."""
        suggestions: list[ExplorationSuggestion] = []

        for fid, prof in profiles.items():
            if prof.total_episodes < self.min_episodes:
                # Not enough data to analyze — suggest basic practice.
                suggestions.append(ExplorationSuggestion(
                    family_id=fid,
                    reason=f"only {prof.total_episodes} episodes, insufficient for analysis",
                    priority=0.8,
                    suggested_action="practice",
                ))
                continue

            # High exploration score = needs more exploration.
            if prof.exploration_score > 0.7:
                suggestions.append(ExplorationSuggestion(
                    family_id=fid,
                    reason=f"high exploration score ({prof.exploration_score:.2f}): "
                           f"low success rate or low diversity",
                    priority=prof.exploration_score,
                    suggested_action="practice",
                ))

            # Low strategy diversity.
            if prof.unique_strategies <= 1 and prof.total_episodes >= self.min_episodes:
                suggestions.append(ExplorationSuggestion(
                    family_id=fid,
                    reason="only one strategy observed — risk of exploration collapse",
                    priority=0.6,
                    suggested_action="try_alternative",
                    metadata={"current_strategies": prof.unique_strategies},
                ))

            # Lots of failures but no active experiences/skills learned.
            if prof.failures > 3 and prof.active_experiences == 0:
                suggestions.append(ExplorationSuggestion(
                    family_id=fid,
                    reason=f"{prof.failures} failures but no experiences learned yet",
                    priority=0.7,
                    suggested_action="practice",
                ))

        # Sort by priority, return top-k.
        suggestions.sort(key=lambda s: s.priority, reverse=True)
        return suggestions[:max_suggestions]

    def diversity_score(self, episodes: list[Episode]) -> float:
        """Measure overall strategy diversity across all episodes.

        0.0 = all episodes use the exact same strategy (collapse risk)
        1.0 = maximum diversity (every episode uses a unique strategy)
        """
        if len(episodes) < 2:
            return 1.0
        strategies = [self._strategy_key(ep) for ep in episodes]
        unique = len(set(strategies))
        # Normalized entropy.
        counts = Counter(strategies)
        total = len(strategies)
        entropy = -sum((c / total) * math.log2(c / total)
                       for c in counts.values() if c > 0)
        max_entropy = math.log2(total) if total > 1 else 1.0
        return round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0

    def exploitation_ratio(self, episodes: list[Episode],
                           experiences: list[ExperienceObject]) -> float:
        """Fraction of episodes where existing experiences were relevant.

        High ratio = heavy exploitation (maybe missing novel approaches).
        Low ratio = lots of novel territory (good for exploration).
        """
        if not episodes or not experiences:
            return 0.0
        active_families = {e.task_family for e in experiences
                          if e.validation_status == ValidationStatus.active}
        exploiting = sum(1 for ep in episodes
                         if ep.task_family in active_families)
        return round(exploiting / len(episodes), 4)

    # ---- internals --------------------------------------------------------
    def _count_strategies(self, episodes: list[Episode]) -> int:
        """Count distinct action sequences (strategy fingerprints)."""
        strategies = {self._strategy_key(ep) for ep in episodes}
        return len(strategies)

    @staticmethod
    def _strategy_key(episode: Episode) -> str:
        """Fingerprint an episode's strategy as its action sequence."""
        if not episode.steps:
            return "no_action"
        return "|".join(s.action for s in episode.steps)

    def _compute_exploration_score(self, success_rate: float, n_episodes: int,
                                   n_variants: int, n_strategies: int) -> float:
        """Higher score = more exploration needed.

        Factors:
          - Low success rate → need more practice
          - Low variant coverage → haven't seen enough diversity
          - Low strategy diversity → risk of collapse
        """
        # Failure component (more failures = more need to explore).
        failure_score = 1.0 - success_rate

        # Coverage component (fewer variants relative to episodes = less diverse).
        coverage = n_variants / max(n_episodes, 1)
        coverage_score = 1.0 - min(1.0, coverage)

        # Strategy diversity component.
        strategy_score = 1.0 / (1.0 + n_strategies) if n_strategies > 0 else 1.0

        # Weighted combination.
        return 0.5 * failure_score + 0.3 * strategy_score + 0.2 * coverage_score
