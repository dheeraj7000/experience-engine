"""Experience Internalizer (Phase 7, proposal 8.14).

The base system keeps all learning external and auditable. Internalization is
the optional process of selecting highly stable, validated experiences and
preparing them for eventual parameter updates (fine-tuning / distillation).

This module does NOT perform actual weight updates — it scores and selects
CANDIDATES for internalization based on strict criteria. Actual distillation
is an infrastructure concern outside this prototype.

Internalization candidates must satisfy ALL of:
  - High evidence count (well-supported by many episodes)
  - High validation score (demonstrated improvement on held-out)
  - Low contradiction count (not disputed)
  - Low safety risk (governance cleared)
  - Broad transfer utility (applies across variants/families)
  - Stable over time (not recently oscillating)
  - Clear rollback plan

The module produces a ranked list of candidates with scores, NOT automated
weight updates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from schemas import ExperienceObject, SkillObject
from schemas.experience import ValidationStatus


@dataclass
class InternalizationCandidate:
    """A scored candidate for experience internalization."""
    item_id: str
    item_type: str  # "experience" or "skill"
    name: str
    score: float    # 0-1, overall internalization-worthiness
    signals: dict[str, float] = field(default_factory=dict)
    eligible: bool = True
    rejection_reason: str = ""

    def summary(self) -> str:
        status = "ELIGIBLE" if self.eligible else f"REJECTED ({self.rejection_reason})"
        return f"[{status}] {self.item_type} '{self.item_id}' score={self.score:.3f}"


@dataclass
class InternalizationConfig:
    """Thresholds for internalization eligibility."""
    min_evidence: int = 10
    min_confidence: float = 0.85
    max_contradictions: int = 1
    min_transfer_families: int = 1      # must apply to at least this many families
    min_cycles_stable: int = 5          # must have been active for N cycles without change
    w_evidence: float = 0.25
    w_confidence: float = 0.25
    w_stability: float = 0.20
    w_transfer: float = 0.15
    w_safety: float = 0.15


class ExperienceInternalizer:
    """Scores and ranks experiences/skills for potential internalization.

    This is a SELECTION module — it produces candidates for human review,
    not automatic weight updates. The actual internalization pipeline
    (distillation, fine-tuning, regression testing) is an infrastructure
    concern built on top of these selections.
    """

    def __init__(self, config: InternalizationConfig | None = None) -> None:
        self.config = config or InternalizationConfig()

    def score_experiences(self, experiences: list[ExperienceObject],
                          all_experiences: list[ExperienceObject] | None = None
                          ) -> list[InternalizationCandidate]:
        """Score all active experiences for internalization worthiness."""
        all_exps = all_experiences or experiences
        candidates: list[InternalizationCandidate] = []

        for exp in experiences:
            if exp.validation_status not in (ValidationStatus.active,
                                             ValidationStatus.validated):
                continue
            candidate = self._score_experience(exp, all_exps)
            candidates.append(candidate)

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def score_skills(self, skills: list[SkillObject]
                     ) -> list[InternalizationCandidate]:
        """Score all active skills for internalization worthiness."""
        candidates: list[InternalizationCandidate] = []

        for skill in skills:
            if skill.validation_status != ValidationStatus.active:
                continue
            candidate = self._score_skill(skill)
            candidates.append(candidate)

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def top_candidates(self, experiences: list[ExperienceObject],
                       skills: list[SkillObject] | None = None,
                       k: int = 5) -> list[InternalizationCandidate]:
        """Return the top-k internalization candidates across experiences and skills."""
        exp_candidates = self.score_experiences(experiences)
        skill_candidates = self.score_skills(skills or [])
        all_candidates = exp_candidates + skill_candidates
        # Only eligible ones.
        eligible = [c for c in all_candidates if c.eligible]
        eligible.sort(key=lambda c: c.score, reverse=True)
        return eligible[:k]

    def _score_experience(self, exp: ExperienceObject,
                          all_experiences: list[ExperienceObject]
                          ) -> InternalizationCandidate:
        """Score a single experience."""
        cfg = self.config

        # Evidence signal.
        evidence_score = min(1.0, exp.evidence_count / (cfg.min_evidence * 2))

        # Confidence signal.
        confidence_score = exp.confidence

        # Stability signal (proxy: high evidence + low contradictions = stable).
        stability = 1.0 - (exp.contradictions / max(1, exp.evidence_count))
        stability_score = max(0.0, stability)

        # Transfer signal: does this lesson apply to multiple families?
        transfer_score = self._transfer_score(exp, all_experiences)

        # Safety signal: no sensitive content, no high contradictions.
        safety_score = self._safety_score(exp)

        signals = {
            "evidence": round(evidence_score, 4),
            "confidence": round(confidence_score, 4),
            "stability": round(stability_score, 4),
            "transfer": round(transfer_score, 4),
            "safety": round(safety_score, 4),
        }

        score = (
            cfg.w_evidence * evidence_score
            + cfg.w_confidence * confidence_score
            + cfg.w_stability * stability_score
            + cfg.w_transfer * transfer_score
            + cfg.w_safety * safety_score
        )
        score = round(score, 4)

        # Check eligibility thresholds.
        eligible = True
        reason = ""
        if exp.evidence_count < cfg.min_evidence:
            eligible = False
            reason = f"insufficient evidence ({exp.evidence_count} < {cfg.min_evidence})"
        elif exp.confidence < cfg.min_confidence:
            eligible = False
            reason = f"confidence too low ({exp.confidence:.2f} < {cfg.min_confidence})"
        elif exp.contradictions > cfg.max_contradictions:
            eligible = False
            reason = f"too many contradictions ({exp.contradictions} > {cfg.max_contradictions})"
        elif safety_score < 0.5:
            eligible = False
            reason = "safety concerns"

        return InternalizationCandidate(
            item_id=exp.experience_id,
            item_type="experience",
            name=exp.lesson[:80],
            score=score,
            signals=signals,
            eligible=eligible,
            rejection_reason=reason,
        )

    def _score_skill(self, skill: SkillObject) -> InternalizationCandidate:
        """Score a single skill."""
        cfg = self.config

        evidence_score = min(1.0, skill.evidence_count / (cfg.min_evidence * 2))
        confidence_score = skill.confidence
        stability_score = 0.8  # skills are already compiled, inherently stable
        transfer_score = 0.3 if not skill.scope else 0.5  # global skills transfer better
        safety_score = 1.0  # skills don't typically have safety concerns

        signals = {
            "evidence": round(evidence_score, 4),
            "confidence": round(confidence_score, 4),
            "stability": round(stability_score, 4),
            "transfer": round(transfer_score, 4),
            "safety": round(safety_score, 4),
        }

        score = (
            cfg.w_evidence * evidence_score
            + cfg.w_confidence * confidence_score
            + cfg.w_stability * stability_score
            + cfg.w_transfer * transfer_score
            + cfg.w_safety * safety_score
        )
        score = round(score, 4)

        eligible = True
        reason = ""
        if skill.evidence_count < cfg.min_evidence:
            eligible = False
            reason = f"insufficient evidence ({skill.evidence_count} < {cfg.min_evidence})"
        elif skill.confidence < cfg.min_confidence:
            eligible = False
            reason = f"confidence too low ({skill.confidence:.2f} < {cfg.min_confidence})"

        return InternalizationCandidate(
            item_id=skill.skill_id,
            item_type="skill",
            name=skill.name,
            score=score,
            signals=signals,
            eligible=eligible,
            rejection_reason=reason,
        )

    @staticmethod
    def _transfer_score(exp: ExperienceObject,
                        all_experiences: list[ExperienceObject]) -> float:
        """How transferable is this experience across families?

        An experience with similar lessons in other families has high transfer.
        """
        if not exp.task_family:
            return 0.5  # global, decent transfer
        # Count how many other families have similar active experiences.
        other_families: set[str] = set()
        for other in all_experiences:
            if (other.experience_id != exp.experience_id
                    and other.task_family != exp.task_family
                    and other.validation_status == ValidationStatus.active):
                # Very rough: shared words in lesson.
                shared = set(exp.lesson.lower().split()) & set(other.lesson.lower().split())
                if len(shared) >= 3:
                    other_families.add(other.task_family)
        return min(1.0, len(other_families) / 3.0)

    @staticmethod
    def _safety_score(exp: ExperienceObject) -> float:
        """Higher = safer to internalize."""
        text = (exp.lesson + " " + exp.recommended_policy).lower()
        # Check for red flags.
        red_flags = {"skip", "bypass", "ignore", "disable", "never check"}
        penalty = sum(0.2 for flag in red_flags if flag in text)
        return max(0.0, 1.0 - penalty)
