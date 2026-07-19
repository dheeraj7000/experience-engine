"""Experience Validator (proposal 8.7) — the anti-self-delusion gate.

A candidate experience does NOT change behavior until it demonstrably improves
performance on HELD-OUT replays. This is what separates the Experience Engine
from raw reflection, and it doubles as a regression test on the agent's own
learning:

    a promoted experience must never reduce held-out performance.
"""
from __future__ import annotations

from schemas import ExperienceObject
from schemas.experience import ValidationStatus


class ExperienceValidator:
    def __init__(self, min_delta: float = 0.0) -> None:
        # Require strictly non-negative improvement; raise for higher bar.
        self.min_delta = min_delta

    def validate(self, experience: ExperienceObject, replay_fn, now: str | None = None) -> bool:
        """
        replay_fn(experience) -> (baseline_score, with_experience_score)
        Runs the agent on held-out variants WITHOUT and WITH the candidate
        injected, and compares. Returns True and marks the experience active
        iff it helps (or at least does not hurt).
        """
        if replay_fn is None:
            # No replay available: keep as provisional, never active.
            experience.validation_status = ValidationStatus.provisional
            return False

        baseline, improved = replay_fn(experience)
        delta = improved - baseline
        experience.last_validated = now
        if delta >= self.min_delta and delta >= 0:
            experience.validation_status = ValidationStatus.active
            return True
        experience.validation_status = ValidationStatus.rejected
        return False
