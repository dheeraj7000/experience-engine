"""Safety / Privacy / Governance Layer (Phase 6, proposal 8.12).

Prevents the Experience Engine from learning unsafe shortcuts, leaking private
data, or promoting spurious behavioral rules. The governance layer provides:

  1. Audit log — every learning action (induction, promotion, decay, rollback)
     is recorded with timestamp, reason, and provenance.
  2. Safety classification — experiences/policies categorized by risk level.
  3. Do-not-learn zones — certain contexts are excluded from learning.
  4. Sensitivity filters — redact or block private/sensitive content.
  5. Human review thresholds — high-impact changes flagged for approval.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from schemas import ExperienceObject, PolicyObject, SkillObject
from schemas.experience import ValidationStatus


class RiskLevel(str, Enum):
    low = "low"           # routine learning, no review needed
    medium = "medium"     # unusual pattern, monitor
    high = "high"         # significant behavior change, flag for review
    critical = "critical" # safety-sensitive, block until approved


class AuditAction(str, Enum):
    experience_induced = "experience_induced"
    experience_promoted = "experience_promoted"
    experience_rejected = "experience_rejected"
    experience_decayed = "experience_decayed"
    experience_archived = "experience_archived"
    policy_created = "policy_created"
    policy_superseded = "policy_superseded"
    policy_rolled_back = "policy_rolled_back"
    skill_compiled = "skill_compiled"
    skill_validated = "skill_validated"
    contradiction_detected = "contradiction_detected"
    human_review_required = "human_review_required"
    blocked_by_governance = "blocked_by_governance"


@dataclass
class AuditEntry:
    """A single entry in the audit log."""
    action: AuditAction
    target_id: str
    target_type: str  # "experience", "policy", "skill"
    reason: str
    risk_level: RiskLevel = RiskLevel.low
    metadata: dict[str, Any] = field(default_factory=dict)
    cycle: int = 0
    approved: bool | None = None  # None = no approval needed, True/False = decision

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "reason": self.reason,
            "risk_level": self.risk_level.value,
            "metadata": self.metadata,
            "cycle": self.cycle,
            "approved": self.approved,
        }


# Keywords that indicate potentially sensitive content.
_SENSITIVE_KEYWORDS = {
    "password", "secret", "token", "api_key", "credential",
    "private", "personal", "ssn", "credit_card", "bank",
}

# Scopes considered high-risk for governance purposes.
_HIGH_RISK_SCOPES = {"security", "auth", "deletion", "production", "financial"}

# Contexts where learning should be suppressed.
_DO_NOT_LEARN_CONTEXTS = {"sandbox_test", "dry_run", "debug_mode"}


class GovernanceLayer:
    """Safety and audit layer for the Experience Engine."""

    def __init__(self) -> None:
        self.audit_log: list[AuditEntry] = []
        self._cycle: int = 0
        self._pending_reviews: list[AuditEntry] = []
        self._do_not_learn: set[str] = set(_DO_NOT_LEARN_CONTEXTS)

    @property
    def cycle(self) -> int:
        return self._cycle

    def advance_cycle(self) -> None:
        self._cycle += 1

    # ---- audit logging ----------------------------------------------------
    def log(self, action: AuditAction, target_id: str, target_type: str,
            reason: str, risk_level: RiskLevel = RiskLevel.low,
            metadata: dict[str, Any] | None = None) -> AuditEntry:
        """Record an action in the audit log."""
        entry = AuditEntry(
            action=action, target_id=target_id, target_type=target_type,
            reason=reason, risk_level=risk_level,
            metadata=metadata or {}, cycle=self._cycle,
        )
        self.audit_log.append(entry)
        if risk_level in (RiskLevel.high, RiskLevel.critical):
            self._pending_reviews.append(entry)
        return entry

    # ---- safety checks ----------------------------------------------------
    def assess_risk(self, experience: ExperienceObject) -> RiskLevel:
        """Classify the risk level of an experience being promoted."""
        # Check scope.
        if experience.task_family.lower() in _HIGH_RISK_SCOPES:
            return RiskLevel.high

        # Check for sensitive content in lesson.
        text = (experience.lesson + " " + experience.recommended_policy).lower()
        if any(kw in text for kw in _SENSITIVE_KEYWORDS):
            return RiskLevel.critical

        # Low confidence + high contradictions = risky.
        if experience.contradictions >= 3 and experience.confidence < 0.5:
            return RiskLevel.medium

        # High evidence, low contradictions, non-sensitive = low risk.
        return RiskLevel.low

    def assess_policy_risk(self, policy: PolicyObject) -> RiskLevel:
        """Classify risk for a policy being activated."""
        if policy.scope.lower() in _HIGH_RISK_SCOPES:
            return RiskLevel.high
        text = policy.behavior.lower()
        if any(kw in text for kw in _SENSITIVE_KEYWORDS):
            return RiskLevel.critical
        if policy.confidence < 0.4:
            return RiskLevel.medium
        return RiskLevel.low

    def should_block(self, experience: ExperienceObject) -> bool:
        """Returns True if this experience should be blocked from promotion."""
        risk = self.assess_risk(experience)
        if risk == RiskLevel.critical:
            self.log(
                AuditAction.blocked_by_governance,
                experience.experience_id, "experience",
                "blocked: contains sensitive content or critical-risk scope",
                risk_level=RiskLevel.critical,
            )
            return True
        return False

    def requires_human_review(self, experience: ExperienceObject) -> bool:
        """Returns True if this experience needs human approval before promotion."""
        risk = self.assess_risk(experience)
        if risk in (RiskLevel.high, RiskLevel.critical):
            self.log(
                AuditAction.human_review_required,
                experience.experience_id, "experience",
                f"high-risk experience requires review (risk={risk.value})",
                risk_level=risk,
            )
            return True
        return False

    # ---- do-not-learn zones -----------------------------------------------
    def should_learn(self, context: dict[str, Any]) -> bool:
        """Returns True if learning is allowed in this context."""
        for key in ("mode", "context_type", "environment"):
            val = str(context.get(key, "")).lower()
            if val in self._do_not_learn:
                return False
        return True

    def add_do_not_learn(self, context_value: str) -> None:
        """Add a context to the do-not-learn set."""
        self._do_not_learn.add(context_value.lower())

    # ---- sensitivity filters ----------------------------------------------
    def contains_sensitive_content(self, text: str) -> bool:
        """Check if text contains potentially sensitive/private information."""
        lower = text.lower()
        return any(kw in lower for kw in _SENSITIVE_KEYWORDS)

    def redact_sensitive(self, text: str) -> str:
        """Replace sensitive keywords with [REDACTED] in text."""
        result = text
        for kw in _SENSITIVE_KEYWORDS:
            # Case-insensitive replacement.
            import re
            result = re.sub(re.escape(kw), "[REDACTED]", result, flags=re.IGNORECASE)
        return result

    # ---- review queue ------------------------------------------------------
    def pending_reviews(self) -> list[AuditEntry]:
        """Return items waiting for human review."""
        return list(self._pending_reviews)

    def approve(self, target_id: str) -> bool:
        """Approve a pending review item."""
        for entry in self._pending_reviews:
            if entry.target_id == target_id:
                entry.approved = True
                self._pending_reviews.remove(entry)
                return True
        return False

    def reject(self, target_id: str) -> bool:
        """Reject a pending review item."""
        for entry in self._pending_reviews:
            if entry.target_id == target_id:
                entry.approved = False
                self._pending_reviews.remove(entry)
                return True
        return False

    # ---- summary -----------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Overview of governance state."""
        actions_by_type = {}
        for entry in self.audit_log:
            actions_by_type[entry.action.value] = actions_by_type.get(
                entry.action.value, 0) + 1
        return {
            "total_entries": len(self.audit_log),
            "pending_reviews": len(self._pending_reviews),
            "actions_by_type": actions_by_type,
            "current_cycle": self._cycle,
            "do_not_learn_zones": sorted(self._do_not_learn),
        }
