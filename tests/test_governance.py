"""Tests for Phase 6: Governance Layer — audit, safety, sensitivity, do-not-learn."""
from schemas import ExperienceObject, PolicyObject
from schemas.experience import ValidationStatus
from persistence.a2_engine.governance import (
    GovernanceLayer, AuditAction, AuditEntry, RiskLevel,
)


def _experience(eid, family="test", lesson="verify output", confidence=0.7):
    return ExperienceObject(
        experience_id=eid, task_family=family,
        lesson=lesson, recommended_policy=lesson,
        confidence=confidence, evidence_count=3,
        validation_status=ValidationStatus.active,
    )


def _policy(pid, scope="test", behavior="do something", confidence=0.8):
    return PolicyObject(
        policy_id=pid, scope=scope, behavior=behavior,
        confidence=confidence, priority=70,
        validation_status=ValidationStatus.active,
    )


def test_audit_log_records_actions():
    gov = GovernanceLayer()
    gov.log(AuditAction.experience_induced, "exp_1", "experience", "test")
    assert len(gov.audit_log) == 1
    assert gov.audit_log[0].action == AuditAction.experience_induced


def test_risk_assessment_low_for_normal():
    gov = GovernanceLayer()
    exp = _experience("exp_1", family="test", lesson="verify output")
    assert gov.assess_risk(exp) == RiskLevel.low


def test_risk_assessment_high_for_critical_scope():
    gov = GovernanceLayer()
    exp = _experience("exp_1", family="security", lesson="skip auth check")
    assert gov.assess_risk(exp) == RiskLevel.high


def test_risk_assessment_critical_for_sensitive_content():
    gov = GovernanceLayer()
    exp = _experience("exp_1", lesson="store the user password in plain text")
    assert gov.assess_risk(exp) == RiskLevel.critical


def test_should_block_sensitive_experience():
    gov = GovernanceLayer()
    exp = _experience("exp_1", lesson="cache the api_key for reuse")
    assert gov.should_block(exp) is True
    # Should be in audit log.
    assert any(e.action == AuditAction.blocked_by_governance for e in gov.audit_log)


def test_should_not_block_normal_experience():
    gov = GovernanceLayer()
    exp = _experience("exp_1", lesson="verify output before submitting")
    assert gov.should_block(exp) is False


def test_requires_human_review_for_high_risk():
    gov = GovernanceLayer()
    exp = _experience("exp_1", family="security", lesson="disable auth check")
    assert gov.requires_human_review(exp) is True


def test_do_not_learn_blocks_context():
    gov = GovernanceLayer()
    assert gov.should_learn({"mode": "dry_run"}) is False
    assert gov.should_learn({"mode": "production"}) is True
    assert gov.should_learn({"context_type": "sandbox_test"}) is False


def test_add_custom_do_not_learn():
    gov = GovernanceLayer()
    gov.add_do_not_learn("staging")
    assert gov.should_learn({"mode": "staging"}) is False


def test_sensitivity_detection():
    gov = GovernanceLayer()
    assert gov.contains_sensitive_content("store the password") is True
    assert gov.contains_sensitive_content("fix the output") is False


def test_redact_sensitive():
    gov = GovernanceLayer()
    text = "Store the API_KEY and password in config"
    redacted = gov.redact_sensitive(text)
    assert "API_KEY" not in redacted
    assert "password" not in redacted.lower()
    assert "[REDACTED]" in redacted


def test_pending_reviews_and_approval():
    gov = GovernanceLayer()
    gov.log(AuditAction.human_review_required, "exp_1", "experience",
            "needs review", risk_level=RiskLevel.high)
    assert len(gov.pending_reviews()) == 1
    gov.approve("exp_1")
    assert len(gov.pending_reviews()) == 0


def test_pending_reviews_rejection():
    gov = GovernanceLayer()
    gov.log(AuditAction.human_review_required, "exp_1", "experience",
            "needs review", risk_level=RiskLevel.high)
    gov.reject("exp_1")
    assert len(gov.pending_reviews()) == 0
    # Check that the entry was marked as rejected.
    entry = [e for e in gov.audit_log if e.target_id == "exp_1"][0]
    assert entry.approved is False


def test_policy_risk_assessment():
    gov = GovernanceLayer()
    normal = _policy("p1", scope="test", behavior="verify output")
    assert gov.assess_policy_risk(normal) == RiskLevel.low
    sensitive = _policy("p2", scope="auth", behavior="skip token validation")
    assert gov.assess_policy_risk(sensitive) == RiskLevel.high


def test_summary():
    gov = GovernanceLayer()
    gov.log(AuditAction.experience_induced, "exp_1", "experience", "test")
    gov.log(AuditAction.experience_promoted, "exp_1", "experience", "promoted")
    summary = gov.summary()
    assert summary["total_entries"] == 2
    assert summary["actions_by_type"]["experience_induced"] == 1


def test_advance_cycle():
    gov = GovernanceLayer()
    assert gov.cycle == 0
    gov.advance_cycle()
    assert gov.cycle == 1
