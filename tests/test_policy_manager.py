"""Tests for Phase 5: upgraded Policy Manager — conflicts, versioning, safety gating."""
from schemas import ExperienceObject, PolicyObject, SkillObject
from schemas.experience import ValidationStatus
from persistence.a2_engine.policy import PolicyManager, PolicyConflict


def _experience(confidence=0.7, family="test", conditions=None):
    return ExperienceObject(
        experience_id="exp_1", task_family=family,
        context_conditions=conditions or {"family": family},
        lesson="verify output", recommended_policy="always verify output",
        confidence=confidence, evidence_count=5,
        validation_status=ValidationStatus.active,
    )


def _policy(pid, scope="test", priority=70, behavior="do something",
            triggers=None, confidence=0.8):
    return PolicyObject(
        policy_id=pid, scope=scope,
        trigger_conditions=triggers or ["applicable"],
        behavior=behavior, priority=priority, confidence=confidence,
        supporting_experiences=["exp_1"],
        validation_status=ValidationStatus.active,
    )


def test_promote_creates_active_policy():
    mgr = PolicyManager()
    exp = _experience(confidence=0.8)
    pol = mgr.promote(exp)
    assert pol.validation_status == ValidationStatus.active
    assert pol.scope == "test"
    assert "verify" in pol.behavior.lower()


def test_promote_with_skill_references_skill():
    mgr = PolicyManager()
    exp = _experience(confidence=0.8)
    skill = SkillObject(
        skill_id="s1", name="verify_workflow", scope="test",
        workflow=["check", "validate", "submit"],
        preconditions=["task_family=test"],
    )
    pol = mgr.promote(exp, skill=skill)
    assert "verify_workflow" in pol.behavior
    assert "Skill" in pol.behavior or "skill" in pol.behavior


def test_safety_gating_blocks_low_confidence():
    mgr = PolicyManager(safety_threshold=0.9)
    # "security" is a safety-critical scope.
    exp = _experience(confidence=0.7, family="security")
    pol = mgr.promote(exp)
    # Should NOT be active — confidence below safety threshold.
    assert pol.validation_status == ValidationStatus.candidate


def test_safety_gating_allows_high_confidence():
    mgr = PolicyManager(safety_threshold=0.85)
    exp = _experience(confidence=0.9, family="security")
    pol = mgr.promote(exp)
    assert pol.validation_status == ValidationStatus.active


def test_detect_conflicts_same_scope():
    mgr = PolicyManager()
    a = _policy("pol_a", scope="test", priority=80, behavior="always search first")
    b = _policy("pol_b", scope="test", priority=60, behavior="reason first")
    conflicts = mgr.detect_conflicts([a, b])
    assert len(conflicts) == 1
    assert conflicts[0].winner() is a  # higher priority


def test_no_conflict_different_scopes():
    mgr = PolicyManager()
    a = _policy("pol_a", scope="sql", priority=80)
    b = _policy("pol_b", scope="csv", priority=60)
    conflicts = mgr.detect_conflicts([a, b])
    assert conflicts == []


def test_resolve_conflicts_adjusts_priority():
    mgr = PolicyManager()
    a = _policy("pol_a", scope="test", priority=70, confidence=0.9)
    b = _policy("pol_b", scope="test", priority=70, confidence=0.6)
    conflicts = mgr.detect_conflicts([a, b])
    actions = mgr.resolve_conflicts(conflicts, [a, b])
    assert actions  # something was done
    # The lower-confidence one should have reduced priority.
    assert b.priority < 70


def test_supersede_deprecates_old():
    mgr = PolicyManager()
    old = _policy("pol_old")
    new = _policy("pol_new")
    mgr.supersede(old, new)
    assert old.validation_status == ValidationStatus.deprecated
    # New policy inherits evidence.
    assert "exp_1" in new.supporting_experiences


def test_rollback_deprecates_policy():
    mgr = PolicyManager()
    pol = _policy("pol_1")
    assert pol.validation_status == ValidationStatus.active
    mgr.rollback(pol)
    assert pol.validation_status == ValidationStatus.deprecated


def test_refine_scope_creates_new_version():
    mgr = PolicyManager()
    pol = _policy("pol_1", triggers=["family=test"])
    refined = mgr.refine_scope(pol, "failure_type=runtime_error")
    # Old is deprecated.
    assert pol.validation_status == ValidationStatus.deprecated
    # New has the additional condition.
    assert "failure_type=runtime_error" in refined.trigger_conditions
    assert "family=test" in refined.trigger_conditions
    assert refined.validation_status == ValidationStatus.active


def test_ordered_policies_respects_priority():
    mgr = PolicyManager()
    lo = _policy("pol_lo", priority=40)
    hi = _policy("pol_hi", priority=90)
    mid = _policy("pol_mid", priority=60)
    ordered = mgr.ordered_policies([lo, hi, mid], scope="test")
    assert [p.policy_id for p in ordered] == ["pol_hi", "pol_mid", "pol_lo"]


def test_ordered_policies_filters_inactive():
    mgr = PolicyManager()
    active = _policy("pol_active", priority=50)
    deprecated = _policy("pol_dep", priority=90)
    deprecated.validation_status = ValidationStatus.deprecated
    ordered = mgr.ordered_policies([active, deprecated], scope="test")
    assert len(ordered) == 1
    assert ordered[0].policy_id == "pol_active"


def test_promote_from_skill():
    mgr = PolicyManager()
    skill = SkillObject(
        skill_id="s1", name="bug_fix_workflow", scope="test",
        workflow=["analyze", "fix", "verify"],
        preconditions=["task_family=test"],
        confidence=0.85,
        validation_status=ValidationStatus.active,
    )
    pol = mgr.promote_from_skill(skill)
    assert pol.validation_status == ValidationStatus.active
    assert "bug_fix_workflow" in pol.behavior
    assert pol.scope == "test"
