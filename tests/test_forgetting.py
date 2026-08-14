"""Tests for Phase 6: Forgetting Manager — decay, staleness, archival."""
from schemas import ExperienceObject, PolicyObject, SkillObject
from schemas.experience import ValidationStatus
from persistence.a2_engine.forgetting import (
    ForgettingManager, DecayConfig, ForgettingAction,
)


def _experience(eid, confidence=0.7, evidence=3, contradictions=0,
                status=ValidationStatus.active):
    return ExperienceObject(
        experience_id=eid, task_family="test",
        lesson="some lesson", confidence=confidence,
        evidence_count=evidence, contradictions=contradictions,
        validation_status=status,
    )


def _policy(pid, confidence=0.8):
    return PolicyObject(
        policy_id=pid, scope="test", behavior="do something",
        confidence=confidence, priority=70,
        validation_status=ValidationStatus.active,
    )


def _skill(sid, confidence=0.7, evidence=3):
    return SkillObject(
        skill_id=sid, name="test_skill", scope="test",
        workflow=["step1"], confidence=confidence,
        evidence_count=evidence,
        validation_status=ValidationStatus.active,
    )


def test_no_decay_on_first_cycle():
    mgr = ForgettingManager()
    exp = _experience("exp_1", confidence=0.8)
    actions = mgr.apply_decay([exp], [], [])
    # First cycle registers items, no decay applied.
    assert all(a.action != "decayed" for a in actions)
    assert exp.confidence == 0.8


def test_decay_reduces_confidence_over_cycles():
    mgr = ForgettingManager(config=DecayConfig(base_decay_rate=0.05))
    exp = _experience("exp_1", confidence=0.8)
    # First cycle: register.
    mgr.apply_decay([exp], [], [])
    # Second cycle: actual decay.
    actions = mgr.apply_decay([exp], [], [])
    assert exp.confidence < 0.8
    assert any(a.action == "decayed" for a in actions)


def test_reinforcement_resets_decay():
    mgr = ForgettingManager(config=DecayConfig(base_decay_rate=0.1))
    exp = _experience("exp_1", confidence=0.8)
    mgr.apply_decay([exp], [], [])  # register
    mgr.mark_reinforced("exp_1")     # just reinforced
    actions = mgr.apply_decay([exp], [], [])
    # Should NOT decay — was just reinforced this cycle.
    assert exp.confidence == 0.8


def test_archival_below_threshold():
    cfg = DecayConfig(base_decay_rate=0.5, archive_threshold=0.3)
    mgr = ForgettingManager(config=cfg)
    exp = _experience("exp_1", confidence=0.3)
    mgr.apply_decay([exp], [], [])  # register
    actions = mgr.apply_decay([exp], [], [])
    # Should be archived (confidence drops below threshold).
    assert exp.validation_status == ValidationStatus.deprecated
    assert any(a.action == "archived" for a in actions)


def test_more_evidence_slower_decay():
    cfg = DecayConfig(base_decay_rate=0.1, evidence_halflife=5)
    mgr = ForgettingManager(config=cfg)
    low_evidence = _experience("exp_lo", confidence=0.8, evidence=1)
    high_evidence = _experience("exp_hi", confidence=0.8, evidence=20)
    mgr.apply_decay([low_evidence, high_evidence], [], [])  # register
    mgr.apply_decay([low_evidence, high_evidence], [], [])  # decay
    # High evidence should decay less.
    assert high_evidence.confidence > low_evidence.confidence


def test_contradictions_accelerate_decay():
    cfg = DecayConfig(base_decay_rate=0.05, contradiction_decay_multiplier=3.0)
    mgr = ForgettingManager(config=cfg)
    clean = _experience("exp_clean", confidence=0.8, contradictions=0)
    dirty = _experience("exp_dirty", confidence=0.8, contradictions=5)
    mgr.apply_decay([clean, dirty], [], [])  # register
    mgr.apply_decay([clean, dirty], [], [])  # decay
    assert dirty.confidence < clean.confidence


def test_staleness_detection():
    cfg = DecayConfig(staleness_cycles=3)
    mgr = ForgettingManager(config=cfg)
    exp = _experience("exp_1")
    # Simulate 3 cycles without reinforcement.
    mgr._cycles_since_reinforced["exp_1"] = 3
    stale = mgr.stale_items([exp])
    assert "exp_1" in stale


def test_revalidation_due():
    cfg = DecayConfig(revalidation_interval=5)
    mgr = ForgettingManager(config=cfg)
    exp = _experience("exp_1")
    mgr._cycles_since_reinforced["exp_1"] = 5
    due = mgr.items_due_revalidation([exp])
    assert "exp_1" in due


def test_skips_already_deprecated():
    mgr = ForgettingManager()
    exp = _experience("exp_1", status=ValidationStatus.deprecated)
    mgr.apply_decay([exp], [], [])
    mgr.apply_decay([exp], [], [])
    # Should not further decay deprecated items.
    assert exp.validation_status == ValidationStatus.deprecated


def test_policy_decay():
    cfg = DecayConfig(base_decay_rate=0.1)
    mgr = ForgettingManager(config=cfg)
    pol = _policy("pol_1", confidence=0.6)
    mgr.apply_decay([], [pol], [])  # register
    mgr.apply_decay([], [pol], [])  # decay
    assert pol.confidence < 0.6


def test_skill_decay():
    cfg = DecayConfig(base_decay_rate=0.1)
    mgr = ForgettingManager(config=cfg)
    skill = _skill("s1", confidence=0.6)
    mgr.apply_decay([], [], [skill])  # register
    mgr.apply_decay([], [], [skill])  # decay
    assert skill.confidence < 0.6
