"""Tests for Phase 7: Experience Internalizer — scoring and candidate selection."""
from schemas import ExperienceObject, SkillObject
from schemas.experience import ValidationStatus
from persistence.a2_engine.internalizer import (
    ExperienceInternalizer, InternalizationCandidate, InternalizationConfig,
)


def _experience(eid, confidence=0.9, evidence=15, contradictions=0,
                lesson="verify output before submitting", family="test"):
    return ExperienceObject(
        experience_id=eid, task_family=family,
        lesson=lesson, recommended_policy=lesson,
        confidence=confidence, evidence_count=evidence,
        contradictions=contradictions,
        validation_status=ValidationStatus.active,
    )


def _skill(sid, confidence=0.9, evidence=12):
    return SkillObject(
        skill_id=sid, name="test_skill", scope="test",
        workflow=["step1", "step2"], confidence=confidence,
        evidence_count=evidence,
        validation_status=ValidationStatus.active,
    )


def test_high_quality_experience_is_eligible():
    internalizer = ExperienceInternalizer()
    exp = _experience("exp_1", confidence=0.92, evidence=20, contradictions=0)
    candidates = internalizer.score_experiences([exp])
    assert len(candidates) == 1
    assert candidates[0].eligible is True
    assert candidates[0].score > 0.5


def test_low_evidence_is_ineligible():
    internalizer = ExperienceInternalizer(config=InternalizationConfig(min_evidence=10))
    exp = _experience("exp_1", evidence=3)
    candidates = internalizer.score_experiences([exp])
    assert candidates[0].eligible is False
    assert "evidence" in candidates[0].rejection_reason


def test_low_confidence_is_ineligible():
    internalizer = ExperienceInternalizer(config=InternalizationConfig(min_confidence=0.85))
    exp = _experience("exp_1", confidence=0.6, evidence=20)
    candidates = internalizer.score_experiences([exp])
    assert candidates[0].eligible is False
    assert "confidence" in candidates[0].rejection_reason


def test_high_contradictions_is_ineligible():
    internalizer = ExperienceInternalizer(config=InternalizationConfig(max_contradictions=1))
    exp = _experience("exp_1", contradictions=5, evidence=20)
    candidates = internalizer.score_experiences([exp])
    assert candidates[0].eligible is False
    assert "contradiction" in candidates[0].rejection_reason


def test_unsafe_lesson_is_ineligible():
    internalizer = ExperienceInternalizer()
    exp = _experience("exp_1",
                      lesson="skip validation and bypass checks and ignore errors",
                      evidence=20)
    candidates = internalizer.score_experiences([exp])
    assert candidates[0].eligible is False
    assert "safety" in candidates[0].rejection_reason


def test_skill_scoring():
    internalizer = ExperienceInternalizer()
    skill = _skill("s1", confidence=0.92, evidence=15)
    candidates = internalizer.score_skills([skill])
    assert len(candidates) == 1
    assert candidates[0].eligible is True
    assert candidates[0].item_type == "skill"


def test_top_candidates_mixed():
    internalizer = ExperienceInternalizer()
    exps = [
        _experience("exp_good", confidence=0.95, evidence=25),
        _experience("exp_bad", confidence=0.4, evidence=2),
    ]
    skills = [_skill("s1", confidence=0.9, evidence=15)]
    top = internalizer.top_candidates(exps, skills, k=3)
    # Only eligible ones returned.
    assert all(c.eligible for c in top)
    # Should include the good experience and the skill.
    ids = [c.item_id for c in top]
    assert "exp_good" in ids
    assert "s1" in ids
    assert "exp_bad" not in ids


def test_ranking_by_score():
    internalizer = ExperienceInternalizer()
    exps = [
        _experience("exp_hi", confidence=0.95, evidence=30),
        _experience("exp_lo", confidence=0.86, evidence=12),
    ]
    candidates = internalizer.score_experiences(exps)
    assert candidates[0].item_id == "exp_hi"
    assert candidates[0].score > candidates[1].score


def test_transfer_score_cross_family():
    internalizer = ExperienceInternalizer()
    # exp_a has similar lesson to exp_b in different family → transfer.
    exp_a = _experience("exp_a", family="sql",
                        lesson="verify schema before writing query code")
    exp_b = _experience("exp_b", family="csv",
                        lesson="verify schema before writing processing code")
    candidates = internalizer.score_experiences([exp_a], all_experiences=[exp_a, exp_b])
    # Transfer signal should be non-zero.
    assert candidates[0].signals["transfer"] > 0


def test_skips_non_active():
    internalizer = ExperienceInternalizer()
    exp = _experience("exp_1", evidence=20)
    exp.validation_status = ValidationStatus.rejected
    candidates = internalizer.score_experiences([exp])
    assert candidates == []


def test_candidate_summary():
    candidate = InternalizationCandidate(
        item_id="exp_1", item_type="experience",
        name="test lesson", score=0.85, eligible=True,
    )
    summary = candidate.summary()
    assert "ELIGIBLE" in summary
    assert "0.85" in summary
