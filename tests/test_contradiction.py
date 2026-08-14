"""Tests for Phase 2: contradiction mining between experiences."""
from schemas import ExperienceObject
from schemas.experience import ValidationStatus
from persistence.a2_engine.contradiction import ContradictionMiner, Contradiction


def _experience(eid, lesson, family="test", conditions=None, confidence=0.7,
                status=ValidationStatus.active):
    return ExperienceObject(
        experience_id=eid,
        task_family=family,
        context_conditions=conditions or {"family": family},
        lesson=lesson,
        recommended_policy=lesson,
        confidence=confidence,
        evidence_count=3,
        validation_status=status,
    )


def test_detects_opposing_lessons():
    miner = ContradictionMiner(min_severity=0.2)
    a = _experience("exp_a", "always inspect schema before querying")
    b = _experience("exp_b", "skip schema inspection to save time")
    contradictions = miner.detect([a, b])
    assert len(contradictions) >= 1
    c = contradictions[0]
    assert c.severity > 0
    assert c.experience_a_id in ("exp_a", "exp_b")
    assert c.experience_b_id in ("exp_a", "exp_b")


def test_no_contradiction_for_different_families():
    miner = ContradictionMiner(min_severity=0.2)
    a = _experience("exp_a", "always inspect schema", family="sql")
    b = _experience("exp_b", "skip schema inspection", family="math")
    contradictions = miner.detect([a, b])
    assert len(contradictions) == 0


def test_no_contradiction_for_non_opposing_lessons():
    miner = ContradictionMiner(min_severity=0.2)
    a = _experience("exp_a", "use chunked processing for large files")
    b = _experience("exp_b", "verify file format before processing")
    contradictions = miner.detect([a, b])
    assert len(contradictions) == 0


def test_only_scans_active_experiences():
    miner = ContradictionMiner(min_severity=0.2)
    a = _experience("exp_a", "always inspect first", status=ValidationStatus.active)
    b = _experience("exp_b", "never inspect first", status=ValidationStatus.rejected)
    contradictions = miner.detect([a, b])
    assert len(contradictions) == 0  # b is rejected, not considered


def test_apply_contradictions_reduces_confidence():
    miner = ContradictionMiner(min_severity=0.2)
    a = _experience("exp_a", "always inspect schema before querying")
    b = _experience("exp_b", "skip schema inspection to save time")
    original_conf_a = a.confidence
    original_conf_b = b.confidence

    contradictions = miner.detect([a, b])
    assert contradictions  # sanity
    miner.apply_contradictions([a, b], contradictions)

    assert a.confidence < original_conf_a
    assert b.confidence < original_conf_b
    assert a.contradictions >= 1
    assert b.contradictions >= 1


def test_resolution_hint_prefers_more_evidence():
    miner = ContradictionMiner(min_severity=0.2)
    a = _experience("exp_a", "always ensure validation first")
    a.evidence_count = 20
    b = _experience("exp_b", "skip validation to save time")
    b.evidence_count = 3
    contradictions = miner.detect([a, b])
    assert contradictions
    hint = contradictions[0].resolution_hint
    # Should suggest narrowing or deprecating the less-supported one.
    assert "exp_a" in hint or "more evidence" in hint
