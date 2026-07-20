"""The non-negotiable gate: a wrong ('poisoned') lesson must be REJECTED,
never promoted. If the validator can't catch this, the system learns noise.
"""
from persistence import ExperienceStore
from persistence.a2_engine.engine import ExperienceEngine
from persistence.a2_engine.validator import ExperienceValidator
from schemas import Episode, ExperienceObject, RewardVector
from schemas.episode import Step
from schemas.experience import ValidationStatus


def test_validator_rejects_harmful_experience():
    exp = ExperienceObject(experience_id="x", task_family="toy", lesson="always skip checks")
    v = ExperienceValidator()
    # Replay shows the lesson HURTS held-out performance (0.8 -> 0.5).
    promoted = v.validate(exp, replay_fn=lambda e: (0.8, 0.5))
    assert promoted is False
    assert exp.validation_status == ValidationStatus.rejected


def test_validator_rejects_vacuous_tie_at_zero_baseline():
    """If baseline held-out success is already 0 (common on a hard set),
    'improved >= 0' is trivially true for ANY replay outcome -- it can't
    distinguish a real lesson from a no-op. A tie (0 -> 0) must be rejected,
    not accepted just because the delta isn't negative."""
    exp = ExperienceObject(experience_id="x", task_family="toy", lesson="no-op")
    v = ExperienceValidator()
    promoted = v.validate(exp, replay_fn=lambda e: (0.0, 0.0))
    assert promoted is False
    assert exp.validation_status == ValidationStatus.rejected


def test_validator_no_replay_stays_provisional():
    exp = ExperienceObject(experience_id="x", task_family="toy", lesson="maybe")
    assert ExperienceValidator().validate(exp, replay_fn=None) is False
    assert exp.validation_status == ValidationStatus.provisional


def test_poisoned_experience_not_promoted_in_full_loop():
    store = ExperienceStore()
    engine = ExperienceEngine(store)
    for i in range(3):
        store.add_episode(Episode(
            episode_id=f"ep_{i}", task_family="toy", task_variant_id="v", goal="g",
            initial_state={"failure_type": "wrong_output"},
            outcome=RewardVector.from_success(False),
            steps=[Step(i=0, action="apply_fix", observation="error")]))
    # A harmful lesson: replay always degrades. No policy should be created.
    engine.consolidate(replay_fn=lambda exp: (0.9, 0.3))
    assert all(e.validation_status == ValidationStatus.rejected for e in store.experiences)
    assert store.policies == []
