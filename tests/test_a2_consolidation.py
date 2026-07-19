"""A2 loop logic, fast (synthetic episodes, no agent/subprocess)."""
from persistence import ExperienceStore
from persistence.a2_engine.engine import ExperienceEngine
from schemas import Episode, RewardVector
from schemas.episode import Step
from schemas.experience import ValidationStatus


def _failing(i):
    return Episode(
        episode_id=f"ep_{i}", task_family="toy", task_variant_id="v",
        goal="fix add", initial_state={"failure_type": "wrong_output"},
        outcome=RewardVector.from_success(False),
        steps=[Step(i=0, action="apply_fix", observation="AssertionError wrong output")],
    )


def test_consolidation_induces_and_promotes_when_validated():
    store = ExperienceStore()
    engine = ExperienceEngine(store)
    for i in range(3):                       # >= MIN_CLUSTER identical failures
        engine.record(_failing(i))

    # replay says the lesson helps (0.4 -> 0.9): should promote to active + policy
    engine.consolidate(replay_fn=lambda exp: (0.4, 0.9))

    assert len(store.experiences) == 1
    exp = store.experiences[0]
    assert exp.validation_status == ValidationStatus.active
    assert len(store.policies) == 1
    assert store.active_policies("toy")


def test_consolidation_does_not_double_process():
    store = ExperienceStore()
    engine = ExperienceEngine(store)
    for i in range(3):
        engine.record(_failing(i))
    engine.consolidate(replay_fn=lambda exp: (0.4, 0.9))
    engine.consolidate(replay_fn=lambda exp: (0.4, 0.9))  # nothing new
    assert len(store.experiences) == 1
