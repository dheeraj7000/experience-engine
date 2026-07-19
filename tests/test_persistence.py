from persistence import ExperienceStore, NoPersistence, MemoryOnly
from persistence.a2_engine.confidence import compute_confidence
from schemas import Episode, RewardVector
from schemas.episode import Step


def _ep(goal, ok, obs="AssertionError"):
    return Episode(episode_id=f"e_{goal}_{ok}", task_family="toy", task_variant_id=goal,
                   goal=goal, outcome=RewardVector.from_success(ok),
                   steps=[Step(i=0, action="apply_fix", observation=obs)])


def test_a0_injects_nothing():
    store = ExperienceStore()
    a0 = NoPersistence(store)
    a0.record(_ep("fix add", False))
    assert a0.retrieve({"goal": "fix add", "family": "toy"}) == ""


def test_a1_retrieves_after_recording():
    store = ExperienceStore()
    a1 = MemoryOnly(store)
    a1.record(_ep("fix add function", True))
    out = a1.retrieve({"goal": "fix add function", "family": "toy"})
    assert "add" in out.lower()


def test_confidence_monotonic_in_evidence():
    low = compute_confidence(evidence_count=1, consistency=1.0, contradictions=0)
    high = compute_confidence(evidence_count=10, consistency=1.0, contradictions=0)
    assert 0.0 <= low < high <= 1.0
    penalized = compute_confidence(evidence_count=10, consistency=1.0, contradictions=5)
    assert penalized < high
