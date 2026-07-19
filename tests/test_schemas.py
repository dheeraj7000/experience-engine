from schemas import RewardVector, Episode, ExperienceObject, PolicyObject
from schemas.episode import Step


def test_reward_overall_bounds_and_success():
    rv = RewardVector.from_success(True, efficiency=1.0, cost=1.0, latency=1.0, safety=1.0)
    assert rv.task_success == 1.0 and rv.reproducible is True
    assert 0.0 <= rv.overall <= 1.0
    fail = RewardVector.from_success(False)
    assert fail.overall < rv.overall


def test_episode_failure_signature_reflects_success():
    ok = Episode(episode_id="e1", task_family="toy", task_variant_id="v1",
                 outcome=RewardVector.from_success(True))
    bad = Episode(episode_id="e2", task_family="toy", task_variant_id="v1",
                  outcome=RewardVector.from_success(False),
                  steps=[Step(i=0, action="apply_fix", observation="AssertionError")])
    assert "success=True" in ok.failure_signature()
    assert "success=False" in bad.failure_signature()


def test_experience_context_matching():
    exp = ExperienceObject(experience_id="x1", task_family="toy",
                           context_conditions={"family": "toy", "failure_type": "wrong_output"})
    assert exp.matches({"family": "toy", "failure_type": "wrong_output", "extra": 1})
    assert not exp.matches({"family": "toy", "failure_type": "timeout"})


def test_policy_directive():
    p = PolicyObject(policy_id="p1", scope="toy", trigger_conditions=["x=1"], behavior="do y")
    assert "do y" in p.as_directive() and "toy" in p.as_directive()
