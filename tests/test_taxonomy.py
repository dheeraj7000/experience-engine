"""Tests for Phase 2: failure taxonomy classification."""
from schemas import Episode, RewardVector
from schemas.episode import Step
from persistence.a2_engine.taxonomy import (
    FailureType, classify_failure, failure_features,
)


def _episode(steps, outcome_success=0.0, initial_state=None):
    return Episode(
        episode_id="ep_test", task_family="test", task_variant_id="v",
        goal="test goal",
        initial_state=initial_state or {},
        outcome=RewardVector(task_success=outcome_success).compute_overall(),
        steps=steps,
    )


def test_classify_no_steps_is_no_attempt():
    ep = _episode(steps=[])
    assert classify_failure(ep) == FailureType.no_attempt


def test_classify_timeout_from_observation():
    ep = _episode(steps=[
        Step(i=0, action="run", observation="execution timed out after 30s"),
    ])
    assert classify_failure(ep) == FailureType.timeout


def test_classify_runtime_error():
    ep = _episode(steps=[
        Step(i=0, action="apply_fix", observation="Traceback: ZeroDivisionError"),
    ])
    assert classify_failure(ep) == FailureType.runtime_error


def test_classify_wrong_output():
    ep = _episode(
        steps=[Step(i=0, action="submit", observation="submitted")],
        outcome_success=0.0,
    )
    assert classify_failure(ep) == FailureType.wrong_output


def test_classify_incomplete_with_partial_credit():
    ep = _episode(steps=[Step(i=0, action="work", observation="ok")])
    ep.outcome = RewardVector(task_success=0.0, partial_credit=0.4).compute_overall()
    assert classify_failure(ep) == FailureType.incomplete


def test_explicit_failure_type_in_initial_state():
    ep = _episode(
        steps=[Step(i=0, action="x", observation="y")],
        initial_state={"failure_type": "test_invalid"},
    )
    assert classify_failure(ep) == FailureType.test_invalid


def test_failure_features_structure():
    ep = _episode(
        steps=[Step(i=0, action="submit_test", observation="assertion failed")],
        initial_state={"failure_type": "wrong_output"},
    )
    feats = failure_features(ep)
    assert feats["failure_type"] == "wrong_output"
    assert feats["last_action"] == "submit_test"
    assert feats["task_family"] == "test"
    assert isinstance(feats["actions_taken"], list)
