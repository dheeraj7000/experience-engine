"""Tests for Phase 2: upgraded causal diagnoser with taxonomy + structured chains."""
from schemas import Episode, RewardVector
from schemas.episode import Step
from persistence.a2_engine.diagnoser import CausalDiagnoser, CausalChain
from persistence.a2_engine.taxonomy import FailureType


def _failing(i, action="submit_test", observation="AssertionError wrong output",
             family="test"):
    return Episode(
        episode_id=f"ep_{i}", task_family=family, task_variant_id="v",
        goal="fix bug", initial_state={"failure_type": "wrong_output"},
        outcome=RewardVector.from_success(False),
        steps=[Step(i=0, action=action, observation=observation)],
    )


def test_diagnose_returns_structured_dict():
    diag = CausalDiagnoser()
    eps = [_failing(i) for i in range(3)]
    result = diag.diagnose(eps)
    assert "root_cause" in result
    assert "failure_type" in result
    assert "critical_step" in result
    assert "counterfactual_repair" in result
    assert "confidence" in result
    assert result["confidence"] > 0


def test_diagnose_empty_cluster():
    diag = CausalDiagnoser()
    result = diag.diagnose([])
    assert result["root_cause"] == "no failures in cluster"
    assert result["confidence"] == 0.0


def test_diagnose_no_failures_in_cluster():
    diag = CausalDiagnoser()
    success = Episode(
        episode_id="ep_s", task_family="test", task_variant_id="v",
        goal="ok", outcome=RewardVector.from_success(True), steps=[],
    )
    result = diag.diagnose([success])
    assert result["confidence"] == 0.0


def test_diagnose_single_episode():
    diag = CausalDiagnoser()
    # No explicit failure_type in initial_state so classification uses observations.
    ep = Episode(
        episode_id="ep_0", task_family="test", task_variant_id="v",
        goal="fix bug", initial_state={},
        outcome=RewardVector.from_success(False),
        steps=[Step(i=0, action="submit_test", observation="Traceback: IndexError in line 5")],
    )
    chain = diag.diagnose_single(ep)
    assert isinstance(chain, CausalChain)
    assert chain.failure_type == FailureType.runtime_error
    assert chain.critical_action == "submit_test"
    assert chain.confidence > 0


def test_confidence_increases_with_cluster_size():
    diag = CausalDiagnoser()
    small_cluster = [_failing(i) for i in range(3)]
    large_cluster = [_failing(i) for i in range(6)]
    result_small = diag.diagnose(small_cluster)
    result_large = diag.diagnose(large_cluster)
    # Larger cluster should have >= confidence (may both hit cap at 0.95).
    assert result_large["confidence"] >= result_small["confidence"]
    # And both should be non-trivial.
    assert result_small["confidence"] >= 0.5


def test_diagnose_identifies_correct_failure_type():
    diag = CausalDiagnoser()
    # Timeout cluster.
    timeout_eps = [
        Episode(
            episode_id=f"ep_t{i}", task_family="test", task_variant_id="v",
            goal="run", initial_state={},
            outcome=RewardVector.from_success(False),
            steps=[Step(i=0, action="execute", observation="execution timed out")],
        )
        for i in range(3)
    ]
    result = diag.diagnose(timeout_eps)
    assert result["failure_type"] == "timeout"


def test_diagnose_contributing_factors():
    diag = CausalDiagnoser()
    # Episode with repeated same action (stuck in loop).
    ep = Episode(
        episode_id="ep_loop", task_family="test", task_variant_id="v",
        goal="solve", initial_state={},
        outcome=RewardVector.from_success(False),
        steps=[
            Step(i=0, action="retry", observation="failed"),
            Step(i=1, action="retry", observation="failed"),
            Step(i=2, action="retry", observation="failed"),
        ],
    )
    chain = diag.diagnose_single(ep)
    assert any("repeated" in f for f in chain.contributing_factors)


def test_causal_chain_to_dict():
    chain = CausalChain(
        root_cause="test root cause",
        failure_type=FailureType.wrong_output,
        critical_step_index=0,
        critical_action="submit",
        contributing_factors=["factor1"],
        counterfactual_repair="fix it",
        confidence=0.7,
    )
    d = chain.to_dict()
    assert d["root_cause"] == "test root cause"
    assert d["failure_type"] == "wrong_output"
    assert d["confidence"] == 0.7
