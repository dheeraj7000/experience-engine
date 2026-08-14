"""Tests for Phase 2: pattern mining and feature-based clustering."""
from schemas import Episode, RewardVector
from schemas.episode import Step
from persistence.a2_engine.pattern_miner import PatternMiner, PatternCluster
from persistence.a2_engine.taxonomy import FailureType


def _failing(i, action="submit_test", observation="AssertionError wrong output",
             family="bug_reproduction", variant="v"):
    return Episode(
        episode_id=f"ep_{i}", task_family=family, task_variant_id=variant,
        goal="fix bug", initial_state={"failure_type": "wrong_output"},
        outcome=RewardVector.from_success(False),
        steps=[Step(i=0, action=action, observation=observation)],
    )


def test_signature_clustering_groups_same_signature():
    miner = PatternMiner(min_cluster_size=2)
    eps = [_failing(i) for i in range(4)]
    clusters = miner.cluster_by_signature(eps)
    assert len(clusters) >= 1
    # All 4 should be in the same cluster (same family, same failure pattern).
    total = sum(len(v) for v in clusters.values())
    assert total == 4


def test_signature_clustering_excludes_successes():
    miner = PatternMiner(min_cluster_size=2)
    eps = [_failing(0), _failing(1)]
    # Add a success — should be excluded.
    success = Episode(
        episode_id="ep_s", task_family="bug_reproduction", task_variant_id="v",
        goal="fix", outcome=RewardVector.from_success(True), steps=[],
    )
    eps.append(success)
    clusters = miner.cluster_by_signature(eps)
    total = sum(len(v) for v in clusters.values())
    assert total == 2  # only failures


def test_feature_clustering_groups_similar_episodes():
    # Use a higher threshold to prevent merging dissimilar groups.
    miner = PatternMiner(similarity_threshold=0.6, min_cluster_size=2)
    # Group A: same action, same observation pattern, same family.
    group_a = [_failing(i, action="submit_test", observation="assert failed")
               for i in range(3)]
    # Group B: different action, different observation, different family.
    group_b = [_failing(i + 10, action="apply_fix", observation="timeout exceeded",
                        family="toy_bug")
               for i in range(3)]
    all_eps = group_a + group_b
    clusters = miner.cluster_by_features(all_eps)
    # Should get at least 2 distinct clusters (groups are quite dissimilar).
    assert len(clusters) >= 2


def test_feature_clustering_returns_empty_for_no_failures():
    miner = PatternMiner(min_cluster_size=2)
    success = Episode(
        episode_id="ep_s", task_family="test", task_variant_id="v",
        goal="ok", outcome=RewardVector.from_success(True), steps=[],
    )
    clusters = miner.cluster_by_features([success])
    assert clusters == []


def test_cross_cluster_patterns_detected():
    miner = PatternMiner(min_cluster_size=2)
    # Two clusters sharing observation keywords.
    c1 = PatternCluster("c1", [
        _failing(0, observation="schema mismatch column error"),
        _failing(1, observation="schema mismatch type error"),
    ])
    c2 = PatternCluster("c2", [
        _failing(2, observation="schema validation failed column",
                 action="validate"),
        _failing(3, observation="schema check column error",
                 action="validate"),
    ])
    patterns = miner.find_cross_cluster_patterns([c1, c2])
    assert len(patterns) >= 1
    # Should mention shared keywords.
    assert any("schema" in str(p.get("shared_keywords", [])) for p in patterns)


def test_pattern_cluster_summary():
    eps = [_failing(i) for i in range(3)]
    cluster = PatternCluster("test_cluster", eps)
    summary = cluster.summary()
    assert summary["size"] == 3
    assert summary["cluster_id"] == "test_cluster"
    assert "failure_type" in summary
