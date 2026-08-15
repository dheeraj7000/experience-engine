"""Tests for Phase 7: Exploration Manager — diversity, profiling, suggestions."""
from schemas import Episode, ExperienceObject, RewardVector, SkillObject
from schemas.episode import Step
from schemas.experience import ValidationStatus
from persistence.a2_engine.exploration import (
    ExplorationManager, FamilyProfile, ExplorationSuggestion,
)


def _episode(eid, family="test", success=True, actions=None):
    actions = actions or ["submit"]
    steps = [Step(i=idx, action=a, observation=f"{a} done")
             for idx, a in enumerate(actions)]
    return Episode(
        episode_id=eid, task_family=family, task_variant_id=f"v_{eid}",
        goal="fix bug", initial_state={},
        outcome=RewardVector.from_success(success),
        steps=steps,
    )


def _experience(eid, family="test"):
    return ExperienceObject(
        experience_id=eid, task_family=family,
        lesson="some lesson", confidence=0.7,
        evidence_count=3, validation_status=ValidationStatus.active,
    )


def test_profile_basic():
    mgr = ExplorationManager()
    eps = [_episode(f"ep_{i}", success=(i % 2 == 0)) for i in range(6)]
    exps = [_experience("exp_1")]
    profiles = mgr.profile_families(eps, exps)
    assert "test" in profiles
    prof = profiles["test"]
    assert prof.total_episodes == 6
    assert prof.successes == 3
    assert prof.failures == 3
    assert prof.success_rate == 0.5
    assert prof.active_experiences == 1


def test_profile_multiple_families():
    mgr = ExplorationManager()
    eps = [_episode("a1", family="sql"), _episode("a2", family="sql"),
           _episode("b1", family="csv")]
    profiles = mgr.profile_families(eps, [])
    assert len(profiles) == 2
    assert "sql" in profiles
    assert "csv" in profiles


def test_exploration_score_high_for_all_failures():
    mgr = ExplorationManager()
    eps = [_episode(f"ep_{i}", success=False) for i in range(6)]
    profiles = mgr.profile_families(eps, [])
    # All failures → high exploration score.
    assert profiles["test"].exploration_score > 0.5


def test_exploration_score_low_for_all_successes():
    mgr = ExplorationManager()
    eps = [_episode(f"ep_{i}", success=True) for i in range(6)]
    profiles = mgr.profile_families(eps, [])
    assert profiles["test"].exploration_score < 0.5


def test_suggestions_for_weak_family():
    mgr = ExplorationManager(min_episodes_for_analysis=3)
    eps = [_episode(f"ep_{i}", success=False) for i in range(6)]
    profiles = mgr.profile_families(eps, [])
    suggestions = mgr.suggest_exploration(profiles)
    assert len(suggestions) >= 1
    assert suggestions[0].family_id == "test"
    assert suggestions[0].priority > 0.5


def test_suggestions_for_low_data():
    mgr = ExplorationManager(min_episodes_for_analysis=10)
    eps = [_episode("ep_1")]
    profiles = mgr.profile_families(eps, [])
    suggestions = mgr.suggest_exploration(profiles)
    # Should suggest practice due to insufficient data.
    assert len(suggestions) == 1
    assert suggestions[0].suggested_action == "practice"


def test_diversity_score_single_strategy():
    mgr = ExplorationManager()
    # All episodes use the same strategy.
    eps = [_episode(f"ep_{i}", actions=["analyze", "submit"]) for i in range(5)]
    score = mgr.diversity_score(eps)
    assert score == 0.0  # zero diversity


def test_diversity_score_unique_strategies():
    mgr = ExplorationManager()
    eps = [
        _episode("ep_0", actions=["analyze", "submit"]),
        _episode("ep_1", actions=["search", "fix"]),
        _episode("ep_2", actions=["debug", "test", "submit"]),
    ]
    score = mgr.diversity_score(eps)
    assert score == 1.0  # maximum diversity (all unique)


def test_diversity_score_mixed():
    mgr = ExplorationManager()
    eps = [
        _episode("ep_0", actions=["submit"]),
        _episode("ep_1", actions=["submit"]),
        _episode("ep_2", actions=["analyze", "submit"]),
        _episode("ep_3", actions=["debug", "submit"]),
    ]
    score = mgr.diversity_score(eps)
    assert 0.0 < score < 1.0


def test_exploitation_ratio():
    mgr = ExplorationManager()
    eps = [_episode("ep_1", family="sql"), _episode("ep_2", family="csv"),
           _episode("ep_3", family="sql")]
    exps = [_experience("exp_1", family="sql")]  # only sql is covered
    ratio = mgr.exploitation_ratio(eps, exps)
    # 2 of 3 episodes are in a family with active experiences.
    assert abs(ratio - 2 / 3) < 0.01


def test_suggestion_for_single_strategy():
    mgr = ExplorationManager(min_episodes_for_analysis=3)
    # All use the same single action.
    eps = [_episode(f"ep_{i}", success=True, actions=["submit"]) for i in range(5)]
    profiles = mgr.profile_families(eps, [])
    suggestions = mgr.suggest_exploration(profiles)
    # Should suggest trying alternatives due to single strategy.
    assert any(s.suggested_action == "try_alternative" for s in suggestions)
