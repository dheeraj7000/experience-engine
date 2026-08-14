"""Tests for Phase 4: Skill Compiler — inducing reusable workflows from successes."""
from schemas import Episode, RewardVector, SkillObject
from schemas.episode import Step
from schemas.experience import ValidationStatus
from persistence.a2_engine.skill_compiler import SkillCompiler


def _success(i, family="test", actions=None):
    """Create a successful episode with given action sequence."""
    actions = actions or ["analyze", "fix", "submit"]
    steps = [Step(i=idx, action=a, observation=f"{a} done")
             for idx, a in enumerate(actions)]
    return Episode(
        episode_id=f"ep_{i}", task_family=family, task_variant_id=f"v{i}",
        goal="fix the bug", initial_state={"failure_type": "wrong_output"},
        outcome=RewardVector.from_success(True),
        steps=steps,
    )


def _failure(i, family="test"):
    return Episode(
        episode_id=f"ep_f{i}", task_family=family, task_variant_id="v",
        goal="fix", initial_state={},
        outcome=RewardVector.from_success(False),
        steps=[Step(i=0, action="submit", observation="failed")],
    )


def test_compiles_skill_from_repeated_successes():
    compiler = SkillCompiler(min_evidence=3)
    eps = [_success(i) for i in range(4)]
    skills = compiler.compile_skills(eps)
    assert len(skills) == 1
    skill = skills[0]
    assert skill.scope == "test"
    assert skill.evidence_count == 4
    assert len(skill.workflow) == 3  # analyze, fix, submit
    assert skill.validation_status == ValidationStatus.candidate


def test_no_skill_from_failures():
    compiler = SkillCompiler(min_evidence=3)
    eps = [_failure(i) for i in range(5)]
    skills = compiler.compile_skills(eps)
    assert skills == []


def test_no_skill_below_min_evidence():
    compiler = SkillCompiler(min_evidence=5)
    eps = [_success(i) for i in range(3)]
    skills = compiler.compile_skills(eps)
    assert skills == []


def test_separate_skills_for_different_workflows():
    compiler = SkillCompiler(min_evidence=3)
    # Group A: analyze → fix → submit
    group_a = [_success(i, actions=["analyze", "fix", "submit"]) for i in range(3)]
    # Group B: search → compare → report
    group_b = [_success(i + 10, actions=["search", "compare", "report"]) for i in range(3)]
    skills = compiler.compile_skills(group_a + group_b)
    assert len(skills) == 2


def test_skill_preconditions_from_shared_state():
    compiler = SkillCompiler(min_evidence=3)
    eps = [_success(i) for i in range(3)]
    skills = compiler.compile_skills(eps)
    assert skills
    # Should include task_family in preconditions.
    assert any("task_family=test" in p for p in skills[0].preconditions)


def test_skill_validation_promotes_when_replay_helps():
    compiler = SkillCompiler()
    skill = SkillObject(
        skill_id="skill_1", name="test_skill", scope="test",
        workflow=["step1", "step2"], confidence=0.7,
        validation_status=ValidationStatus.candidate,
    )
    validated = compiler.validate_skill(skill, replay_fn=lambda s: (0.4, 0.8))
    assert validated is True
    assert skill.validation_status == ValidationStatus.active


def test_skill_validation_stays_candidate_when_no_improvement():
    compiler = SkillCompiler()
    skill = SkillObject(
        skill_id="skill_1", name="test_skill", scope="test",
        workflow=["step1"], confidence=0.7,
        validation_status=ValidationStatus.candidate,
    )
    validated = compiler.validate_skill(skill, replay_fn=lambda s: (0.8, 0.7))
    assert validated is False
    assert skill.validation_status == ValidationStatus.candidate


def test_reinforce_skill():
    compiler = SkillCompiler()
    skill = SkillObject(
        skill_id="skill_1", name="test", scope="test",
        workflow=["step1"], confidence=0.7, evidence_count=3,
        source_episodes=["ep_1"],
    )
    new_ep = _success(99)
    compiler.reinforce_skill(skill, new_ep)
    assert skill.evidence_count == 4
    assert "ep_99" in skill.source_episodes
    assert skill.confidence > 0.7


def test_does_not_duplicate_existing_skill():
    compiler = SkillCompiler(min_evidence=3)
    eps = [_success(i, actions=["analyze", "fix"]) for i in range(4)]
    existing = SkillObject(
        skill_id="skill_existing", name="existing", scope="test",
        workflow=["analyze", "fix"], confidence=0.8,
        validation_status=ValidationStatus.active,
    )
    skills = compiler.compile_skills(eps, existing_skills=[existing])
    # Should not re-compile the same workflow.
    assert skills == []


def test_skill_as_instruction():
    skill = SkillObject(
        skill_id="s1", name="bug_fix_workflow", scope="test",
        preconditions=["task_family=test"],
        workflow=["analyze error", "write fix", "verify"],
    )
    text = skill.as_instruction()
    assert "bug_fix_workflow" in text
    assert "1. analyze error" in text
    assert "2. write fix" in text
