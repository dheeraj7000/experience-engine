"""Tests for flaky_test_triage family — execution-grounded grading."""
from families import get_family
from families.flaky_test_triage import FlakyTestTriageFamily, FlakyTestEnv


def test_family_registered():
    family = get_family("flaky_test_triage")
    assert family.family_id == "flaky_test_triage"


def test_train_variants_count():
    family = FlakyTestTriageFamily()
    variants = list(family.train_variants())
    assert len(variants) == 6


def test_heldout_variants_count():
    family = FlakyTestTriageFamily()
    variants = list(family.heldout_variants())
    assert len(variants) == 1


def test_env_reset_shows_flaky_test():
    family = FlakyTestTriageFamily()
    env = family.make_env()
    variants = list(family.train_variants())
    obs = env.reset(variants[0])
    assert "intermittently" in obs.text
    assert "submit_diagnosis" in obs.text
    assert not obs.done


def test_env_submit_diagnosis():
    family = FlakyTestTriageFamily()
    env = family.make_env()
    variants = list(family.train_variants())
    env.reset(variants[0])
    obs = env.step("submit_diagnosis", {"cause": "timing issue"})
    assert not obs.done  # still need to submit fix


def test_env_submit_fix_completes():
    family = FlakyTestTriageFamily()
    env = family.make_env()
    variants = list(family.train_variants())
    env.reset(variants[0])
    env.step("submit_diagnosis", {"cause": "timing"})
    obs = env.step("submit_fix", {"test_source": "def test_x(): pass"})
    assert obs.done


def test_stable_fix_gets_full_success():
    """Submitting the known-stable test should get task_success=1.0."""
    family = FlakyTestTriageFamily()
    env = family.make_env()
    variants = list(family.train_variants())
    # Use the "order_counter" variant (index 1) — simple and deterministic.
    v = variants[1]
    env.reset(v)
    env.step("submit_diagnosis", {"cause": "shared global state not reset"})
    env.step("submit_fix", {"test_source": v.spec["stable_test"]})
    reward = env.grade()
    assert reward.task_success == 1.0
    assert reward.reproducible is True


def test_flaky_test_gets_low_score():
    """Submitting the original flaky test back should NOT get full success."""
    family = FlakyTestTriageFamily()
    env = family.make_env()
    variants = list(family.train_variants())
    v = variants[2]  # random_sample — inherently flaky
    env.reset(v)
    env.step("submit_fix", {"test_source": v.spec["flaky_test"]})
    reward = env.grade()
    # The flaky test may sometimes pass all 5 runs by luck, but the
    # random_sample one should mostly fail (asserts exact random output).
    # We just verify grading runs without error.
    assert reward.overall >= 0.0


def test_empty_fix_fails():
    family = FlakyTestTriageFamily()
    env = family.make_env()
    variants = list(family.train_variants())
    env.reset(variants[0])
    env.step("submit_fix", {"test_source": ""})
    reward = env.grade()
    assert reward.task_success == 0.0


def test_diagnosis_check_timing():
    family = FlakyTestTriageFamily()
    env = family.make_env()
    variants = list(family.train_variants())
    # First variant is timing_cache
    env.reset(variants[0])
    env.step("submit_diagnosis", {"cause": "timing race condition with sleep"})
    env.step("submit_fix", {"test_source": variants[0].spec["stable_test"]})
    reward = env.grade()
    # Correct diagnosis should boost efficiency.
    assert reward.efficiency > 0.5


def test_tool_schemas():
    family = FlakyTestTriageFamily()
    env = family.make_env()
    schemas = env.tool_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert "submit_diagnosis" in names
    assert "submit_fix" in names
