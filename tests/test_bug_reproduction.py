"""bug_reproduction: the first real QA family (not a toy). Reward is
two-sided execution-grounded — fails on buggy code AND passes on the fixed
reference — so a vacuous or broken test can't earn success."""
from families import get_family
from families.bug_reproduction import BugReproductionEnv, BugReproductionFamily


def test_registered_in_family_registry():
    fam = get_family("bug_reproduction")
    assert isinstance(fam, BugReproductionFamily)
    assert len(list(fam.train_variants())) == 7
    assert len(list(fam.heldout_variants())) == 2


def test_reproducing_test_succeeds():
    fam = BugReproductionFamily()
    v = next(v for v in fam.train_variants() if v.spec["function_name"] == "is_even")
    env = BugReproductionEnv()
    env.reset(v)
    env.step("submit_test", {
        "test_source": "from solution import is_even\n\n"
                        "def test_is_even():\n    assert is_even(4) is True\n",
    })
    rv = env.grade()
    assert rv.task_success == 1.0
    assert rv.reproducible is True


def test_vacuous_test_gets_partial_credit_not_success():
    """Passes on both buggy and fixed code -> never exercised the bug."""
    fam = BugReproductionFamily()
    v = next(v for v in fam.train_variants() if v.spec["function_name"] == "is_even")
    env = BugReproductionEnv()
    env.reset(v)
    env.step("submit_test", {
        "test_source": "from solution import is_even\n\n"
                        "def test_is_even():\n    assert isinstance(is_even(4), bool)\n",
    })
    rv = env.grade()
    assert rv.task_success == 0.0
    assert 0.0 < rv.partial_credit < 1.0


def test_broken_test_gets_no_credit():
    """Fails on the fixed reference too -> the test itself is wrong, not the bug."""
    fam = BugReproductionFamily()
    v = next(v for v in fam.train_variants() if v.spec["function_name"] == "is_even")
    env = BugReproductionEnv()
    env.reset(v)
    env.step("submit_test", {"test_source": "not valid python !!"})
    rv = env.grade()
    assert rv.task_success == 0.0
    assert rv.partial_credit == 0.0


def test_no_submission_fails():
    fam = BugReproductionFamily()
    v = next(iter(fam.train_variants()))
    env = BugReproductionEnv()
    env.reset(v)
    assert env.grade().task_success == 0.0


def test_oracle_agent_reproduces_every_bug():
    """An agent that submits a real reproducing test (derived from the known
    fixed/buggy diff) should drive success to 1.0 for every train variant —
    the curve responds to test-writing quality, not just plumbing."""
    from providers import DryRunProvider, ModelResponse, ToolCall
    from agent import AgentController

    fam = BugReproductionFamily()
    for v in fam.train_variants():
        fn = v.spec["function_name"]
        # A minimal oracle test per function, keyed off the known bug.
        oracle_tests = {
            "is_even": f"from solution import {fn}\n\ndef test_x():\n    assert {fn}(4) is True\n",
            "first_n": f"from solution import {fn}\n\ndef test_x():\n    assert {fn}([1,2,3,4], 2) == [1, 2]\n",
            "safe_div": f"from solution import {fn}\n\ndef test_x():\n    assert {fn}(10, 0) is None\n",
            "title_case": f"from solution import {fn}\n\ndef test_x():\n    assert {fn}('hello world') == 'Hello World'\n",
            "dedupe": f"from solution import {fn}\n\ndef test_x():\n    assert {fn}([3,1,3,2,1]) == [3, 1, 2]\n",
            "clamp": f"from solution import {fn}\n\ndef test_x():\n    assert {fn}(-5, 0, 10) == 0\n",
            "last_n": f"from solution import {fn}\n\ndef test_x():\n    assert {fn}([1,2,3,4,5], 2) == [4, 5]\n",
        }
        test_src = oracle_tests[fn]
        prov = DryRunProvider(handler=lambda m, t=test_src: ModelResponse(
            tool_calls=[ToolCall(name="submit_test", arguments={"test_source": t})]))
        ctrl = AgentController(prov, max_steps=2)
        env = fam.make_env()
        ctrl.run(env, v, injected_context="", seed=1)
        assert env.grade().task_success == 1.0, f"oracle failed to reproduce {fn}"
