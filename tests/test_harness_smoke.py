"""End-to-end plumbing: real pytest-graded toy family through the full harness
with a network-free provider. Proves reset -> agent -> apply_fix -> grade ->
record -> consolidate -> report runs and emits curves."""
from providers import DryRunProvider, ModelResponse, ToolCall
from harness import Runner, RunConfig, Reporter
from families import get_family
from families.toy_bug import ToyBugEnv, ToyBugFamily


def test_toy_env_execution_grounded_reward():
    """The reward is a FACT about executed code, not a judge."""
    fam = ToyBugFamily()
    v = next(iter(fam.train_variants()))
    env = ToyBugEnv()
    env.reset(v)

    env.step("apply_fix", {"source": v.spec["buggy_src"]})   # still buggy
    assert env.grade().task_success == 0.0

    env.step("apply_fix", {"source": v.spec["correct_src"]}) # correct fix
    assert env.grade().task_success == 1.0


def test_harness_runs_and_reports(tmp_path):
    provider = DryRunProvider()   # does nothing -> episodes fail, but loop runs
    runner = Runner(online=provider, offline=provider)
    cfg = RunConfig(configs=["a0", "a2"], seeds=[1], checkpoint_every=6,
                    max_steps=2, out_dir=str(tmp_path))
    records = runner.run(get_family("toy_bug"), cfg)

    assert len(records) == 12          # 2 configs x 6 train variants x 1 seed
    metrics = Reporter(records).metrics()
    assert "a0" in metrics and "a2" in metrics
    assert len(metrics["a0"]["curve"]) == 6
    assert "learning_slope" in metrics["a2"]


def test_oracle_provider_produces_success():
    """A provider that submits the correct fix should drive success to 1.0 —
    demonstrating the curve responds to agent quality, not just plumbing."""
    fam = ToyBugFamily()

    def oracle(messages):
        # Cheat for the toy only: the correct source is derivable from context
        # embedded in the last user message is NOT available, so use a handler
        # keyed off the current variant via closure below.
        return ModelResponse(text="", tool_calls=[])

    # Build a per-variant oracle by running variants directly.
    from agent import AgentController
    from harness.environment import Variant
    for v in fam.train_variants():
        correct = v.spec["correct_src"]
        prov = DryRunProvider(handler=lambda m, c=correct: ModelResponse(
            tool_calls=[ToolCall(name="apply_fix", arguments={"source": c})]))
        ctrl = AgentController(prov, max_steps=2)
        env = fam.make_env()
        ep = ctrl.run(env, v, injected_context="", seed=1)
        assert env.grade().task_success == 1.0
