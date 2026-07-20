"""LifelongAgentBench adapter (Phase 1 anchor) — STUB.

Upstream: github.com/caixd-220529/LifelongAgentBench (paper: arXiv:2505.11942;
dataset: huggingface.co/datasets/csyq/LifelongAgentBench). Confirmed by reading
its source (2026-07-20), not guessed:

  - Four task types under src/tasks/instance/: db_bench, os_interaction,
    knowledge_graph, web_shopping. db_bench and os_interaction run inside
    Docker containers (`docker pull mysql` / `docker pull ubuntu` + a custom
    image build) — Docker is available on this box, so that's not a blocker.
  - Its core abstraction is `TaskInterface` (src/tasks/task.py): stateful,
    session-based — get_sample_index_list() -> reset(session) ->
    interact(session) [the step loop] -> complete(session) ->
    calculate_metric(session_partials) -> release(). This is richer than our
    Environment protocol (sessions carry multi-turn Role-tagged chat history
    and its own SampleStatus/AgentAction enums), so the adapter's `make_env`
    needs a real translation layer, not a thin wrapper: drive one of their
    Task instances through our reset/step/grade by holding a live Session
    internally and mapping our single tool call -> their AgentAction.EXECUTE.
  - Reward: calculate_metric returns a MetricDict from their own graders
    (execution-grounded for db_bench/os_interaction — SQL/shell exit state —
    judge-graded for some knowledge_graph/web_shopping items). Only map
    execution-graded task types into RewardVector.task_success for now, to
    keep the "reward is a fact about execution" invariant this repo is built
    around; flag the rest as reproducible=False until verified otherwise.

Keep this adapter THIN where possible: new benchmarks (SWE-bench subset, your
own QA families) plug into the same two interfaces. This one genuinely isn't
thin because their env is heavier than ours — that's the real scoping finding
here, not a placeholder note.

TODO(phase1):
  - vendor/pip the benchmark (Python 3.11 per their pyproject; this repo is on
    3.13 — check compat before adding as a hard dependency)
  - pick ONE task type to start (os_interaction is the closest execution-
    graded analogue to our QA framing) and map its sample loader -> our
    train/heldout Variants
  - implement make_env: hold a Session + TaskInterface instance, translate
    interact()'s turn into our step(), translate calculate_metric -> RewardVector
"""
from __future__ import annotations

from typing import Iterable

from harness.environment import Environment, Variant


class LifelongAgentBenchFamily:
    family_id = "lifelongagentbench"

    def train_variants(self) -> Iterable[Variant]:
        raise NotImplementedError("LifelongAgentBench adapter not yet wired — see TODO(phase1).")

    def heldout_variants(self) -> Iterable[Variant]:
        raise NotImplementedError

    def make_env(self) -> Environment:
        raise NotImplementedError
