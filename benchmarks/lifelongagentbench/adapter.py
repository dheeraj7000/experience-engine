"""LifelongAgentBench adapter (Phase 1 anchor) — STUB.

Goal: wrap LifelongAgentBench's interdependent multi-session tasks (DB / OS /
knowledge-graph environments) behind our TaskFamily + Environment interfaces so
the same Runner/Reporter measures learning curves on it. Its native task
sequence maps onto our Sequencer.

Keep this adapter THIN: new benchmarks (SWE-bench subset, your own QA families)
plug into the same two interfaces. Implement `make_env` to translate its step
API into Observation, and `grade` to read its task success signal.

TODO(phase1):
  - vendor / pip the benchmark, map its task loader -> train/heldout variants
  - translate action space -> tool_schemas()
  - map its reward -> RewardVector (execution-grounded where possible)
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
