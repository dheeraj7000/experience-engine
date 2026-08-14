"""Failure Taxonomy (Phase 2, proposal 8.3 + 8.5).

A structured classification of failure types that enables targeted diagnosis
and pattern mining. Instead of treating all failures equally, the taxonomy
routes episodes to the right diagnostic strategy and lets the pattern miner
cluster by failure *kind*, not just surface keywords.

Taxonomy is open (new types can be added), but the core set covers the QA
domain well. Each type implies a different likely root cause and a different
counterfactual repair strategy.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from schemas import Episode


class FailureType(str, Enum):
    """Coarse failure categories. A single episode maps to exactly one."""
    wrong_output = "wrong_output"           # code ran, wrong result
    runtime_error = "runtime_error"         # exception / crash
    timeout = "timeout"                     # hung / exceeded budget
    missing_action = "missing_action"       # agent never called the right tool
    wrong_tool_args = "wrong_tool_args"     # right tool, bad arguments
    incomplete = "incomplete"               # partial progress, stopped early
    test_invalid = "test_invalid"           # submitted test is itself broken
    no_attempt = "no_attempt"               # agent produced nothing useful
    unknown = "unknown"


# Signals that classify failure type from episode data.
_ERROR_KEYWORDS = ("error", "exception", "traceback", "raise", "crash")
_TIMEOUT_KEYWORDS = ("timeout", "timed out", "exceeded")


def classify_failure(episode: Episode) -> FailureType:
    """Heuristic classification from episode structure. Families can override
    with domain-specific logic via initial_state['failure_type']."""
    # Respect explicit annotation from the environment.
    explicit = episode.initial_state.get("failure_type")
    if explicit and explicit in FailureType.__members__:
        return FailureType(explicit)

    if not episode.steps:
        return FailureType.no_attempt

    # Check observations for signals.
    all_obs = " ".join(s.observation.lower() for s in episode.steps)

    if any(kw in all_obs for kw in _TIMEOUT_KEYWORDS):
        return FailureType.timeout

    if any(kw in all_obs for kw in _ERROR_KEYWORDS):
        return FailureType.runtime_error

    # Agent never called a tool that produced a done signal.
    actions = [s.action for s in episode.steps]
    if not actions or all(a == "" for a in actions):
        return FailureType.no_attempt

    # If the episode has steps but outcome is failure, it's likely wrong output
    # (ran to completion but answer was incorrect).
    if episode.outcome and episode.outcome.task_success == 0.0:
        if episode.outcome.partial_credit > 0:
            return FailureType.incomplete
        return FailureType.wrong_output

    return FailureType.unknown


def failure_features(episode: Episode) -> dict[str, Any]:
    """Extract structured features for pattern mining and clustering."""
    ftype = classify_failure(episode)
    steps = episode.steps
    last_obs = steps[-1].observation if steps else ""
    last_action = steps[-1].action if steps else ""
    n_steps = len(steps)
    has_tool_call = any(s.tool for s in steps)

    return {
        "failure_type": ftype.value,
        "n_steps": n_steps,
        "last_action": last_action,
        "last_observation_prefix": last_obs[:200],
        "has_tool_call": has_tool_call,
        "task_family": episode.task_family,
        "variant_id": episode.task_variant_id,
        "actions_taken": [s.action for s in steps],
    }
