"""Specs for the real QA families ("all QA"). These are declarations of intent
— what each task is and how its reward is EXECUTED (never judge-graded).
Implement each behind the same TaskFamily/Environment interface as ToyBugFamily.
"""
from __future__ import annotations

QA_FAMILY_SPECS = {
    "bug_reproduction": {
        "task": "Given a bug report + repo, produce a failing test that reproduces it.",
        "reward_ground_truth": "A new test now fails on buggy code and passes on the fixed ref.",
        "variants_source": "curated issues w/ known fix commits (e.g. from real repos)",
    },
    "flaky_test_triage": {
        "task": "Diagnose why a test fails intermittently and propose a fix.",
        "reward_ground_truth": "Repro-rate over N runs + correctness of identified root cause.",
        "variants_source": "injected concurrency/timing/order-dependence faults",
    },
    "test_authoring": {
        "task": "Write tests for a code change.",
        "reward_ground_truth": "Coverage delta; tests pass on good code, fail on injected mutant.",
        "variants_source": "mutation testing over real functions",
    },
    "regression_bisect": {
        "task": "Find the commit that introduced a regression.",
        "reward_ground_truth": "Exact commit match vs known-bad commit.",
        "variants_source": "synthetic git histories with a planted breaking commit",
    },
    "failure_clustering": {
        "task": "Group a batch of CI failures by underlying cause.",
        "reward_ground_truth": "Cluster purity / ARI vs labeled cause.",
        "variants_source": "labeled CI failure logs",
    },
}

# Recommended first two (execution-graded, cheap to synthesize):
FIRST_FAMILIES = ["bug_reproduction", "flaky_test_triage"]
