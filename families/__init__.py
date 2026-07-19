"""Registry of QA task families ("all QA").

Only the toy family is fully executable in Phase 0. The rest are declared
stubs describing the intended task + execution-based reward, to be filled in
as real datasets/generators land. Each must be execution-graded (run the
code), never judge-graded.
"""
from __future__ import annotations

from .toy_bug import ToyBugFamily

_REGISTRY = {
    "toy_bug": ToyBugFamily,
}

# Declared but not yet implemented (see qa_families.py for the spec).
PLANNED = [
    "bug_reproduction", "flaky_test_triage", "test_authoring",
    "regression_bisect", "failure_clustering",
]


def get_family(family_id: str):
    if family_id not in _REGISTRY:
        raise KeyError(
            f"family {family_id!r} not implemented. Available: {list(_REGISTRY)}; "
            f"planned: {PLANNED}"
        )
    return _REGISTRY[family_id]()
