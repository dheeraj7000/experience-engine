"""Real QA family #1 (see qa_families.py spec): bug reproduction.

Task: given a bug report + a buggy function, WRITE A TEST that reproduces the
bug — a test that fails against the buggy implementation and would pass once
the bug is fixed. This is the QA skill that actually matters before you can
fix anything: proving you understand the failure.

Reward is execution-grounded and two-sided (harder to game than toy_bug's
single fix-and-run):
    fails on buggy code   AND   passes on the fixed reference   =>  success
A test that always passes (never exercises the bug) or always fails (broken
test) gets partial credit at best, never success.
"""
from __future__ import annotations

from typing import Any, Iterable

from agent.tools import run_pytest
from harness.environment import Environment, Observation, Variant
from schemas import RewardVector

# (name, buggy_src, fixed_src, bug_report)
_BUGS = [
    (
        "is_even",
        "def is_even(n):\n    return n % 2 == 1\n",
        "def is_even(n):\n    return n % 2 == 0\n",
        "is_even(4) returns False, but 4 is even so it should return True.",
    ),
    (
        "first_n",
        "def first_n(items, n):\n    return items[:n - 1]\n",
        "def first_n(items, n):\n    return items[:n]\n",
        "first_n([1, 2, 3, 4], 2) returns [1] instead of [1, 2] — one short.",
    ),
    (
        "safe_div",
        "def safe_div(a, b):\n    return a / b\n",
        "def safe_div(a, b):\n    return None if b == 0 else a / b\n",
        "safe_div(10, 0) raises ZeroDivisionError; it should return None for "
        "division by zero instead of crashing.",
    ),
    (
        "title_case",
        "def title_case(s):\n    return s.capitalize()\n",
        "def title_case(s):\n    return s.title()\n",
        "title_case('hello world') returns 'Hello world' — only the first "
        "word is capitalized, not every word.",
    ),
    (
        "dedupe",
        "def dedupe(items):\n    return list(set(items))\n",
        "def dedupe(items):\n"
        "    seen = set()\n    out = []\n"
        "    for x in items:\n"
        "        if x not in seen:\n            seen.add(x)\n            out.append(x)\n"
        "    return out\n",
        "dedupe([3, 1, 3, 2, 1]) returns [1, 2, 3] — order is not preserved. "
        "It should return [3, 1, 2] (first-seen order).",
    ),
    (
        "clamp",
        "def clamp(x, lo, hi):\n    if x < lo:\n        return hi\n"
        "    if x > hi:\n        return lo\n    return x\n",
        "def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n"
        "    if x > hi:\n        return hi\n    return x\n",
        "clamp(-5, 0, 10) returns 10 instead of 0 — the low/high bounds are "
        "swapped when clamping.",
    ),
    (
        "last_n",
        "def last_n(items, n):\n    return items[n:]\n",
        "def last_n(items, n):\n    return items[-n:] if n else []\n",
        "last_n([1, 2, 3, 4, 5], 2) returns [3, 4, 5] instead of [4, 5] — it "
        "slices from the front, not the back.",
    ),
]
_HELDOUT = [
    (
        "count_vowels",
        "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')\n",
        "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
        "count_vowels('Apple') returns 1 instead of 2 — uppercase vowels "
        "aren't counted.",
    ),
    (
        "flatten_one_level",
        "def flatten_one_level(lists):\n    return lists[0] + lists[1] if len(lists) > 1 else lists[0]\n",
        "def flatten_one_level(lists):\n"
        "    out = []\n    for l in lists:\n        out.extend(l)\n    return out\n",
        "flatten_one_level([[1], [2], [3]]) returns [1, 2] instead of "
        "[1, 2, 3] — only the first two sublists are used.",
    ),
]


def _make_variant(name, buggy_src, fixed_src, bug_report, heldout=False) -> Variant:
    return Variant(
        variant_id=f"bugrepro_{name}",
        family="bug_reproduction",
        goal=f"Write a test that reproduces the reported bug in `{name}`.",
        spec={
            "buggy_src": buggy_src,
            "fixed_src": fixed_src,
            "bug_report": bug_report,
            "function_name": name,
            "failure_type": "unreproduced_bug",
        },
        heldout=heldout,
    )


class BugReproductionEnv:
    def __init__(self) -> None:
        self._buggy_src = ""
        self._fixed_src = ""
        self._submitted_test = ""
        self._ctx: dict[str, Any] = {}

    def reset(self, variant: Variant) -> Observation:
        self._buggy_src = variant.spec["buggy_src"]
        self._fixed_src = variant.spec["fixed_src"]
        self._submitted_test = ""
        self._ctx = {"family": variant.family, "failure_type": variant.spec.get("failure_type")}
        fn = variant.spec["function_name"]
        return Observation(
            text=(
                f"Bug report: {variant.spec['bug_report']}\n\n"
                f"Current implementation (importable as `from solution import {fn}`):\n"
                f"{self._buggy_src}\n"
                "Write a pytest test module that FAILS against this buggy code and "
                "would PASS once the bug is fixed. "
                "Call submit_test(test_source=<full pytest test module source>)."
            ),
            done=False,
        )

    def step(self, action_name: str, args: dict[str, Any]) -> Observation:
        if action_name == "submit_test" and "test_source" in args:
            self._submitted_test = str(args["test_source"])
            return Observation(text="test submitted", done=True)
        return Observation(text=f"unknown action {action_name!r}", done=False)

    def grade(self) -> RewardVector:
        if not self._submitted_test.strip():
            return RewardVector.from_success(False)
        buggy_res = run_pytest(self._buggy_src, self._submitted_test)
        fixed_res = run_pytest(self._fixed_src, self._submitted_test)
        reproduces = (not buggy_res.passed) and fixed_res.passed
        if reproduces:
            return RewardVector.from_success(True, efficiency=1.0, cost=1.0, latency=1.0)
        # Well-formed test that just never exercises the bug (passes either
        # way) gets partial credit; anything that fails on the FIXED
        # reference too is a broken test and gets none.
        partial = 0.3 if (buggy_res.passed and fixed_res.passed) else 0.0
        rv = RewardVector(task_success=0.0, partial_credit=partial, reproducible=False)
        return rv.compute_overall()

    def tool_schemas(self) -> list[dict]:
        return [{
            "type": "function",
            "function": {
                "name": "submit_test",
                "description": (
                    "Submit a pytest test module that reproduces the bug: it "
                    "must fail against the buggy code and pass once the bug "
                    "is fixed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"test_source": {"type": "string"}},
                    "required": ["test_source"],
                },
            },
        }]

    def context(self) -> dict[str, Any]:
        return dict(self._ctx)


class BugReproductionFamily:
    family_id = "bug_reproduction"

    def train_variants(self) -> Iterable[Variant]:
        return [_make_variant(*b) for b in _BUGS]

    def heldout_variants(self) -> Iterable[Variant]:
        return [_make_variant(*b, heldout=True) for b in _HELDOUT]

    def make_env(self) -> Environment:
        return BugReproductionEnv()
