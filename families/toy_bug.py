"""A fully-executable toy QA family: fix an arithmetic bug so its test passes.

Purpose: exercise the WHOLE loop end-to-end (reset -> agent -> apply_fix ->
grade via real pytest -> record -> consolidate -> report) with zero external
dependencies. The reward is execution-grounded: pytest is actually run.

This is a smoke/plumbing family, not a research benchmark. Real families live
alongside it under the same interface.
"""
from __future__ import annotations

from typing import Any, Iterable

from agent.tools import run_pytest
from harness.environment import Environment, Observation, Variant
from schemas import RewardVector

# (name, buggy_body, correct_body, test_body)
_BUGS = [
    ("add", "return a - b", "return a + b", "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"),
    ("mul", "return a + b", "return a * b", "from solution import mul\n\ndef test_mul():\n    assert mul(3, 4) == 12\n"),
    ("sub", "return a + b", "return a - b", "from solution import sub\n\ndef test_sub():\n    assert sub(9, 4) == 5\n"),
    ("mx",  "return a if a < b else b", "return a if a > b else b", "from solution import mx\n\ndef test_mx():\n    assert mx(2, 7) == 7\n"),
    ("dbl", "return x", "return x * 2", "from solution import dbl\n\ndef test_dbl():\n    assert dbl(5) == 10\n"),
    ("neg", "return x", "return -x", "from solution import neg\n\ndef test_neg():\n    assert neg(4) == -4\n"),
]
_HELDOUT = [
    ("inc", "return x", "return x + 1", "from solution import inc\n\ndef test_inc():\n    assert inc(7) == 8\n"),
    ("sq",  "return x", "return x * x", "from solution import sq\n\ndef test_sq():\n    assert sq(6) == 36\n"),
]


def _module(name: str, body: str, arity: int) -> str:
    args = "a, b" if arity == 2 else "x"
    return f"def {name}({args}):\n    {body}\n"


def _make_variant(name, buggy, correct, test, heldout=False) -> Variant:
    arity = 2 if name in {"add", "mul", "sub", "mx"} else 1
    return Variant(
        variant_id=f"toy_{name}",
        family="toy_bug",
        goal=f"Fix the function `{name}` so its test passes.",
        spec={
            "buggy_src": _module(name, buggy, arity),
            "correct_src": _module(name, correct, arity),
            "test_src": test,
            "failure_type": "wrong_output",
        },
        heldout=heldout,
    )


class ToyBugEnv:
    def __init__(self) -> None:
        self._current_src = ""
        self._test_src = ""
        self._ctx: dict[str, Any] = {}

    def reset(self, variant: Variant) -> Observation:
        self._current_src = variant.spec["buggy_src"]
        self._test_src = variant.spec["test_src"]
        self._ctx = {"family": variant.family, "failure_type": variant.spec.get("failure_type")}
        # Grading runs pytest; reset only presents the task (keeps the loop cheap).
        return Observation(
            text=(f"Buggy module (its test currently fails):\n{self._current_src}\n"
                  f"Test:\n{self._test_src}\n"
                  "Call apply_fix(source=<full corrected module source>)."),
            done=False,
        )

    def step(self, action_name: str, args: dict[str, Any]) -> Observation:
        if action_name == "apply_fix" and "source" in args:
            self._current_src = str(args["source"])
            return Observation(text="fix applied", done=True)
        return Observation(text=f"unknown action {action_name!r}", done=False)

    def grade(self) -> RewardVector:
        res = run_pytest(self._current_src, self._test_src)
        return RewardVector.from_success(res.passed, efficiency=1.0, cost=1.0, latency=1.0)

    def tool_schemas(self) -> list[dict]:
        return [{
            "type": "function",
            "function": {
                "name": "apply_fix",
                "description": "Replace the module source with a corrected version.",
                "parameters": {
                    "type": "object",
                    "properties": {"source": {"type": "string"}},
                    "required": ["source"],
                },
            },
        }]

    def context(self) -> dict[str, Any]:
        return dict(self._ctx)


class ToyBugFamily:
    family_id = "toy_bug"

    def train_variants(self) -> Iterable[Variant]:
        return [_make_variant(*b) for b in _BUGS]

    def heldout_variants(self) -> Iterable[Variant]:
        return [_make_variant(*b, heldout=True) for b in _HELDOUT]

    def make_env(self) -> Environment:
        return ToyBugEnv()
