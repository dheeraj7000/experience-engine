"""Real QA family #2 (see qa_families.py spec): flaky test triage.

Task: given a test that fails intermittently, DIAGNOSE the root cause of the
flakiness and PROPOSE a fix. The agent must identify whether the flakiness is
caused by timing, ordering, shared state, randomness, or concurrency — and
produce a corrected test that passes reliably.

Reward is execution-grounded and multi-dimensional:
  - Reproducibility: the fixed test passes consistently over N runs
  - Root cause: the diagnosed cause matches the injected fault type
  - Efficiency: fewer steps to reach the diagnosis

Flakiness is INJECTED synthetically: each variant has a deterministic "stable"
version and a flaky version with a specific fault class. The agent sees the
flaky version + failure logs and must fix it.
"""
from __future__ import annotations

from typing import Any, Iterable

from agent.tools import run_pytest
from harness.environment import Environment, Observation, Variant
from schemas import RewardVector


# Fault classes for injected flakiness:
#   timing       - test depends on sleep/timing that occasionally races
#   order_depend - test depends on execution order (shared mutable state)
#   random_seed  - test uses random without seeding
#   global_state - test pollutes module-level state across runs
#   float_approx - test uses exact float comparison that drifts

# (name, flaky_module, stable_module, flaky_test, stable_test, fault_type, diagnosis_hint)
_FLAKY_VARIANTS = [
    (
        "timing_cache",
        # Module: cache with TTL — flaky because test sleeps too little
        (
            "import time\n\n"
            "_cache = {}\n_timestamps = {}\n\n"
            "def get_cached(key, ttl=0.1):\n"
            "    if key in _cache and (time.time() - _timestamps[key]) < ttl:\n"
            "        return _cache[key]\n"
            "    return None\n\n"
            "def set_cached(key, value):\n"
            "    _cache[key] = value\n"
            "    _timestamps[key] = time.time()\n"
        ),
        # Stable module (same)
        (
            "import time\n\n"
            "_cache = {}\n_timestamps = {}\n\n"
            "def get_cached(key, ttl=0.1):\n"
            "    if key in _cache and (time.time() - _timestamps[key]) < ttl:\n"
            "        return _cache[key]\n"
            "    return None\n\n"
            "def set_cached(key, value):\n"
            "    _cache[key] = value\n"
            "    _timestamps[key] = time.time()\n"
        ),
        # Flaky test: sleeps 0.05s but TTL is 0.1s — SOMETIMES the sleep
        # plus execution overhead exceeds TTL, sometimes not.
        (
            "import time\nfrom solution import get_cached, set_cached\n\n"
            "def test_cache_expires():\n"
            "    set_cached('x', 42)\n"
            "    time.sleep(0.05)  # less than TTL, should still be cached\n"
            "    assert get_cached('x') == 42  # flaky: sometimes execution delay makes it expire\n"
        ),
        # Stable test: uses explicit time mocking or generous margin
        (
            "import time\nfrom solution import get_cached, set_cached\n\n"
            "def test_cache_expires():\n"
            "    set_cached('x', 42)\n"
            "    # Well within TTL — no timing dependency\n"
            "    assert get_cached('x', ttl=10.0) == 42\n"
        ),
        "timing",
        "Test relies on sleep duration close to TTL boundary — execution "
        "jitter causes intermittent expiration.",
    ),
    (
        "order_counter",
        # Module: global counter
        (
            "counter = 0\n\n"
            "def increment():\n"
            "    global counter\n"
            "    counter += 1\n"
            "    return counter\n\n"
            "def reset():\n"
            "    global counter\n"
            "    counter = 0\n"
        ),
        (
            "counter = 0\n\n"
            "def increment():\n"
            "    global counter\n"
            "    counter += 1\n"
            "    return counter\n\n"
            "def reset():\n"
            "    global counter\n"
            "    counter = 0\n"
        ),
        # Flaky test: doesn't reset between tests — order-dependent
        (
            "from solution import increment, reset\n\n"
            "def test_first_increment():\n"
            "    assert increment() == 1\n\n"
            "def test_second_increment():\n"
            "    assert increment() == 2\n"
        ),
        # Stable test: resets state in each test
        (
            "from solution import increment, reset\n\n"
            "def test_first_increment():\n"
            "    reset()\n"
            "    assert increment() == 1\n\n"
            "def test_second_increment():\n"
            "    reset()\n"
            "    assert increment() == 1\n"
        ),
        "order_depend",
        "Tests share global state (counter) without resetting — result depends "
        "on execution order.",
    ),
    (
        "random_sample",
        # Module: random sampling without seed
        (
            "import random\n\n"
            "def pick_top(items, n):\n"
            "    sample = random.sample(items, min(n, len(items)))\n"
            "    return sorted(sample)\n"
        ),
        (
            "import random\n\n"
            "def pick_top(items, n, seed=None):\n"
            "    rng = random.Random(seed)\n"
            "    sample = rng.sample(items, min(n, len(items)))\n"
            "    return sorted(sample)\n"
        ),
        # Flaky test: asserts specific output but random changes each run
        (
            "from solution import pick_top\n\n"
            "def test_pick_top():\n"
            "    result = pick_top([1, 2, 3, 4, 5], 3)\n"
            "    assert result == [1, 2, 3]  # only true sometimes\n"
        ),
        # Stable test: uses seed or tests invariants
        (
            "from solution import pick_top\n\n"
            "def test_pick_top():\n"
            "    result = pick_top([1, 2, 3, 4, 5], 3)\n"
            "    assert len(result) == 3\n"
            "    assert all(x in [1, 2, 3, 4, 5] for x in result)\n"
            "    assert result == sorted(result)\n"
        ),
        "random_seed",
        "Test asserts exact output of an unseeded random function — result "
        "varies between runs.",
    ),
    (
        "global_registry",
        # Module: global registry that leaks across tests
        (
            "_registry = []\n\n"
            "def register(name):\n"
            "    _registry.append(name)\n\n"
            "def list_registered():\n"
            "    return list(_registry)\n\n"
            "def clear():\n"
            "    _registry.clear()\n"
        ),
        (
            "_registry = []\n\n"
            "def register(name):\n"
            "    _registry.append(name)\n\n"
            "def list_registered():\n"
            "    return list(_registry)\n\n"
            "def clear():\n"
            "    _registry.clear()\n"
        ),
        # Flaky test: doesn't clear registry — accumulates across test runs
        (
            "from solution import register, list_registered\n\n"
            "def test_register_one():\n"
            "    register('alice')\n"
            "    assert list_registered() == ['alice']\n"
        ),
        # Stable test: clears before each test
        (
            "from solution import register, list_registered, clear\n\n"
            "def test_register_one():\n"
            "    clear()\n"
            "    register('alice')\n"
            "    assert list_registered() == ['alice']\n"
        ),
        "global_state",
        "Test pollutes global registry without cleanup — accumulates entries "
        "across repeated runs or test collection.",
    ),
    (
        "float_precision",
        # Module: floating point arithmetic
        (
            "def circle_area(radius):\n"
            "    import math\n"
            "    return math.pi * radius * radius\n"
        ),
        (
            "def circle_area(radius):\n"
            "    import math\n"
            "    return math.pi * radius * radius\n"
        ),
        # Flaky test: exact float equality (fragile across platforms)
        (
            "from solution import circle_area\n\n"
            "def test_circle_area():\n"
            "    assert circle_area(1.0) == 3.141592653589793\n"
            "    assert circle_area(2.5) == 19.634954084936208\n"
        ),
        # Stable test: uses approximate comparison
        (
            "import math\nfrom solution import circle_area\n\n"
            "def test_circle_area():\n"
            "    assert math.isclose(circle_area(1.0), math.pi, rel_tol=1e-9)\n"
            "    assert math.isclose(circle_area(2.5), math.pi * 6.25, rel_tol=1e-9)\n"
        ),
        "float_approx",
        "Test uses exact float equality — fragile to platform/compiler "
        "differences in floating-point representation.",
    ),
    (
        "env_variable",
        # Module: reads from environment
        (
            "import os\n\n"
            "def get_mode():\n"
            "    return os.environ.get('APP_MODE', 'production')\n"
        ),
        (
            "import os\n\n"
            "def get_mode():\n"
            "    return os.environ.get('APP_MODE', 'production')\n"
        ),
        # Flaky test: depends on environment variable that may or may not be set
        (
            "from solution import get_mode\n\n"
            "def test_default_mode():\n"
            "    assert get_mode() == 'production'\n"
        ),
        # Stable test: explicitly controls environment
        (
            "import os\nfrom solution import get_mode\n\n"
            "def test_default_mode(monkeypatch):\n"
            "    monkeypatch.delenv('APP_MODE', raising=False)\n"
            "    assert get_mode() == 'production'\n"
        ),
        "global_state",
        "Test assumes environment variable is unset — fails when APP_MODE "
        "is set in the CI/testing environment.",
    ),
]

_HELDOUT_VARIANTS = [
    (
        "dict_order",
        (
            "def first_key(d):\n"
            "    return list(d.keys())[0]\n"
        ),
        (
            "def first_key(d):\n"
            "    return list(d.keys())[0]\n"
        ),
        # Flaky: dict ordering not guaranteed in older Python conceptually,
        # but more importantly the test constructs dict from kwargs which
        # might not preserve insertion order in some edge cases.
        (
            "from solution import first_key\n\n"
            "def test_first_key():\n"
            "    d = dict(b=2, a=1, c=3)  # order may vary\n"
            "    assert first_key(d) == 'b'\n"
        ),
        (
            "from solution import first_key\n\n"
            "def test_first_key():\n"
            "    from collections import OrderedDict\n"
            "    d = OrderedDict([('b', 2), ('a', 1), ('c', 3)])\n"
            "    assert first_key(d) == 'b'\n"
        ),
        "order_depend",
        "Test relies on dict construction order from kwargs.",
    ),
]


def _make_variant(name, flaky_mod, stable_mod, flaky_test, stable_test,
                  fault_type, diagnosis, heldout=False) -> Variant:
    return Variant(
        variant_id=f"flaky_{name}",
        family="flaky_test_triage",
        goal=(f"The test in `test_solution.py` fails intermittently. "
              f"Diagnose the root cause and submit a fixed test."),
        spec={
            "flaky_module": flaky_mod,
            "stable_module": stable_mod,
            "flaky_test": flaky_test,
            "stable_test": stable_test,
            "fault_type": fault_type,
            "diagnosis": diagnosis,
            "function_name": name,
            "failure_type": "flaky_test",
        },
        heldout=heldout,
    )


class FlakyTestEnv:
    """Environment for flaky test triage.

    The agent sees the flaky test + module and must:
    1. Diagnose the root cause (submit_diagnosis)
    2. Submit a fixed test (submit_fix)

    Grading:
    - Fixed test must pass reliably (5 consecutive runs)
    - Bonus for correct root cause identification
    """

    def __init__(self) -> None:
        self._module_src = ""
        self._flaky_test = ""
        self._stable_test = ""
        self._fault_type = ""
        self._diagnosis_hint = ""
        self._submitted_diagnosis = ""
        self._submitted_fix = ""
        self._ctx: dict[str, Any] = {}

    def reset(self, variant: Variant) -> Observation:
        self._module_src = variant.spec["flaky_module"]
        self._flaky_test = variant.spec["flaky_test"]
        self._stable_test = variant.spec["stable_test"]
        self._fault_type = variant.spec["fault_type"]
        self._diagnosis_hint = variant.spec["diagnosis"]
        self._submitted_diagnosis = ""
        self._submitted_fix = ""
        self._ctx = {"family": variant.family,
                     "failure_type": variant.spec.get("failure_type")}
        return Observation(
            text=(
                f"This test fails intermittently:\n\n"
                f"```python\n{self._flaky_test}```\n\n"
                f"Module under test:\n```python\n{self._module_src}```\n\n"
                "Diagnose why this test is flaky and submit a fix.\n"
                "1. Call submit_diagnosis(cause=<root cause description>)\n"
                "2. Call submit_fix(test_source=<full fixed test module>)"
            ),
            done=False,
        )

    def step(self, action_name: str, args: dict[str, Any]) -> Observation:
        if action_name == "submit_diagnosis" and "cause" in args:
            self._submitted_diagnosis = str(args["cause"])
            return Observation(
                text="Diagnosis recorded. Now submit the fixed test.",
                done=False,
            )
        if action_name == "submit_fix" and "test_source" in args:
            self._submitted_fix = str(args["test_source"])
            return Observation(text="Fix submitted.", done=True)
        return Observation(text=f"Unknown action {action_name!r}. "
                           "Use submit_diagnosis or submit_fix.", done=False)

    def grade(self) -> RewardVector:
        if not self._submitted_fix.strip():
            return RewardVector.from_success(False)

        # Run the fixed test multiple times to check it's stable.
        passes = 0
        n_runs = 5
        for _ in range(n_runs):
            res = run_pytest(self._module_src, self._submitted_fix)
            if res.passed:
                passes += 1

        reproducible = passes == n_runs
        pass_rate = passes / n_runs

        # Check if diagnosis matches the injected fault type.
        diagnosis_correct = self._check_diagnosis()

        if reproducible:
            # Full success: stable test + correct diagnosis.
            efficiency = 1.0 if diagnosis_correct else 0.7
            return RewardVector.from_success(True, efficiency=efficiency,
                                            cost=1.0, latency=1.0)

        # Partial credit for mostly-stable fix.
        partial = pass_rate * 0.6 + (0.2 if diagnosis_correct else 0.0)
        rv = RewardVector(
            task_success=pass_rate if pass_rate >= 0.8 else 0.0,
            partial_credit=partial,
            efficiency=0.5 if diagnosis_correct else 0.0,
            reproducible=reproducible,
        )
        return rv.compute_overall()

    def _check_diagnosis(self) -> bool:
        """Check if the submitted diagnosis matches the fault type."""
        if not self._submitted_diagnosis:
            return False
        diag = self._submitted_diagnosis.lower()
        # Map fault types to expected keywords in diagnosis.
        keywords = {
            "timing": ["timing", "sleep", "race", "ttl", "expir", "delay"],
            "order_depend": ["order", "state", "reset", "shared", "global",
                            "depend", "isolation"],
            "random_seed": ["random", "seed", "nondetermin", "unpredictab"],
            "global_state": ["global", "state", "leak", "pollut", "cleanup",
                            "environment", "env"],
            "float_approx": ["float", "precision", "approx", "tolerance",
                            "isclose", "epsilon"],
        }
        expected = keywords.get(self._fault_type, [])
        return any(kw in diag for kw in expected)

    def tool_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "submit_diagnosis",
                    "description": "Submit your diagnosis of why the test is flaky.",
                    "parameters": {
                        "type": "object",
                        "properties": {"cause": {"type": "string"}},
                        "required": ["cause"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "submit_fix",
                    "description": (
                        "Submit a fixed test module that passes reliably "
                        "(should not be flaky)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"test_source": {"type": "string"}},
                        "required": ["test_source"],
                    },
                },
            },
        ]

    def context(self) -> dict[str, Any]:
        return dict(self._ctx)


class FlakyTestTriageFamily:
    family_id = "flaky_test_triage"

    def train_variants(self) -> Iterable[Variant]:
        return [_make_variant(*v) for v in _FLAKY_VARIANTS]

    def heldout_variants(self) -> Iterable[Variant]:
        return [_make_variant(*v, heldout=True) for v in _HELDOUT_VARIANTS]

    def make_env(self) -> Environment:
        return FlakyTestEnv()
