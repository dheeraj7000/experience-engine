"""Execution-grounded grading: write module + test to a temp dir and run
pytest in a subprocess. This is the reward substrate for QA tasks — success
is a fact about executed code, not an opinion.

Sandboxing note (Phase 0): subprocess + temp dir + timeout is the minimum.
Harden to containers before running untrusted / model-authored code at scale.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PytestResult:
    passed: bool
    returncode: int
    output: str


def run_pytest(module_src: str, test_src: str, timeout: float = 30.0,
               module_name: str = "solution") -> PytestResult:
    """Run `test_src` against `module_src`. Returns pass/fail + captured output."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / f"{module_name}.py").write_text(module_src)
        (root / "test_solution.py").write_text(test_src)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--no-header", "test_solution.py"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            return PytestResult(passed=proc.returncode == 0, returncode=proc.returncode, output=out[-4000:])
        except subprocess.TimeoutExpired:
            return PytestResult(passed=False, returncode=-1, output="TIMEOUT")
