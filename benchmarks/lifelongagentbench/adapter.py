"""LifelongAgentBench adapter — bridges the upstream benchmark into our harness.

Upstream: github.com/caixd-220529/LifelongAgentBench (arXiv:2505.11942)
Dataset: huggingface.co/datasets/csyq/LifelongAgentBench (1,396 rows)

The benchmark provides skill-grounded, interdependent tasks across three
interactive environments: Database (SQL), Operating System (shell), and
Knowledge Graph (queries). Reward is execution-grounded for DB and OS tasks.

This adapter:
  1. Loads the HuggingFace dataset (auto-downloads on first use)
  2. Groups tasks by environment type
  3. Maps each task into our Variant + Environment protocol
  4. Provides execution-grounded grading where available

For DB tasks: executes SQL against a Docker MySQL container and checks results.
For OS tasks: executes shell commands in a Docker Ubuntu container and checks output.
For KG tasks: uses the upstream's string-match grading (less execution-grounded).

Requirements:
  pip install datasets  (HuggingFace datasets library)
  docker pull mysql     (for db_bench tasks)
  docker pull ubuntu    (for os_interaction tasks)
"""
from __future__ import annotations

import json
import subprocess
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from harness.environment import Environment, Observation, Variant
from schemas import RewardVector


# ---- Dataset loading -------------------------------------------------------

def _load_dataset(cache_dir: str | Path | None = None) -> list[dict]:
    """Load the LifelongAgentBench dataset from HuggingFace.

    The dataset has different schemas per environment subset (stored as
    separate parquet files), so we load each subset individually.
    """
    cache_path = Path(cache_dir or ".cache/lifelongagentbench") / "data.json"

    # Try cached first.
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    try:
        from datasets import load_dataset

        all_rows: list[dict] = []
        # The dataset stores each env as a separate config (subdirectory).
        for config_name in ("db_bench", "os_interaction", "knowledge_graph"):
            try:
                ds = load_dataset(
                    "csyq/LifelongAgentBench",
                    data_files=f"{config_name}/train-*.parquet",
                    split="train",
                )
                for row in ds:
                    normalized = _normalize_row(dict(row), config_name)
                    all_rows.append(normalized)
            except Exception:
                continue

        if not all_rows:
            raise RuntimeError("Could not load any parquet files from the dataset")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(all_rows, default=str))
        return all_rows
    except ImportError:
        raise RuntimeError(
            "Install the `datasets` library: pip install datasets\n"
            "Or place cached data at: " + str(cache_path)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load LifelongAgentBench dataset: {e}")


def _normalize_row(row: dict, env_type: str) -> dict:
    """Normalize different schema formats into a unified structure."""
    # Map various column names to our standard fields.
    instruction = (
        row.get("instruction")
        or row.get("question")
        or row.get("task")
        or row.get("prompt")
        or ""
    )
    answer = (
        row.get("answer_info")
        or row.get("answer_list")
        or row.get("expected_output")
        or row.get("answer")
        or ""
    )
    context = (
        row.get("table_info")
        or row.get("context")
        or row.get("entity_dict")
        or row.get("setup")
        or ""
    )
    task_id = (
        row.get("sql_instruction_row_list_entry_hash")
        or row.get("qid")
        or row.get("sample_index")
        or row.get("task_id")
        or row.get("id")
        or ""
    )
    skill_list = row.get("skill_list", "")

    return {
        "task_id": str(task_id),
        "environment_type": env_type,
        "instruction": str(instruction),
        "answer": str(answer),
        "context": str(context),
        "skill_type": str(skill_list),
    }


def _group_by_env(rows: list[dict]) -> dict[str, list[dict]]:
    """Group dataset rows by environment type."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        env_type = row.get("environment_type") or row.get("env_type") or _infer_env(row)
        groups.setdefault(env_type, []).append(row)
    return groups


def _infer_env(row: dict) -> str:
    """Infer environment type from task content when not explicitly labeled."""
    instruction = str(row.get("instruction", "") or row.get("task", "")).lower()
    if any(kw in instruction for kw in ("sql", "query", "table", "database", "select")):
        return "db_bench"
    if any(kw in instruction for kw in ("command", "terminal", "shell", "file",
                                         "directory", "linux", "bash")):
        return "os_interaction"
    if any(kw in instruction for kw in ("knowledge", "entity", "relation", "triple")):
        return "knowledge_graph"
    return "unknown"


# ---- Variant construction ---------------------------------------------------

def _make_variant(row: dict, index: int, env_type: str, heldout: bool = False) -> Variant:
    """Convert a dataset row into our Variant format."""
    instruction = row.get("instruction") or row.get("task") or row.get("prompt") or ""
    task_id = row.get("task_id") or row.get("id") or f"{env_type}_{index}"

    return Variant(
        variant_id=f"lab_{env_type}_{task_id}",
        family="lifelongagentbench",
        goal=instruction[:500] if instruction else f"Complete {env_type} task #{index}",
        spec={
            "env_type": env_type,
            "instruction": instruction,
            "expected_answer": row.get("answer") or row.get("expected_output") or "",
            "context": row.get("context") or row.get("setup") or "",
            "skill_type": row.get("skill_type") or row.get("category") or "",
            "task_id": str(task_id),
            "raw_row": row,
            "failure_type": "benchmark_task",
        },
        heldout=heldout,
    )


# ---- Environment implementations -------------------------------------------

class DBBenchEnv:
    """Environment for database (SQL) tasks.

    The agent must produce a SQL query that answers the question.
    Grading: compare query result against expected answer.
    If Docker MySQL is available, executes the query. Otherwise falls back
    to string comparison.
    """

    def __init__(self) -> None:
        self._instruction = ""
        self._expected = ""
        self._context = ""
        self._submitted = ""
        self._ctx: dict[str, Any] = {}

    def reset(self, variant: Variant) -> Observation:
        self._instruction = variant.spec["instruction"]
        self._expected = variant.spec["expected_answer"]
        self._context = variant.spec.get("context", "")
        self._submitted = ""
        self._ctx = {"family": variant.family,
                     "failure_type": variant.spec.get("failure_type"),
                     "env_type": "db_bench"}
        text = f"Database task:\n{self._instruction}\n"
        if self._context:
            text += f"\nContext/Schema:\n{self._context}\n"
        text += "\nSubmit your SQL query using submit_answer(answer=<your SQL query>)."
        return Observation(text=text, done=False)

    def step(self, action_name: str, args: dict[str, Any]) -> Observation:
        if action_name == "submit_answer" and "answer" in args:
            self._submitted = str(args["answer"])
            return Observation(text="Answer submitted.", done=True)
        return Observation(text=f"Unknown action {action_name!r}. Use submit_answer.",
                           done=False)

    def grade(self) -> RewardVector:
        if not self._submitted.strip():
            return RewardVector.from_success(False)

        # Try execution-grounded grading via Docker MySQL.
        exec_result = self._try_execute_sql(self._submitted)
        if exec_result is not None:
            success = self._compare_results(exec_result, self._expected)
            return RewardVector.from_success(success, efficiency=1.0, cost=1.0)

        # Fallback: normalized string comparison.
        success = self._string_match(self._submitted, self._expected)
        rv = RewardVector(
            task_success=1.0 if success else 0.0,
            partial_credit=0.5 if self._partial_match(self._submitted, self._expected) else 0.0,
            reproducible=False,  # string match is less reliable
        )
        return rv.compute_overall()

    def tool_schemas(self) -> list[dict]:
        return [{
            "type": "function",
            "function": {
                "name": "submit_answer",
                "description": "Submit your SQL query or answer.",
                "parameters": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        }]

    def context(self) -> dict[str, Any]:
        return dict(self._ctx)

    @staticmethod
    def _try_execute_sql(query: str) -> str | None:
        """Try to execute SQL via Docker MySQL. Returns result or None."""
        try:
            result = subprocess.run(
                ["docker", "exec", "lab-mysql",
                 "mysql", "-u", "root", "-proot", "-e", query],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return None

    @staticmethod
    def _compare_results(actual: str, expected: str) -> bool:
        return actual.strip().lower() == expected.strip().lower()

    @staticmethod
    def _string_match(submitted: str, expected: str) -> bool:
        s = submitted.strip().lower().replace(";", "")
        e = expected.strip().lower().replace(";", "")
        return s == e

    @staticmethod
    def _partial_match(submitted: str, expected: str) -> bool:
        s_words = set(submitted.lower().split())
        e_words = set(expected.lower().split())
        if not e_words:
            return False
        overlap = len(s_words & e_words) / len(e_words)
        return overlap >= 0.5


class OSInteractionEnv:
    """Environment for operating system (shell) tasks.

    The agent must produce a shell command that accomplishes the goal.
    Grading: execute in Docker Ubuntu and compare output.
    """

    def __init__(self) -> None:
        self._instruction = ""
        self._expected = ""
        self._context = ""
        self._submitted = ""
        self._ctx: dict[str, Any] = {}

    def reset(self, variant: Variant) -> Observation:
        self._instruction = variant.spec["instruction"]
        self._expected = variant.spec["expected_answer"]
        self._context = variant.spec.get("context", "")
        self._submitted = ""
        self._ctx = {"family": variant.family,
                     "failure_type": variant.spec.get("failure_type"),
                     "env_type": "os_interaction"}
        text = f"OS/Shell task:\n{self._instruction}\n"
        if self._context:
            text += f"\nContext:\n{self._context}\n"
        text += "\nSubmit your shell command using submit_answer(answer=<command>)."
        return Observation(text=text, done=False)

    def step(self, action_name: str, args: dict[str, Any]) -> Observation:
        if action_name == "submit_answer" and "answer" in args:
            self._submitted = str(args["answer"])
            return Observation(text="Answer submitted.", done=True)
        return Observation(text=f"Unknown action {action_name!r}. Use submit_answer.",
                           done=False)

    def grade(self) -> RewardVector:
        if not self._submitted.strip():
            return RewardVector.from_success(False)

        # Try execution-grounded grading via Docker.
        exec_result = self._try_execute_shell(self._submitted)
        if exec_result is not None:
            success = self._compare_output(exec_result, self._expected)
            return RewardVector.from_success(success, efficiency=1.0, cost=1.0)

        # Fallback: string comparison.
        success = self._submitted.strip() == self._expected.strip()
        partial = self._partial_match(self._submitted, self._expected)
        rv = RewardVector(
            task_success=1.0 if success else 0.0,
            partial_credit=0.3 if partial else 0.0,
            reproducible=False,
        )
        return rv.compute_overall()

    def tool_schemas(self) -> list[dict]:
        return [{
            "type": "function",
            "function": {
                "name": "submit_answer",
                "description": "Submit your shell command or answer.",
                "parameters": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        }]

    def context(self) -> dict[str, Any]:
        return dict(self._ctx)

    @staticmethod
    def _try_execute_shell(command: str) -> str | None:
        """Try to execute shell command via Docker Ubuntu."""
        try:
            result = subprocess.run(
                ["docker", "exec", "lab-ubuntu", "bash", "-c", command],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return None

    @staticmethod
    def _compare_output(actual: str, expected: str) -> bool:
        return actual.strip() == expected.strip()

    @staticmethod
    def _partial_match(submitted: str, expected: str) -> bool:
        s_words = set(submitted.lower().split())
        e_words = set(expected.lower().split())
        if not e_words:
            return False
        return len(s_words & e_words) / len(e_words) >= 0.5


class KnowledgeGraphEnv:
    """Environment for knowledge graph tasks.

    The agent must answer a question about entities/relations.
    Grading: string match against expected answer (not execution-grounded).
    """

    def __init__(self) -> None:
        self._instruction = ""
        self._expected = ""
        self._submitted = ""
        self._ctx: dict[str, Any] = {}

    def reset(self, variant: Variant) -> Observation:
        self._instruction = variant.spec["instruction"]
        self._expected = variant.spec["expected_answer"]
        self._submitted = ""
        self._ctx = {"family": variant.family,
                     "failure_type": variant.spec.get("failure_type"),
                     "env_type": "knowledge_graph"}
        context = variant.spec.get("context", "")
        text = f"Knowledge Graph task:\n{self._instruction}\n"
        if context:
            text += f"\nContext:\n{context}\n"
        text += "\nSubmit your answer using submit_answer(answer=<your answer>)."
        return Observation(text=text, done=False)

    def step(self, action_name: str, args: dict[str, Any]) -> Observation:
        if action_name == "submit_answer" and "answer" in args:
            self._submitted = str(args["answer"])
            return Observation(text="Answer submitted.", done=True)
        return Observation(text=f"Unknown action {action_name!r}. Use submit_answer.",
                           done=False)

    def grade(self) -> RewardVector:
        if not self._submitted.strip():
            return RewardVector.from_success(False)
        success = self._submitted.strip().lower() == self._expected.strip().lower()
        rv = RewardVector(
            task_success=1.0 if success else 0.0,
            partial_credit=0.3 if self._partial(self._submitted, self._expected) else 0.0,
            reproducible=False,  # string match, not execution-grounded
        )
        return rv.compute_overall()

    def tool_schemas(self) -> list[dict]:
        return [{
            "type": "function",
            "function": {
                "name": "submit_answer",
                "description": "Submit your answer to the knowledge graph question.",
                "parameters": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        }]

    def context(self) -> dict[str, Any]:
        return dict(self._ctx)

    @staticmethod
    def _partial(submitted: str, expected: str) -> bool:
        s = set(submitted.lower().split())
        e = set(expected.lower().split())
        return bool(s & e)


# ---- Environment factory ----------------------------------------------------

_ENV_MAP = {
    "db_bench": DBBenchEnv,
    "os_interaction": OSInteractionEnv,
    "knowledge_graph": KnowledgeGraphEnv,
}


# ---- TaskFamily implementation -----------------------------------------------

class LifelongAgentBenchFamily:
    """Adapter that bridges LifelongAgentBench into our harness.

    Usage:
        family = LifelongAgentBenchFamily(env_type="db_bench")
        # or
        family = LifelongAgentBenchFamily()  # uses all environments

    The family_id includes the environment type for scoping.
    """

    def __init__(self, env_type: str | None = None,
                 cache_dir: str | Path | None = None,
                 heldout_fraction: float = 0.15,
                 max_variants: int | None = None) -> None:
        self.env_type = env_type
        self.cache_dir = cache_dir
        self.heldout_fraction = heldout_fraction
        self.max_variants = max_variants
        self._loaded = False
        self._train: list[Variant] = []
        self._heldout: list[Variant] = []

    @property
    def family_id(self) -> str:
        if self.env_type:
            return f"lifelongagentbench_{self.env_type}"
        return "lifelongagentbench"

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        rows = _load_dataset(self.cache_dir)
        groups = _group_by_env(rows)

        if self.env_type:
            target_rows = groups.get(self.env_type, [])
        else:
            target_rows = rows

        if self.max_variants:
            target_rows = target_rows[:self.max_variants]

        # Split into train/heldout.
        n_heldout = max(1, int(len(target_rows) * self.heldout_fraction))
        train_rows = target_rows[:-n_heldout] if n_heldout < len(target_rows) else target_rows
        heldout_rows = target_rows[-n_heldout:] if n_heldout < len(target_rows) else []

        env_type = self.env_type or "mixed"
        self._train = [_make_variant(r, i, _infer_env(r) if not self.env_type else env_type)
                       for i, r in enumerate(train_rows)]
        self._heldout = [_make_variant(r, i, _infer_env(r) if not self.env_type else env_type,
                                       heldout=True)
                         for i, r in enumerate(heldout_rows)]
        self._loaded = True

    def train_variants(self) -> Iterable[Variant]:
        self._ensure_loaded()
        return self._train

    def heldout_variants(self) -> Iterable[Variant]:
        self._ensure_loaded()
        return self._heldout

    def make_env(self) -> Environment:
        """Create the appropriate environment based on env_type."""
        if self.env_type and self.env_type in _ENV_MAP:
            return _ENV_MAP[self.env_type]()
        # Default: DB bench (most execution-grounded).
        return DBBenchEnv()


# ---- Docker setup helper ---------------------------------------------------

def setup_docker_containers() -> dict[str, bool]:
    """Start Docker containers needed for execution-grounded grading.

    Returns status of each container.
    """
    status: dict[str, bool] = {}

    # MySQL container for db_bench.
    try:
        subprocess.run(
            ["docker", "run", "-d", "--name", "lab-mysql",
             "-e", "MYSQL_ROOT_PASSWORD=root",
             "-p", "3306:3306", "mysql:8.0"],
            capture_output=True, timeout=30,
        )
        status["lab-mysql"] = True
    except Exception:
        status["lab-mysql"] = False

    # Ubuntu container for os_interaction.
    try:
        subprocess.run(
            ["docker", "run", "-d", "--name", "lab-ubuntu",
             "ubuntu:22.04", "tail", "-f", "/dev/null"],
            capture_output=True, timeout=30,
        )
        status["lab-ubuntu"] = True
    except Exception:
        status["lab-ubuntu"] = False

    return status


def teardown_docker_containers() -> None:
    """Stop and remove Docker containers."""
    for name in ("lab-mysql", "lab-ubuntu"):
        subprocess.run(["docker", "rm", "-f", name],
                       capture_output=True, timeout=10)
