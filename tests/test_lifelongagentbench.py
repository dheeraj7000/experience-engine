"""Tests for LifelongAgentBench adapter — unit tests that don't require
Docker or HuggingFace network access. Uses mocked dataset rows."""
import json
import tempfile
from pathlib import Path

from benchmarks.lifelongagentbench.adapter import (
    LifelongAgentBenchFamily,
    DBBenchEnv, OSInteractionEnv, KnowledgeGraphEnv,
    _make_variant, _group_by_env, _infer_env,
)
from harness.environment import Variant
from schemas import RewardVector


# ---- Mock dataset rows ---------------------------------------------------

_MOCK_ROWS = [
    {
        "task_id": "db_001",
        "environment_type": "db_bench",
        "instruction": "Write a SQL query to select all users older than 30.",
        "answer": "SELECT * FROM users WHERE age > 30",
        "context": "Table: users (id INT, name TEXT, age INT)",
        "skill_type": "sql_filtering",
    },
    {
        "task_id": "db_002",
        "environment_type": "db_bench",
        "instruction": "Count the number of orders placed in 2025.",
        "answer": "SELECT COUNT(*) FROM orders WHERE year = 2025",
        "context": "Table: orders (id INT, product TEXT, year INT)",
        "skill_type": "sql_aggregation",
    },
    {
        "task_id": "os_001",
        "environment_type": "os_interaction",
        "instruction": "List all .py files in the current directory.",
        "answer": "ls *.py",
        "context": "",
        "skill_type": "file_listing",
    },
    {
        "task_id": "os_002",
        "environment_type": "os_interaction",
        "instruction": "Find the total disk usage of /home.",
        "answer": "du -sh /home",
        "context": "",
        "skill_type": "disk_usage",
    },
    {
        "task_id": "kg_001",
        "environment_type": "knowledge_graph",
        "instruction": "What is the capital of France?",
        "answer": "Paris",
        "context": "",
        "skill_type": "entity_lookup",
    },
]


def _write_cache(rows, tmpdir):
    cache_path = Path(tmpdir) / "data.json"
    cache_path.write_text(json.dumps(rows))
    return Path(tmpdir)


# ---- Tests ----------------------------------------------------------------

def test_infer_env_db():
    row = {"instruction": "Write a SQL query to select users from database"}
    assert _infer_env(row) == "db_bench"


def test_infer_env_os():
    row = {"instruction": "Run a bash command to list directory contents"}
    assert _infer_env(row) == "os_interaction"


def test_infer_env_kg():
    row = {"instruction": "Find the entity with the relation 'capital_of'"}
    assert _infer_env(row) == "knowledge_graph"


def test_group_by_env():
    groups = _group_by_env(_MOCK_ROWS)
    assert len(groups["db_bench"]) == 2
    assert len(groups["os_interaction"]) == 2
    assert len(groups["knowledge_graph"]) == 1


def test_make_variant():
    v = _make_variant(_MOCK_ROWS[0], 0, "db_bench")
    assert v.variant_id == "lab_db_bench_db_001"
    assert v.family == "lifelongagentbench"
    assert "SQL" in v.goal
    assert v.spec["env_type"] == "db_bench"


def test_family_loads_from_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = _write_cache(_MOCK_ROWS, tmpdir)
        family = LifelongAgentBenchFamily(env_type="db_bench", cache_dir=cache_dir)
        train = list(family.train_variants())
        heldout = list(family.heldout_variants())
        assert len(train) >= 1
        assert len(heldout) >= 1
        assert all(v.spec["env_type"] == "db_bench" for v in train)


def test_family_id_includes_env_type():
    family = LifelongAgentBenchFamily(env_type="os_interaction")
    assert family.family_id == "lifelongagentbench_os_interaction"


def test_family_id_generic():
    family = LifelongAgentBenchFamily()
    assert family.family_id == "lifelongagentbench"


def test_db_env_reset():
    env = DBBenchEnv()
    v = _make_variant(_MOCK_ROWS[0], 0, "db_bench")
    obs = env.reset(v)
    assert "Database task" in obs.text
    assert "SQL" in obs.text or "select" in obs.text.lower()
    assert not obs.done


def test_db_env_submit():
    env = DBBenchEnv()
    v = _make_variant(_MOCK_ROWS[0], 0, "db_bench")
    env.reset(v)
    obs = env.step("submit_answer", {"answer": "SELECT * FROM users WHERE age > 30"})
    assert obs.done


def test_db_env_grade_correct():
    env = DBBenchEnv()
    v = _make_variant(_MOCK_ROWS[0], 0, "db_bench")
    env.reset(v)
    env.step("submit_answer", {"answer": "SELECT * FROM users WHERE age > 30"})
    reward = env.grade()
    # String match should succeed (no Docker needed for exact match).
    assert reward.task_success == 1.0


def test_db_env_grade_wrong():
    env = DBBenchEnv()
    v = _make_variant(_MOCK_ROWS[0], 0, "db_bench")
    env.reset(v)
    env.step("submit_answer", {"answer": "SELECT * FROM orders"})
    reward = env.grade()
    assert reward.task_success == 0.0


def test_os_env_reset_and_submit():
    env = OSInteractionEnv()
    v = _make_variant(_MOCK_ROWS[2], 0, "os_interaction")
    env.reset(v)
    obs = env.step("submit_answer", {"answer": "ls *.py"})
    assert obs.done


def test_os_env_grade_correct():
    env = OSInteractionEnv()
    v = _make_variant(_MOCK_ROWS[2], 0, "os_interaction")
    env.reset(v)
    env.step("submit_answer", {"answer": "ls *.py"})
    reward = env.grade()
    assert reward.task_success == 1.0


def test_kg_env_grade_correct():
    env = KnowledgeGraphEnv()
    v = _make_variant(_MOCK_ROWS[4], 0, "knowledge_graph")
    env.reset(v)
    env.step("submit_answer", {"answer": "Paris"})
    reward = env.grade()
    assert reward.task_success == 1.0


def test_kg_env_grade_case_insensitive():
    env = KnowledgeGraphEnv()
    v = _make_variant(_MOCK_ROWS[4], 0, "knowledge_graph")
    env.reset(v)
    env.step("submit_answer", {"answer": "paris"})
    reward = env.grade()
    assert reward.task_success == 1.0


def test_kg_env_grade_wrong():
    env = KnowledgeGraphEnv()
    v = _make_variant(_MOCK_ROWS[4], 0, "knowledge_graph")
    env.reset(v)
    env.step("submit_answer", {"answer": "London"})
    reward = env.grade()
    assert reward.task_success == 0.0


def test_empty_submission_fails():
    env = DBBenchEnv()
    v = _make_variant(_MOCK_ROWS[0], 0, "db_bench")
    env.reset(v)
    env.step("submit_answer", {"answer": ""})
    reward = env.grade()
    assert reward.task_success == 0.0


def test_tool_schemas():
    for EnvCls in (DBBenchEnv, OSInteractionEnv, KnowledgeGraphEnv):
        env = EnvCls()
        schemas = env.tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "submit_answer"


def test_max_variants_limit():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = _write_cache(_MOCK_ROWS, tmpdir)
        family = LifelongAgentBenchFamily(cache_dir=cache_dir, max_variants=3)
        train = list(family.train_variants())
        heldout = list(family.heldout_variants())
        assert len(train) + len(heldout) <= 3
