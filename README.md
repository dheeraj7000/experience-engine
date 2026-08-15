# Experience Engine

A continual-learning architecture that transforms agent trajectories into
**validated, reusable competence** — not just retrievable memory.

> Current agents treat the past as **context** (retrieval).  
> The Experience Engine treats the past as **training signal**.

[![Tests](https://img.shields.io/badge/tests-187%20passing-brightgreen)]()
[![Phases](https://img.shields.io/badge/phases-7%2F7%20complete-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

**[Live Demo Site](https://dheeraj7000.github.io/experience-engine/)** ·
**[Full Research Proposal](docs/EXPERIENCE_ENGINE_PROPOSAL.md)** ·
**[Paper Reference (arXiv:2505.11942)](https://arxiv.org/abs/2505.11942)**

---

## Table of Contents

- [Overview](#overview)
- [The Central Hypothesis](#the-central-hypothesis)
- [Architecture](#architecture)
- [The Three Configurations](#the-three-configurations)
- [Project Layout](#project-layout)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Model Routing](#model-routing)
- [Task Families](#task-families)
- [Benchmarks](#benchmarks)
- [Implementation Phases](#implementation-phases)
- [How the Learning Loop Works](#how-the-learning-loop-works)
- [Key Design Decisions](#key-design-decisions)
- [Testing](#testing)
- [GPU Requirements](#gpu-requirements)
- [Contributing](#contributing)

---

## Overview

The Experience Engine is a **Phase 0–7 research prototype** that implements the
full continual-learning pipeline described in the
[research proposal](docs/EXPERIENCE_ENGINE_PROPOSAL.md). It answers one question:

> **On recurring QA task families, does an Experience Engine (A2) show a steeper
> success curve than a memory-only agent (A1), with fewer repeated failures —
> net of overhead?**

The system records agent episodes, evaluates outcomes via real code execution
(pytest), diagnoses root causes, induces conditional lessons, **validates on
held-out data before promotion**, compiles skills from successes, manages
policy conflicts, decays stale knowledge, and governs safety — all without
updating model weights.

---

## The Central Hypothesis

Treating persistence as an **experience-to-policy pipeline** rather than a
retrieval-only memory system yields steeper, safer learning curves:

```
Record → Evaluate → Diagnose → Induce → Validate → Compile → Policy → Forget → Govern
```

Expected improvements over memory-only (A1):
- Steeper success curve over accumulated episodes
- Fewer repeated failures (same mistake doesn't recur)
- Transfer across related task variants
- Calibrated confidence (knows when to apply vs not apply a lesson)
- Controlled forgetting (no infinite rule accumulation)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       EXPERIENCE ENGINE (A2)                       │
│                                                                    │
│  Episode Recorder → Outcome Evaluator → Causal Diagnoser          │
│       ↓                                        ↓                  │
│  Pattern Miner ← Experience Inducer ← Experience Validator        │
│  Contradiction      ↓                        ↓                    │
│  Miner          Skill Compiler → Policy Manager (conflict/version)│
│       ↓                                        ↓                  │
│  Forgetting Manager    Governance & Audit    Exploration Manager   │
└──────────────────────────────────────────────────────────────────┘
         ↕                    ↕                       ↕
┌─────────────┐  ┌────────────────────┐  ┌──────────────────────┐
│ Episode     │  │ Experience Graph   │  │ Skill Library &      │
│ Store       │  │ (typed nodes/edges)│  │ Policy Registry      │
│ (JSONL)     │  │                    │  │                      │
└─────────────┘  └────────────────────┘  └──────────────────────┘
```

**Online path** (during task execution): lightweight retrieval of relevant
experiences, skills, and policies via the hybrid multi-signal retriever.

**Offline path** (between checkpoints): heavier pattern mining, contradiction
resolution, validation, skill compilation, policy revision, and decay.

---

## The Three Configurations

Same model, same tools, same environment — **only the persistence layer changes**:

| Config | Persistence Layer | What It Represents |
|--------|-------------------|--------------------|
| **A0** | `NoPersistence` — stateless | Today's default agent |
| **A1** | `MemoryOnly` — RAG over past episodes + shallow reflections | Reflexion / ExpeL class |
| **A2** | `ExperienceEngine` — full MVES loop | This proposal |

This design makes the comparison **clean**: any difference in the learning curve
is attributable solely to how past episodes are processed, not to model or
environment differences.

---

## Project Layout

```
experience-engine/
├── run.py                    # Experiment entry point (CLI)
├── config/
│   └── models.yaml           # Model routing (online/offline roles)
├── providers/                # Model abstraction layer
│   ├── base.py               #   ModelProvider protocol
│   ├── dry_run.py            #   Stub (no network, for testing)
│   ├── openai_compat.py      #   Any OpenAI-compatible backend
│   └── registry.py           #   Build providers from config
├── schemas/                  # Pydantic data models
│   ├── episode.py            #   Episode, Step, Cost
│   ├── reward.py             #   RewardVector (multi-dimensional)
│   ├── experience.py         #   ExperienceObject, ValidationStatus
│   ├── policy.py             #   PolicyObject
│   └── skill.py              #   SkillObject
├── agent/                    # ReAct controller
│   ├── controller.py         #   Agent loop (identical across A0/A1/A2)
│   └── tools/
│       └── pytest_tool.py    #   Sandboxed pytest execution (reward substrate)
├── persistence/              # The experimental variable
│   ├── base.py               #   PersistenceLayer protocol
│   ├── store.py              #   JSONL store + bag-of-words search
│   ├── graph.py              #   Experience Graph (typed nodes/edges)
│   ├── graph_builder.py      #   Auto-builds graph from store
│   ├── hybrid_retriever.py   #   Multi-signal ranked retrieval
│   ├── a0_none.py            #   A0: stateless baseline
│   ├── a1_memory.py          #   A1: memory-only (Reflexion class)
│   └── a2_engine/            #   A2: the Experience Engine
│       ├── engine.py         #     Main consolidation loop
│       ├── taxonomy.py       #     Failure type classification
│       ├── diagnoser.py      #     Causal diagnosis (CausalChain)
│       ├── pattern_miner.py  #     Feature-based clustering
│       ├── contradiction.py  #     Conflict detection
│       ├── inducer.py        #     Experience induction
│       ├── validator.py      #     Held-out validation gate
│       ├── confidence.py     #     Confidence scoring
│       ├── skill_compiler.py #     Workflow extraction from successes
│       ├── policy.py         #     Full policy lifecycle
│       ├── forgetting.py     #     Decay + staleness + archival
│       ├── governance.py     #     Audit + safety + human review
│       ├── exploration.py    #     Diversity scoring + suggestions
│       └── internalizer.py   #     Internalization candidate scoring
├── harness/                  # Experiment infrastructure
│   ├── environment.py        #   Environment + Variant protocols
│   ├── task_family.py        #   TaskFamily protocol
│   ├── sequencer.py          #   Orders variants into episodes
│   ├── runner.py             #   Executes (config × family × seed)
│   └── reporter.py           #   Metrics + ASCII curves + JSON
├── families/                 # Executable QA task families
│   ├── toy_bug.py            #   Fix arithmetic bugs (smoke test)
│   ├── bug_reproduction.py   #   Write test reproducing a bug
│   ├── flaky_test_triage.py  #   Diagnose + fix flaky tests
│   └── qa_families.py        #   Specs for planned families
├── benchmarks/               # External benchmark adapters
│   └── lifelongagentbench/   #   LifelongAgentBench (1,396 tasks)
│       ├── adapter.py        #     Full adapter (DB/OS/KG envs)
│       └── run_benchmark.py  #     CLI runner
├── tests/                    # 187 tests (unit + integration)
├── docs/
│   ├── EXPERIENCE_ENGINE_PROPOSAL.md  # Full research proposal
│   ├── index.html            # GitHub Pages site
│   └── style.css
└── runs/                     # Experiment outputs (gitignored data)
```

---

## Installation

```bash
git clone https://github.com/dheeraj7000/experience-engine.git
cd experience-engine
pip install -r requirements.txt

# Verify everything works:
pytest  # 187 tests, no model/network needed
```

**Dependencies** (minimal by design):
- `pydantic>=2.5` — schemas and validation
- `pyyaml>=6.0` — config parsing
- `numpy>=1.24` — metrics and curves
- `httpx>=0.24` — HTTP client for model API calls

**Optional** (for benchmarks):
- `datasets` — HuggingFace dataset loading for LifelongAgentBench
- `docker` — execution-grounded grading for benchmark tasks

---

## Quickstart

### 1. Dry run (no model, no GPU, no network)

Exercises the full pipeline with a stub model. Curves are flat (model does
nothing), but all logic runs: clustering, diagnosis, induction, validation,
skill compilation, policy management, forgetting, governance.

```bash
python run.py --family toy_bug --configs a0 a1 a2 --seeds 1 2 3 --provider dry_run
```

### 2. Real learning curves (requires Ollama, 4GB VRAM)

```bash
# Pull the model (one-time, ~2GB download):
ollama pull qwen2.5:3b-instruct

# Run with enough episodes for A2 to learn:
python run.py --family bug_reproduction --provider config \
    --n-episodes 28 --checkpoint-every 7 --max-steps 3
```

### 3. LifelongAgentBench (external benchmark)

```bash
pip install datasets

# Quick test (no Docker needed):
python -m benchmarks.lifelongagentbench.run_benchmark \
    --env-type db_bench --max-variants 10 --provider dry_run

# Full with Docker execution grading:
python -m benchmarks.lifelongagentbench.run_benchmark \
    --env-type db_bench --provider config --setup-docker
```

### Output

```
Compounding curve — success vs episode index (mean over seeds)
  a0: ..........  slope=+0.0000 final=0.00
  a1: ..#.#.....  slope=+0.0120 final=0.14
  a2: .##.###.##  slope=+0.0340 final=0.57

Report: runs/bug_reproduction/report.json
  a0: mean_success=0.000 slope=+0.0000 repeated_failures=0.86
  a1: mean_success=0.107 slope=+0.0120 repeated_failures=0.71
  a2: mean_success=0.392 slope=+0.0340 repeated_failures=0.43
```

*(Example output with a real model — your numbers will vary.)*

---

## Model Routing

Two roles in `config/models.yaml`, routed independently:

| Role | Purpose | Default | Why |
|------|---------|---------|-----|
| **online** | Agent loop (100–1000 episodes) | Local Ollama `qwen2.5:3b-instruct` | No rate limits, cost = hardware |
| **offline** | Consolidation (diagnose/induce) | Same local Ollama | Zero network, zero keys |

Every backend speaks OpenAI `/v1/chat/completions`, so switching is a config
change:

```yaml
roles:
  online:
    provider: openai_compat
    base_url: http://localhost:11434/v1
    model: qwen2.5:3b-instruct
    api_key: ""

  offline:
    provider: openai_compat
    base_url: https://api.groq.com/openai/v1  # free tier
    model: llama-3.3-70b-versatile
    api_key: ${GROQ_API_KEY}
    rpm_limit: 30
```

Compatible backends: Ollama, vLLM, SGLang, llama.cpp, LM Studio (local);
Groq, OpenRouter, Google AI Studio, Cerebras, Together (free-tier hosted).

---

## Task Families

All rewards are **execution-grounded** — run the tests, not an LLM judge.

| Family | Task | Grading | Variants |
|--------|------|---------|----------|
| `toy_bug` | Fix arithmetic bug so test passes | pytest pass/fail | 6 train + 2 held-out |
| `bug_reproduction` | Write a test reproducing a reported bug | Fails on buggy AND passes on fixed | 7 train + 2 held-out |
| `flaky_test_triage` | Diagnose + fix intermittent test | 5 consecutive passes + root cause match | 6 train + 1 held-out |

**Planned** (declared in `qa_families.py`): `test_authoring`, `regression_bisect`, `failure_clustering`.

---

## Benchmarks

### LifelongAgentBench

Adapter for [LifelongAgentBench](https://arxiv.org/abs/2505.11942) — 1,396
tasks across Database (SQL), Operating System (shell), and Knowledge Graph
environments.

```bash
python -m benchmarks.lifelongagentbench.run_benchmark --help
```

Features:
- Auto-downloads from HuggingFace (caches locally)
- Three environment adapters (DBBenchEnv, OSInteractionEnv, KnowledgeGraphEnv)
- Optional Docker containers for execution-grounded SQL/shell grading
- Falls back to string comparison without Docker

---

## Implementation Phases

| Phase | What | Status |
|-------|------|--------|
| 1 | Minimal Engine (recorder, evaluator, schema, retrieval, validation, policy) | ✅ |
| 2 | Causal Diagnosis & Pattern Mining (taxonomy, CausalChain, clustering, contradictions) | ✅ |
| 3 | Experience Graph (typed graph, hybrid retriever, graph-boosted ranking) | ✅ |
| 4 | Skill Compilation (workflow induction, preconditions, validation, versioning) | ✅ |
| 5 | Policy Management (conflicts, scope refinement, versioning, rollback, safety gating) | ✅ |
| 6 | Forgetting & Governance (decay, staleness, audit log, sensitive blocking, human review) | ✅ |
| 7 | Exploration & Internalization (diversity scoring, suggestions, candidate selection) | ✅ |

---

## How the Learning Loop Works

### Online (during task execution)
```python
context = {"family": "bug_reproduction", "goal": "..."}
injected = engine.retrieve(context)
# → "- Policy: [bug_reproduction] When applicable: verify output before submitting"
# → "- Experience (conf 0.82): When this context recurs, check assertion logic..."
# → "- Skill: [bug_fix_workflow] 1. analyze error  2. write fix  3. verify"
```

### Offline (at each checkpoint)
```python
stats = engine.consolidate(replay_fn=replay)
# stats.summary():
# {
#   "clusters_found": 3,
#   "experiences_induced": 1,
#   "experiences_promoted": 1,
#   "skills_compiled": 1,
#   "contradictions_found": 0,
#   "items_decayed": 2,
#   "policy_conflicts_resolved": 0,
#   "items_blocked_governance": 0,
# }
```

### The Anti-Self-Delusion Gate

A candidate experience **cannot** influence behavior until it demonstrably
improves performance on held-out variants:

```python
baseline, improved = replay_fn(experience)  # (0.4, 0.9)
# delta = 0.5 > 0 → promote to active
# If delta <= 0 → REJECTED, never promoted
```

This is the single most important safety mechanism. Without it, the engine
could learn noise and degrade over time.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Execution-grounded reward | `task_success` from pytest, not LLM judge. Trustworthy signal. |
| Held-out validation gate | Anti-self-delusion. Lessons must help on unseen variants. |
| Free/open models only | Zero cost, zero API keys. 4GB VRAM minimum. |
| No weight updates | External, auditable, reversible. Internalization is Phase 7 scoring only. |
| Controlled forgetting | Confidence decays without reinforcement. No infinite rule accumulation. |
| Governance blocking | Sensitive content (credentials, PII) never promoted. Safety-critical scopes need higher confidence. |
| Causal, not surface | Diagnosis infers WHY (root cause + counterfactual), not just THAT failure occurred. |
| Contradiction → refine, not delete | Conflicting lessons narrow their scope rather than being destroyed. |

---

## Testing

```bash
pytest                           # 187 tests, ~15 seconds
pytest tests/test_poisoned_experience.py  # the non-negotiable gate
pytest -k "phase2"               # just Phase 2 tests
pytest -v                        # verbose output
```

### Key test files

| File | What it validates |
|------|-------------------|
| `test_harness_smoke.py` | Execution-grounded reward from real pytest |
| `test_poisoned_experience.py` | Harmful lessons MUST be rejected |
| `test_a2_consolidation.py` | Full cluster→diagnose→induce→validate→policy loop |
| `test_taxonomy.py` | Failure classification |
| `test_pattern_miner.py` | Feature-based clustering |
| `test_contradiction.py` | Conflicting experience detection |
| `test_graph.py` | Graph traversal, serialization |
| `test_hybrid_retriever.py` | Multi-signal ranked retrieval |
| `test_skill_compiler.py` | Workflow induction from successes |
| `test_policy_manager.py` | Conflicts, versioning, rollback, safety |
| `test_forgetting.py` | Decay, reinforcement, archival |
| `test_governance.py` | Audit, risk assessment, blocking |
| `test_exploration.py` | Diversity scoring, exploration suggestions |
| `test_internalizer.py` | Internalization eligibility scoring |
| `test_flaky_triage.py` | Flaky test triage family end-to-end |
| `test_lifelongagentbench.py` | Benchmark adapter (mocked, no Docker) |

---

## GPU Requirements

| Setup | VRAM | Model | Notes |
|-------|------|-------|-------|
| **Testing only** | 0 GB | `dry_run` | No GPU needed, full pipeline exercises |
| **Minimum** | 4 GB | `qwen2.5:3b-instruct` | Default config, verified working |
| **Better** | 8 GB | `qwen2.5:7b-instruct` | Stronger tool-calling |
| **Good** | 16 GB | `qwen2.5:14b-instruct` | Noticeably better reasoning |
| **Split roles** | 4 GB + free API | 3B local + Groq offline | Free tier handles consolidation |
| **CPU-only** | 0 GB | Ollama CPU mode | Works, ~5-10x slower |

A 28-episode run with the 3B model on 4GB VRAM takes ~5-10 minutes.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Run tests (`pytest`)
4. Commit with a descriptive message
5. Push and open a PR

**Areas where contributions are welcome:**
- New task families (see `qa_families.py` for specs)
- Better embedding models for semantic search
- Visualization dashboard for curves + experience graph
- Additional benchmark adapters (SWE-bench, MemoryArena)
- Containerized sandboxing (Docker-based pytest execution)

---

## Citation

If you use this work in research, please cite:

```bibtex
@software{experience_engine_2026,
  title={Experience Engine: Transforming Agent Trajectories into Validated, Reusable Competence},
  author={Dheeraj},
  url={https://github.com/dheeraj7000/experience-engine},
  year={2026}
}
```

---

## License

MIT
