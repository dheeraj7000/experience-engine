# Experience Engine

A continual-learning layer that turns agent trajectories into **validated,
reusable competence** — instead of just retrievable memory. This repo is the
**Phase 0–1 prototype**: the harness, the three-agent experimental spine, and a
minimal Experience Engine, wired end-to-end on an execution-graded QA task.

See [`docs/EXPERIENCE_ENGINE_PROPOSAL.md`](docs/EXPERIENCE_ENGINE_PROPOSAL.md)
for the full research proposal.

## The one idea

> Current agents treat the past as **context** (retrieval). The Experience
> Engine treats the past as **training signal**: record → evaluate → diagnose →
> induce a conditional lesson → **validate on held-out** → update policy.

The whole prototype exists to answer one question cheaply:

> **On recurring QA task families, does the Experience Engine (A2) show a
> steeper success curve than a memory-only agent (A1), with fewer repeated
> failures — net of overhead?**

## Why QA

The hardest module in the architecture is the **Outcome Evaluator** ("did the
agent succeed?"). QA gives you ground truth for free — **run the tests**. The
reward is a fact about executed code, not an LLM judge. That de-risks the whole
loop, and it's why the reward vector's `task_success` comes from `pytest`.

## The three configurations (the only thing that changes)

Same model, same tools, same environment — only the `PersistenceLayer` differs:

| Config | Layer | Represents |
|--------|-------|------------|
| **A0** | `NoPersistence` — stateless | today's default agent |
| **A1** | `MemoryOnly` — RAG over past episodes + reflections | Reflexion / ExpeL |
| **A2** | `ExperienceEngine` — evaluate→diagnose→induce→validate→policy | this proposal |

## Layout

```
providers/        Model abstraction. Any free/open backend behind OpenAI /v1.
                  dry_run (no network) · openai_compat (Ollama/vLLM/Groq/...)
schemas/          Episode · RewardVector · ExperienceObject · PolicyObject (pydantic)
agent/            ReAct controller + sandboxed pytest tool (the reward substrate)
persistence/      base · store (jsonl+vector) · graph · graph_builder ·
                  hybrid_retriever · a0 · a1 · a2_engine/ (the MVES loop)
  a2_engine/        taxonomy · diagnoser · pattern_miner · contradiction ·
                    inducer · validator · confidence · policy · engine
harness/          environment · task_family · sequencer · runner · reporter
families/         toy_bug + bug_reproduction (both executable) + qa_families specs
benchmarks/       lifelongagentbench adapter (Phase-1 anchor, stub)
docs/             the full research proposal, as markdown
tests/            unit · integration · eval-validity (poisoned-experience gate)
run.py            experiment entry point
```

## Quickstart

```bash
pip install -r requirements.txt
pytest                       # full suite, no model needed

# Plumbing demo (network-free stub model -> flat curves, full loop runs):
python run.py --family toy_bug --configs a0 a1 a2 --seeds 1 2 3 --provider dry_run

# Real learning curves need a real online model. config/models.yaml already
# points at a local Ollama endpoint (qwen2.5:3b-instruct, verified tool
# calling on a 4GB-VRAM box) for both roles, so this runs fully offline once
# `ollama pull qwen2.5:3b-instruct` has been done:
python run.py --provider config

# --n-episodes repeats/reshuffles a family beyond its variant count within a
# single seed's persistent store, so A2 gets enough recurring failures to
# actually cluster -> diagnose -> induce -> validate -> promote a policy:
python run.py --family bug_reproduction --provider config --n-episodes 28 \
    --checkpoint-every 7 --max-steps 3
```

## Model routing (free / open only)

Two roles in `config/models.yaml`, routed independently to dodge the free-tier
rate-limit vs long-run tension:

- **online** (the 100–1000-episode agent loop) → a **local** open model
  (Ollama / vLLM): no rate limits.
- **offline** (bursty consolidation) → a **free-tier hosted** model (Groq,
  OpenRouter, …) works well here, but the shipped default points both roles
  at the same local Ollama instance so the whole system runs with zero
  network calls and zero API keys out of the box.

Every backend speaks the OpenAI `/v1/chat/completions` shape, so switching is a
config change, not a code change.

## Phase 2: Causal Diagnosis and Pattern Mining (implemented)

Phase 2 upgrades the learning loop with structured failure analysis:

- **Failure taxonomy** (`persistence/a2_engine/taxonomy.py`): classifies every
  failure into typed categories (wrong_output, runtime_error, timeout,
  missing_action, etc.) so diagnosis is targeted, not one-size-fits-all.
- **Upgraded causal diagnoser** (`diagnoser.py`): multi-step trace analysis,
  critical decision point detection, cross-episode agreement scoring, structured
  `CausalChain` output. Falls back to heuristics when no LLM provider is available.
- **Pattern miner** (`pattern_miner.py`): feature-based agglomerative clustering
  beyond the Phase 1 signature-only approach. Finds structural similarity across
  failure types, action sequences, and observations. Also detects cross-cluster
  patterns (shared root causes that manifest differently on the surface).
- **Contradiction miner** (`contradiction.py`): scans active experiences for
  conflicting recommendations. Detects opposition via keyword-pair heuristics,
  penalizes confidence, and suggests resolution strategies. Contradictions refine
  scope — they don't delete experiences.
- **Consolidation stats** (`ConsolidationStats`): full observability on what
  happened during each offline pass (clusters found, induced, promoted, rejected,
  reinforced, contradictions, cross-patterns).

## Phase 3: Experience Graph (implemented)

Phase 3 adds a graph store and hybrid multi-signal retrieval:

- **Experience Graph** (`persistence/graph.py`): lightweight in-process typed
  graph with nodes (episode, experience, policy, task_family, failure_mode) and
  edges (supports, contradicts, reinforces, similar_to, transfers_to, promoted_to,
  etc.). Supports BFS traversal, subgraph extraction, evidence/provenance queries,
  and JSON serialization.
- **Graph Builder** (`persistence/graph_builder.py`): constructs the graph from
  ExperienceStore contents — episodes become nodes, experiences link to their
  source episodes, policies link to their supporting experiences, cross-family
  lessons get `transfers_to` edges, and similar experiences get `similar_to` edges.
  Supports both full rebuild and incremental updates.
- **Hybrid Retriever** (`persistence/hybrid_retriever.py`): multi-signal ranked
  retrieval combining semantic similarity, confidence weighting, evidence strength,
  graph proximity (BFS from family/context anchors), and recency. Replaces the
  Phase 1 simple cosine search in the A2 engine's `retrieve()` method.
- **Engine integration**: the A2 engine now builds and maintains the graph
  incrementally during `record()` and `consolidate()`, and uses the hybrid
  retriever for online context injection.

## What's deliberately NOT here yet (post go/no-go)

Skill compiler, forgetting, full governance,
exploration, weight-internalization. Building them before the core loop is shown
to compound is spending ahead of the bet. See the proposal's phase plan.

## Tests worth knowing about

- `test_harness_smoke.py::test_toy_env_execution_grounded_reward` — reward comes
  from real pytest.
- `test_poisoned_experience.py` — the **non-negotiable gate**: a harmful lesson
  must be rejected, never promoted.
- `test_a2_consolidation.py` — the cluster→diagnose→induce→validate→policy loop.
- `test_taxonomy.py` — Phase 2: failure classification into typed categories.
- `test_pattern_miner.py` — Phase 2: feature-based clustering and cross-cluster
  pattern detection.
- `test_contradiction.py` — Phase 2: detecting and handling conflicting experiences.
- `test_diagnoser_phase2.py` — Phase 2: structured causal chains, taxonomy-aware
  diagnosis, contributing factors.
- `test_graph.py` — Phase 3: graph nodes, edges, BFS traversal, subgraph,
  serialization roundtrip.
- `test_graph_builder.py` — Phase 3: automatic graph construction from store,
  similarity/transfer edges, incremental updates.
- `test_hybrid_retriever.py` — Phase 3: multi-signal ranked retrieval, context
  filtering, graph proximity boosting.
```
