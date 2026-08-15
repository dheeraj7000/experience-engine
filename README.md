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

## Phase 4: Skill Compilation (implemented)

Phase 4 extracts reusable procedures from repeated successes:

- **Skill schema** (`schemas/skill.py`): `SkillObject` with preconditions,
  ordered workflow steps, postconditions, evidence tracking, versioning, and
  validation lifecycle. `as_instruction()` formats for injection.
- **Skill Compiler** (`persistence/a2_engine/skill_compiler.py`): induces skills
  from clusters of successful episodes sharing a common action sequence. Validates
  via held-out replay, reinforces with new evidence, deduplicates against existing
  skills.
- **Engine integration**: consolidation now compiles skills from successes
  (alongside learning from failures). Active skills are injected during
  `retrieve()` — the agent gets both lessons (what went wrong) and procedures
  (what works).

## Phase 5: Policy Management (implemented)

Phase 5 upgrades the policy lifecycle to production-grade:

- **Conflict detection** (`PolicyManager.detect_conflicts()`): finds overlapping
  active policies (same scope + shared triggers) and resolves by priority/confidence.
- **Scope refinement** (`refine_scope()`): narrows a policy's applicability by
  adding conditions, creating a new version (old one superseded, not deleted).
- **Versioning** (`supersede()`): deprecated policies remain for audit; new
  versions inherit evidence chains.
- **Rollback** (`rollback()`): deactivate a harmful policy without data loss.
- **Safety gating**: safety-critical scopes (security, auth, deletion, production)
  require higher confidence thresholds for promotion.
- **Skill routing**: policies can reference compiled skills (`promote_from_skill()`),
  creating a full chain: episodes → experience → skill → policy.
- **Priority ordering** (`ordered_policies()`): deterministic resolution when
  multiple policies apply.

## Phase 6: Forgetting & Governance (implemented)

Phase 6 ensures the system doesn't accumulate infinite stale rules and provides
safety/audit controls:

- **Forgetting Manager** (`persistence/a2_engine/forgetting.py`): time-based
  confidence decay with evidence-weighted retention. More evidence = slower decay.
  More contradictions = faster decay. Items archived (not deleted) when confidence
  drops below threshold. Staleness detection and revalidation scheduling.
  Reinforcement resets the decay clock.
- **Governance Layer** (`persistence/a2_engine/governance.py`): audit log for
  every learning action (induction, promotion, decay, rollback). Risk assessment
  (low/medium/high/critical) based on scope and content. Sensitive content
  detection and redaction. Do-not-learn zones. Human review queue for high-risk
  changes. Safety gating blocks experiences containing credentials/secrets.
- **Engine integration**: decay runs each consolidation cycle. Governance blocks
  sensitive experiences before promotion. Audit log tracks all learning actions.

## Phase 7: Exploration & Internalization (implemented)

Phase 7 adds autonomous exploration guidance and internalization candidate scoring:

- **Exploration Manager** (`persistence/a2_engine/exploration.py`): profiles each
  task family (success rate, variant coverage, strategy diversity). Computes
  exploration scores and generates prioritized suggestions (practice, try
  alternative, expand variants). Measures strategy diversity via normalized
  entropy to detect exploration collapse. Tracks exploitation ratio.
- **Experience Internalizer** (`persistence/a2_engine/internalizer.py`): scores
  experiences and skills for internalization-worthiness using 5 signals (evidence,
  confidence, stability, transfer utility, safety). Strict eligibility thresholds
  (min evidence, min confidence, max contradictions, safety check). Produces
  ranked candidate lists for human review — does NOT perform automatic weight
  updates.

## All phases complete

The full Experience Engine architecture is implemented (Phases 1–7). The system
is ready for real evaluation with a live model to answer the central question:
does A2 compound better than A1?

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
- `test_skill_compiler.py` — Phase 4: workflow induction from successes,
  validation, reinforcement, deduplication.
- `test_policy_manager.py` — Phase 5: conflict detection/resolution, scope
  refinement, versioning, rollback, safety gating, skill routing.
- `test_forgetting.py` — Phase 6: confidence decay, evidence-weighted retention,
  reinforcement, archival, staleness detection.
- `test_governance.py` — Phase 6: audit logging, risk assessment, sensitive
  content blocking, do-not-learn zones, human review queue.
- `test_exploration.py` — Phase 7: family profiling, diversity scoring,
  exploration suggestions, exploitation ratio.
- `test_internalizer.py` — Phase 7: internalization-worthiness scoring,
  eligibility thresholds, safety checks, candidate ranking.
```
