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
persistence/      base · store (jsonl+vector) · a0 · a1 · a2_engine/ (the MVES loop)
harness/          environment · task_family · sequencer · runner · reporter
families/         toy_bug (executable) + qa_families specs ("all QA")
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

# Real learning curves need a real online model. Point config/models.yaml at a
# local Ollama/vLLM endpoint, then:
python run.py --provider config
```

## Model routing (free / open only)

Two roles in `config/models.yaml`, routed independently to dodge the free-tier
rate-limit vs long-run tension:

- **online** (the 100–1000-episode agent loop) → a **local** open model
  (Ollama / vLLM): no rate limits.
- **offline** (bursty consolidation) → a **free-tier hosted** model (Groq,
  OpenRouter, …) is fine.

Every backend speaks the OpenAI `/v1/chat/completions` shape, so switching is a
config change, not a code change.

## What's deliberately NOT here yet (post go/no-go)

Skill compiler, graph store, contradiction mining, forgetting, full governance,
exploration, weight-internalization. Building them before the core loop is shown
to compound is spending ahead of the bet. See the proposal's phase plan.

## Tests worth knowing about

- `test_harness_smoke.py::test_toy_env_execution_grounded_reward` — reward comes
  from real pytest.
- `test_poisoned_experience.py` — the **non-negotiable gate**: a harmful lesson
  must be rejected, never promoted.
- `test_a2_consolidation.py` — the cluster→diagnose→induce→validate→policy loop.
```
