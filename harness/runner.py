"""Runner — executes (config x family x seed), checkpoints the store, and
collects per-episode metrics. Supplies the A2 validator its held-out replay
function, so validation and contamination control share one held-out set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from providers import ModelProvider
from agent import AgentController
from persistence import ExperienceStore, build_persistence
from schemas import Episode
from schemas.reward import RewardVector
from .environment import Variant
from .sequencer import Sequencer
from .task_family import TaskFamily


@dataclass
class EpisodeRecord:
    index: int
    config: str
    seed: int
    variant_id: str
    success: float
    tokens: int
    signature: str


@dataclass
class RunConfig:
    configs: list[str] = field(default_factory=lambda: ["a0", "a1", "a2"])
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    checkpoint_every: int = 5
    max_steps: int = 4
    n_episodes: int | None = None
    out_dir: str = "runs"


class Runner:
    def __init__(self, online: ModelProvider, offline: ModelProvider | None = None) -> None:
        self.online = online
        self.offline = offline
        self.controller = AgentController(online)
        self.seq = Sequencer()

    def run(self, family: TaskFamily, cfg: RunConfig) -> list[EpisodeRecord]:
        self.controller.max_steps = cfg.max_steps
        records: list[EpisodeRecord] = []
        for config in cfg.configs:
            for seed in cfg.seeds:
                records.extend(self._run_one(family, config, seed, cfg))
        return records

    def _run_one(self, family: TaskFamily, config: str, seed: int,
                 cfg: RunConfig) -> list[EpisodeRecord]:
        snap_root = Path(cfg.out_dir) / f"{family.family_id}" / f"{config}_seed{seed}"
        store = ExperienceStore(root=snap_root)
        persistence = build_persistence(config, store, offline_provider=self.offline)
        replay_fn = self._make_replay_fn(family, seed)

        variants = self.seq.order(family, seed, cfg.n_episodes)
        out: list[EpisodeRecord] = []
        for idx, variant in enumerate(variants):
            episode = self._play(family, variant, persistence, seed)
            persistence.record(episode)
            if (idx + 1) % cfg.checkpoint_every == 0:
                persistence.consolidate(replay_fn=replay_fn)
                store.snapshot(snap_root / f"checkpoint_{idx + 1}.json")
            out.append(EpisodeRecord(
                index=idx, config=config, seed=seed, variant_id=variant.variant_id,
                success=episode.outcome.task_success if episode.outcome else 0.0,
                tokens=episode.cost.tokens,
                signature=episode.failure_signature(),
            ))
        # Final consolidation + snapshot.
        persistence.consolidate(replay_fn=replay_fn)
        store.snapshot(snap_root / "final.json")
        return out

    def _play(self, family: TaskFamily, variant: Variant, persistence, seed: int) -> Episode:
        env = family.make_env()
        context = {"family": variant.family, "goal": variant.goal, **variant.spec}
        injected = persistence.retrieve(context)
        episode = self.controller.run(env, variant, injected, seed=seed)
        episode.outcome = env.grade().compute_overall()
        return episode

    def _make_replay_fn(self, family: TaskFamily, seed: int):
        """Return replay_fn(experience) -> (baseline_success, with_exp_success)
        over held-out variants. Baseline injects nothing; the with-exp arm
        injects the candidate lesson. These held-out episodes are never
        recorded into the store (contamination control)."""
        heldout = self.seq.heldout(family)

        def replay_fn(experience) -> tuple[float, float]:
            if not heldout:
                return (0.0, 0.0)
            base = self._avg_success(family, heldout, seed, injected="")
            withx = self._avg_success(family, heldout, seed, injected=experience.lesson)
            return (base, withx)

        return replay_fn

    def _avg_success(self, family: TaskFamily, variants: list[Variant],
                     seed: int, injected: str) -> float:
        scores = []
        for v in variants:
            env = family.make_env()
            ep = self.controller.run(env, v, injected, seed=seed)
            rv: RewardVector = env.grade()
            scores.append(rv.task_success)
        return sum(scores) / len(scores) if scores else 0.0
