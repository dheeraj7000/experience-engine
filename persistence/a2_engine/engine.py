"""A2 Experience Engine — the MVES closed loop.

retrieve  : inject validated experiences + active policies (scoped, confident)
record    : store the structured episode
consolidate: cluster -> diagnose -> induce -> VALIDATE -> promote to policy

Consolidation runs OFFLINE (between checkpoints), so online latency stays
comparable to A1 and the overhead metric is honest.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from schemas import Episode
from schemas.experience import ValidationStatus
from ..store import ExperienceStore
from .diagnoser import CausalDiagnoser
from .inducer import ExperienceInducer
from .validator import ExperienceValidator
from .policy import PolicyManager

MIN_CLUSTER = 3  # episodes with a shared signature before we induce a lesson


class ExperienceEngine:
    name = "a2"

    def __init__(self, store: ExperienceStore, offline_provider=None,
                 min_confidence: float = 0.4) -> None:
        self.store = store
        self.min_confidence = min_confidence
        self.diagnoser = CausalDiagnoser(provider=offline_provider)
        self.inducer = ExperienceInducer()
        self.validator = ExperienceValidator()
        self.policy_mgr = PolicyManager()
        self._consolidated_ids: set[str] = set()

    # ---- online -----------------------------------------------------------
    def retrieve(self, context: dict[str, Any]) -> str:
        query = context.get("goal", "") + " " + context.get("family", "")
        parts: list[str] = []

        policies = self.store.active_policies(context.get("family", ""))
        for p in policies[:3]:
            parts.append(f"- Policy: {p.as_directive()}")

        exps = self.store.search_experiences(
            context, query, k=3, min_confidence=self.min_confidence)
        for e in exps:
            parts.append(f"- Experience (conf {e.confidence:.2f}): {e.lesson}")

        return "\n".join(parts)

    def record(self, episode: Episode) -> None:
        self.store.add_episode(episode)

    # ---- offline (the learning) ------------------------------------------
    def consolidate(self, replay_fn=None) -> None:
        clusters = self._cluster_new_failures()
        for signature, eps in clusters.items():
            if len(eps) < MIN_CLUSTER:
                continue
            context = self._shared_context(eps)
            diagnosis = self.diagnoser.diagnose(eps)
            candidate = self.inducer.induce(eps, diagnosis, context)
            if candidate.confidence < self.min_confidence:
                continue
            # VALIDATE before it can influence behavior.
            promoted = self.validator.validate(candidate, replay_fn)
            self.store.add_experience(candidate)
            if promoted and candidate.validation_status == ValidationStatus.active:
                self.store.add_policy(self.policy_mgr.promote(candidate))
            for e in eps:
                self._consolidated_ids.add(e.episode_id)

    # ---- helpers ----------------------------------------------------------
    def _cluster_new_failures(self) -> dict[str, list[Episode]]:
        clusters: dict[str, list[Episode]] = defaultdict(list)
        for e in self.store.episodes:
            if e.episode_id in self._consolidated_ids:
                continue
            if e.outcome and e.outcome.task_success >= 1.0:
                continue  # cluster failures; success patterns -> Phase 3 skills
            clusters[e.failure_signature()].append(e)
        return clusters

    @staticmethod
    def _shared_context(eps: list[Episode]) -> dict[str, Any]:
        ctx: dict[str, Any] = {"family": eps[0].task_family}
        # Carry through any spec keys shared identically across the cluster.
        common_keys = set(eps[0].initial_state)
        for e in eps[1:]:
            common_keys &= set(e.initial_state)
        for k in common_keys:
            vals = {str(e.initial_state.get(k)) for e in eps}
            if len(vals) == 1:
                ctx[k] = eps[0].initial_state[k]
        return ctx
