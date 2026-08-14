"""A2 Experience Engine — Phase 3 with Experience Graph + Hybrid Retrieval.

retrieve  : inject validated experiences + active policies via hybrid retriever
record    : store the structured episode + update graph
consolidate: cluster -> diagnose -> induce -> VALIDATE -> promote to policy
             + contradiction detection + pattern mining + graph update

Phase 3 additions over Phase 2:
  - ExperienceGraph: typed nodes and edges representing relationships
  - GraphBuilder: automatic graph construction from store contents
  - HybridRetriever: multi-signal ranking (semantic + confidence + evidence +
    graph proximity + recency) replaces the simple cosine search
  - Graph is updated incrementally during record() and consolidate()
  - Graph serialization in snapshots

Consolidation runs OFFLINE (between checkpoints), so online latency stays
comparable to A1 and the overhead metric is honest.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from schemas import Episode
from schemas.experience import ValidationStatus
from ..graph import ExperienceGraph, EdgeType, NodeType
from ..graph_builder import GraphBuilder
from ..hybrid_retriever import HybridRetriever
from ..store import ExperienceStore
from .contradiction import ContradictionMiner
from .diagnoser import CausalDiagnoser
from .inducer import ExperienceInducer
from .pattern_miner import PatternMiner
from .skill_compiler import SkillCompiler
from .taxonomy import classify_failure
from .validator import ExperienceValidator
from .policy import PolicyManager

MIN_CLUSTER = 3  # episodes with a shared signature before we induce a lesson


class ConsolidationStats:
    """Track what happened during a consolidation pass (observability)."""

    def __init__(self) -> None:
        self.clusters_found: int = 0
        self.clusters_below_threshold: int = 0
        self.experiences_induced: int = 0
        self.experiences_promoted: int = 0
        self.experiences_rejected: int = 0
        self.experiences_reinforced: int = 0
        self.contradictions_found: int = 0
        self.cross_patterns_found: int = 0
        self.skills_compiled: int = 0
        self.skills_validated: int = 0
        self.policy_conflicts_found: int = 0
        self.policy_conflicts_resolved: int = 0

    def summary(self) -> dict[str, int]:
        return {
            "clusters_found": self.clusters_found,
            "clusters_below_threshold": self.clusters_below_threshold,
            "experiences_induced": self.experiences_induced,
            "experiences_promoted": self.experiences_promoted,
            "experiences_rejected": self.experiences_rejected,
            "experiences_reinforced": self.experiences_reinforced,
            "contradictions_found": self.contradictions_found,
            "cross_patterns_found": self.cross_patterns_found,
            "skills_compiled": self.skills_compiled,
            "skills_validated": self.skills_validated,
            "policy_conflicts_found": self.policy_conflicts_found,
            "policy_conflicts_resolved": self.policy_conflicts_resolved,
        }


class ExperienceEngine:
    name = "a2"

    def __init__(self, store: ExperienceStore, offline_provider=None,
                 min_confidence: float = 0.4, use_graph: bool = True) -> None:
        self.store = store
        self.min_confidence = min_confidence
        self.diagnoser = CausalDiagnoser(provider=offline_provider)
        self.inducer = ExperienceInducer()
        self.validator = ExperienceValidator()
        self.policy_mgr = PolicyManager()
        self.pattern_miner = PatternMiner(min_cluster_size=MIN_CLUSTER)
        self.contradiction_miner = ContradictionMiner()
        self.skill_compiler = SkillCompiler()
        self._consolidated_ids: set[str] = set()
        self.last_stats: ConsolidationStats | None = None

        # Phase 3: Graph + hybrid retrieval.
        self.use_graph = use_graph
        self.graph = ExperienceGraph() if use_graph else None
        self._graph_builder = GraphBuilder() if use_graph else None
        self._retriever: HybridRetriever | None = None
        if use_graph:
            self._retriever = HybridRetriever(
                store, self.graph, min_confidence=min_confidence)

    # ---- online -----------------------------------------------------------
    def retrieve(self, context: dict[str, Any]) -> str:
        """Phase 3: use hybrid retriever if graph is available, else fall back."""
        if self._retriever and self.graph and self.graph.node_count > 0:
            text = self._retriever.retrieve_text(context)
            # Phase 4: also inject applicable skills.
            skill_text = self._retrieve_skills(context)
            if skill_text:
                text = (text + "\n" + skill_text) if text else skill_text
            return text

        # Fallback: Phase 1/2 simple retrieval.
        query = context.get("goal", "") + " " + context.get("family", "")
        parts: list[str] = []

        policies = self.store.active_policies(context.get("family", ""))
        for p in policies[:3]:
            parts.append(f"- Policy: {p.as_directive()}")

        exps = self.store.search_experiences(
            context, query, k=3, min_confidence=self.min_confidence)
        for e in exps:
            parts.append(f"- Experience (conf {e.confidence:.2f}): {e.lesson}")

        # Phase 4: inject applicable skills.
        skill_text = self._retrieve_skills(context)
        if skill_text:
            parts.append(skill_text)

        return "\n".join(parts)

    def _retrieve_skills(self, context: dict[str, Any]) -> str:
        """Retrieve applicable active skills for injection."""
        family = context.get("family", "")
        skills = self.store.active_skills(family)
        if not skills:
            return ""
        # Inject top 2 skills max to avoid prompt bloat.
        lines = []
        for s in skills[:2]:
            lines.append(f"- Skill: {s.as_instruction()}")
        return "\n".join(lines)

    def record(self, episode: Episode) -> None:
        self.store.add_episode(episode)
        # Phase 3: update graph incrementally.
        if self._graph_builder and self.graph:
            self._graph_builder.update_incremental(
                self.graph, self.store, new_episodes=[episode])

    # ---- offline (the learning) ------------------------------------------
    def consolidate(self, replay_fn=None) -> ConsolidationStats:
        """Phase 2 consolidation: pattern mining + diagnosis + induction +
        validation + contradiction detection."""
        stats = ConsolidationStats()

        # Step 1: Cluster new failures (Phase 1 signature-based for backward compat).
        clusters = self._cluster_new_failures()
        stats.clusters_found = len(clusters)

        # Step 2: Process each cluster.
        for signature, eps in clusters.items():
            if len(eps) < MIN_CLUSTER:
                stats.clusters_below_threshold += 1
                continue
            context = self._shared_context(eps)
            diagnosis = self.diagnoser.diagnose(eps)
            candidate = self.inducer.induce(eps, diagnosis, context)
            if candidate.confidence < self.min_confidence:
                stats.clusters_below_threshold += 1
                continue

            existing = self.store.find_mergeable(candidate.task_family,
                                                candidate.context_conditions)
            if existing is not None:
                self._reinforce(existing, eps)
                stats.experiences_reinforced += 1
                for e in eps:
                    self._consolidated_ids.add(e.episode_id)
                continue

            # VALIDATE before it can influence behavior.
            promoted = self.validator.validate(candidate, replay_fn)
            self.store.add_experience(candidate)
            stats.experiences_induced += 1
            # Phase 3: update graph with new experience.
            if self._graph_builder and self.graph:
                self._graph_builder.update_incremental(
                    self.graph, self.store, new_experiences=[candidate])
            if promoted and candidate.validation_status == ValidationStatus.active:
                policy = self.policy_mgr.promote(candidate)
                self.store.add_policy(policy)
                stats.experiences_promoted += 1
                # Phase 3: update graph with new policy.
                if self._graph_builder and self.graph:
                    self._graph_builder.update_incremental(
                        self.graph, self.store, new_policies=[policy])
            else:
                stats.experiences_rejected += 1
            for e in eps:
                self._consolidated_ids.add(e.episode_id)

        # Step 3 (Phase 2): Contradiction detection across all active experiences.
        if self.store.experiences:
            contradictions = self.contradiction_miner.detect(self.store.experiences)
            stats.contradictions_found = len(contradictions)
            if contradictions:
                self.contradiction_miner.apply_contradictions(
                    self.store.experiences, contradictions)
                # Phase 3: add contradicts edges to graph.
                if self.graph:
                    for c in contradictions:
                        self.graph.add_edge(
                            c.experience_a_id, c.experience_b_id,
                            EdgeType.contradicts, weight=c.severity)

        # Step 4 (Phase 2): Cross-cluster pattern detection (informational).
        new_failures = [e for e in self.store.episodes
                        if not (e.outcome and e.outcome.task_success >= 1.0)]
        if len(new_failures) >= MIN_CLUSTER * 2:
            feature_clusters = self.pattern_miner.cluster_by_features(new_failures)
            cross_patterns = self.pattern_miner.find_cross_cluster_patterns(feature_clusters)
            stats.cross_patterns_found = len(cross_patterns)

        # Step 5 (Phase 4): Skill compilation from successful episodes.
        new_skills = self.skill_compiler.compile_skills(
            self.store.episodes, existing_skills=self.store.skills)
        for skill in new_skills:
            # Validate skill if replay is available.
            if replay_fn:
                validated = self.skill_compiler.validate_skill(skill, replay_fn)
                if validated:
                    stats.skills_validated += 1
            self.store.add_skill(skill)
            stats.skills_compiled += 1

        # Step 6 (Phase 5): Policy conflict detection and resolution.
        if self.store.policies:
            conflicts = self.policy_mgr.detect_conflicts(self.store.policies)
            stats.policy_conflicts_found = len(conflicts)
            if conflicts:
                actions = self.policy_mgr.resolve_conflicts(
                    conflicts, self.store.policies)
                stats.policy_conflicts_resolved = len(actions)

        self.last_stats = stats
        return stats

    @staticmethod
    def _reinforce(existing, new_eps: list[Episode]) -> None:
        existing.evidence_count += len(new_eps)
        existing.source_episodes.extend(e.episode_id for e in new_eps)
        existing.confidence = min(0.95, existing.confidence + 0.05)

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
