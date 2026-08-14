"""Graph Builder (Phase 3) — constructs the ExperienceGraph from store contents.

The builder creates nodes and edges from the existing ExperienceStore data:
  - Episodes become episode nodes
  - Experiences become experience nodes with edges to their source episodes
  - Policies become policy nodes linked to their supporting experiences
  - Task families become family nodes
  - Contradiction relationships from Phase 2 become contradicts edges
  - Cross-family experiences get transfers_to edges
  - Similar experiences (by lesson overlap) get similar_to edges

The builder can be called incrementally (only process new items) or from scratch.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from schemas import Episode, ExperienceObject, PolicyObject
from schemas.experience import ValidationStatus
from .graph import (
    ExperienceGraph, NodeType, EdgeType,
)
from .store import ExperienceStore

_WORD = re.compile(r"[a-z0-9_]+")


def _bow(text: str) -> Counter:
    return Counter(_WORD.findall(text.lower()))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class GraphBuilder:
    """Builds and maintains the ExperienceGraph from store contents."""

    def __init__(self, similarity_threshold: float = 0.5) -> None:
        self.similarity_threshold = similarity_threshold

    def build_full(self, store: ExperienceStore) -> ExperienceGraph:
        """Build the complete graph from scratch."""
        graph = ExperienceGraph()
        self._add_episodes(graph, store.episodes)
        self._add_experiences(graph, store.experiences)
        self._add_policies(graph, store.policies)
        self._add_family_nodes(graph, store)
        self._add_similarity_edges(graph, store.experiences)
        self._add_transfer_edges(graph, store.experiences)
        return graph

    def update_incremental(self, graph: ExperienceGraph, store: ExperienceStore,
                           new_episodes: list[Episode] | None = None,
                           new_experiences: list[ExperienceObject] | None = None,
                           new_policies: list[PolicyObject] | None = None) -> None:
        """Add new items to an existing graph without rebuilding."""
        if new_episodes:
            self._add_episodes(graph, new_episodes)
        if new_experiences:
            self._add_experiences(graph, new_experiences)
            self._add_similarity_edges(graph, store.experiences)
            self._add_transfer_edges(graph, store.experiences)
        if new_policies:
            self._add_policies(graph, new_policies)

    def _add_episodes(self, graph: ExperienceGraph, episodes: list[Episode]) -> None:
        """Add episode nodes + belongs_to edges."""
        for ep in episodes:
            success = bool(ep.outcome and ep.outcome.task_success >= 1.0)
            graph.add_node(ep.episode_id, NodeType.episode, metadata={
                "task_family": ep.task_family,
                "variant_id": ep.task_variant_id,
                "goal": ep.goal[:100],
                "success": success,
                "tokens": ep.cost.tokens,
            })
            # Edge to task_family node.
            if ep.task_family:
                family_node = f"family:{ep.task_family}"
                graph.add_node(family_node, NodeType.task_family,
                               metadata={"family_id": ep.task_family})
                graph.add_edge(ep.episode_id, family_node, EdgeType.belongs_to)

    def _add_experiences(self, graph: ExperienceGraph,
                         experiences: list[ExperienceObject]) -> None:
        """Add experience nodes + supports/reinforces edges to source episodes."""
        for exp in experiences:
            graph.add_node(exp.experience_id, NodeType.experience, metadata={
                "task_family": exp.task_family,
                "lesson": exp.lesson[:200],
                "confidence": exp.confidence,
                "evidence_count": exp.evidence_count,
                "validation_status": exp.validation_status.value,
                "contradictions": exp.contradictions,
            })
            # Edges from source episodes.
            for i, ep_id in enumerate(exp.source_episodes):
                edge_type = EdgeType.supports if i < 3 else EdgeType.reinforces
                if graph.get_node(ep_id):
                    graph.add_edge(ep_id, exp.experience_id, edge_type,
                                   weight=1.0 / (i + 1))

            # Family edge.
            if exp.task_family:
                family_node = f"family:{exp.task_family}"
                graph.add_node(family_node, NodeType.task_family,
                               metadata={"family_id": exp.task_family})
                graph.add_edge(exp.experience_id, family_node, EdgeType.belongs_to)

    def _add_policies(self, graph: ExperienceGraph,
                      policies: list[PolicyObject]) -> None:
        """Add policy nodes + promoted_to edges from supporting experiences."""
        for pol in policies:
            graph.add_node(pol.policy_id, NodeType.policy, metadata={
                "scope": pol.scope,
                "behavior": pol.behavior[:200],
                "priority": pol.priority,
                "confidence": pol.confidence,
                "validation_status": pol.validation_status.value,
            })
            for exp_id in pol.supporting_experiences:
                if graph.get_node(exp_id):
                    graph.add_edge(exp_id, pol.policy_id, EdgeType.promoted_to)

    def _add_family_nodes(self, graph: ExperienceGraph,
                          store: ExperienceStore) -> None:
        """Ensure all referenced task families have nodes."""
        families: set[str] = set()
        for ep in store.episodes:
            if ep.task_family:
                families.add(ep.task_family)
        for exp in store.experiences:
            if exp.task_family:
                families.add(exp.task_family)
        for fam in families:
            graph.add_node(f"family:{fam}", NodeType.task_family,
                           metadata={"family_id": fam})

    def _add_similarity_edges(self, graph: ExperienceGraph,
                              experiences: list[ExperienceObject]) -> None:
        """Add similar_to edges between experiences with similar lessons."""
        for i, a in enumerate(experiences):
            bow_a = _bow(a.lesson + " " + a.root_cause)
            for b in experiences[i + 1:]:
                # Skip if different validation states (rejected vs active is irrelevant).
                if (a.validation_status == ValidationStatus.rejected or
                        b.validation_status == ValidationStatus.rejected):
                    continue
                bow_b = _bow(b.lesson + " " + b.root_cause)
                sim = _cosine(bow_a, bow_b)
                if sim >= self.similarity_threshold:
                    graph.add_edge(a.experience_id, b.experience_id,
                                   EdgeType.similar_to, weight=round(sim, 3))

    def _add_transfer_edges(self, graph: ExperienceGraph,
                            experiences: list[ExperienceObject]) -> None:
        """Add transfers_to edges between experiences in different families
        that share similar lessons (cross-domain transfer)."""
        for i, a in enumerate(experiences):
            if not a.task_family:
                continue
            bow_a = _bow(a.lesson + " " + a.root_cause)
            for b in experiences[i + 1:]:
                if not b.task_family or b.task_family == a.task_family:
                    continue
                if (a.validation_status == ValidationStatus.rejected or
                        b.validation_status == ValidationStatus.rejected):
                    continue
                bow_b = _bow(b.lesson + " " + b.root_cause)
                sim = _cosine(bow_a, bow_b)
                if sim >= self.similarity_threshold:
                    graph.add_edge(a.experience_id, b.experience_id,
                                   EdgeType.transfers_to, weight=round(sim, 3))
