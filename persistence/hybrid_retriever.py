"""Hybrid Retriever (Phase 3, proposal 15.2).

Combines multiple retrieval signals for higher-quality experience injection:
  1. Semantic similarity (bag-of-words cosine, from Phase 1 store.search_*)
  2. Structured metadata filters (task_family, failure_type, confidence)
  3. Graph traversal (related experiences within N hops)
  4. Utility-aware ranking (how much did this experience improve outcomes?)
  5. Recency weighting (recent evidence is more relevant)
  6. Confidence weighting (higher confidence = higher rank)

The retriever produces a ranked list of experiences + policies to inject,
respecting a token budget to avoid prompt bloat.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from schemas import ExperienceObject, PolicyObject
from schemas.experience import ValidationStatus
from .graph import ExperienceGraph, EdgeType, NodeType
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


@dataclass
class RetrievalResult:
    """A scored retrieval hit."""
    item_id: str
    item_type: str  # "experience" or "policy"
    text: str       # the text to inject
    score: float    # combined ranking score [0, 1]
    signals: dict[str, float] = field(default_factory=dict)  # breakdown


class HybridRetriever:
    """Multi-signal retrieval combining semantic, structural, and graph signals.

    Weights control the relative importance of each signal:
      semantic:   raw text similarity to the query
      confidence: the experience's self-assessed confidence
      evidence:   how much evidence backs it
      graph:      graph proximity to relevant nodes
      recency:    preference for recently-validated experiences
    """

    def __init__(
        self,
        store: ExperienceStore,
        graph: ExperienceGraph | None = None,
        *,
        w_semantic: float = 0.35,
        w_confidence: float = 0.25,
        w_evidence: float = 0.15,
        w_graph: float = 0.15,
        w_recency: float = 0.10,
        min_confidence: float = 0.3,
        max_results: int = 5,
    ) -> None:
        self.store = store
        self.graph = graph
        self.w_semantic = w_semantic
        self.w_confidence = w_confidence
        self.w_evidence = w_evidence
        self.w_graph = w_graph
        self.w_recency = w_recency
        self.min_confidence = min_confidence
        self.max_results = max_results

    def retrieve(self, context: dict[str, Any], query: str = "",
                 max_results: int | None = None) -> list[RetrievalResult]:
        """Retrieve ranked experiences + policies for the given context."""
        k = max_results or self.max_results
        results: list[RetrievalResult] = []

        # Score experiences.
        query_text = query or (context.get("goal", "") + " " + context.get("family", ""))
        query_bow = _bow(query_text)

        candidates = self._filter_candidates(context)
        graph_scores = self._graph_scores(context) if self.graph else {}

        for exp in candidates:
            signals = self._score_experience(exp, query_bow, graph_scores)
            combined = self._combine_signals(signals)
            results.append(RetrievalResult(
                item_id=exp.experience_id,
                item_type="experience",
                text=f"Experience (conf {exp.confidence:.2f}): {exp.lesson}",
                score=combined,
                signals=signals,
            ))

        # Score policies.
        policies = self.store.active_policies(context.get("family", ""))
        for pol in policies:
            signals = self._score_policy(pol, query_bow, graph_scores)
            combined = self._combine_signals(signals)
            results.append(RetrievalResult(
                item_id=pol.policy_id,
                item_type="policy",
                text=f"Policy: {pol.as_directive()}",
                score=combined,
                signals=signals,
            ))

        # Rank and return top-k.
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def retrieve_text(self, context: dict[str, Any], query: str = "",
                      max_results: int | None = None) -> str:
        """Convenience: return the injection text string (matches engine.retrieve interface)."""
        results = self.retrieve(context, query, max_results)
        if not results:
            return ""
        return "\n".join(f"- {r.text}" for r in results)

    # ---- filtering --------------------------------------------------------
    def _filter_candidates(self, context: dict[str, Any]) -> list[ExperienceObject]:
        """Pre-filter experiences by metadata before scoring."""
        active = [
            e for e in self.store.experiences
            if e.validation_status in (ValidationStatus.active, ValidationStatus.validated)
            and e.confidence >= self.min_confidence
            and e.matches(context)
        ]
        return active

    # ---- scoring ----------------------------------------------------------
    def _score_experience(self, exp: ExperienceObject, query_bow: Counter,
                          graph_scores: dict[str, float]) -> dict[str, float]:
        """Compute per-signal scores for an experience."""
        # Semantic similarity.
        exp_bow = _bow(exp.lesson + " " + exp.root_cause)
        semantic = _cosine(query_bow, exp_bow)

        # Confidence signal.
        confidence = exp.confidence

        # Evidence strength (diminishing returns on large counts).
        evidence = 1.0 - (1.0 / (1.0 + exp.evidence_count))

        # Graph proximity (if graph is available).
        graph = graph_scores.get(exp.experience_id, 0.0)

        # Recency: no timestamp in Phase 1/2 schema, so proxy with evidence freshness.
        # More recent = more source_episodes. This is a weak signal, improved in
        # later phases when timestamps are added.
        recency = min(1.0, len(exp.source_episodes) / 10.0)

        return {
            "semantic": round(semantic, 4),
            "confidence": round(confidence, 4),
            "evidence": round(evidence, 4),
            "graph": round(graph, 4),
            "recency": round(recency, 4),
        }

    def _score_policy(self, pol: PolicyObject, query_bow: Counter,
                      graph_scores: dict[str, float]) -> dict[str, float]:
        """Score a policy using available signals."""
        pol_bow = _bow(pol.behavior + " " + pol.scope)
        semantic = _cosine(query_bow, pol_bow)
        confidence = pol.confidence
        # Policies are always high-evidence (promoted from validated experiences).
        evidence = 0.8
        graph = graph_scores.get(pol.policy_id, 0.0)
        recency = 0.5  # neutral without timestamps

        return {
            "semantic": round(semantic, 4),
            "confidence": round(confidence, 4),
            "evidence": round(evidence, 4),
            "graph": round(graph, 4),
            "recency": round(recency, 4),
        }

    def _combine_signals(self, signals: dict[str, float]) -> float:
        """Weighted combination of all signals."""
        score = (
            self.w_semantic * signals.get("semantic", 0.0)
            + self.w_confidence * signals.get("confidence", 0.0)
            + self.w_evidence * signals.get("evidence", 0.0)
            + self.w_graph * signals.get("graph", 0.0)
            + self.w_recency * signals.get("recency", 0.0)
        )
        return round(score, 4)

    # ---- graph-based scoring ----------------------------------------------
    def _graph_scores(self, context: dict[str, Any]) -> dict[str, float]:
        """Score nodes by graph proximity to context-relevant anchors."""
        if not self.graph:
            return {}

        # Find anchor nodes: the task family node + recent episode nodes.
        anchors: list[str] = []
        family = context.get("family", "")
        if family:
            family_node = f"family:{family}"
            if self.graph.get_node(family_node):
                anchors.append(family_node)

        if not anchors:
            return {}

        # BFS from anchors, accumulate scores by proximity.
        scores: dict[str, float] = {}
        for anchor in anchors:
            reached = self.graph.walk(
                anchor,
                edge_types=[EdgeType.supports, EdgeType.reinforces,
                            EdgeType.similar_to, EdgeType.promoted_to,
                            EdgeType.belongs_to],
                max_hops=3,
                direction="both",
            )
            for node_id, distance in reached.items():
                if node_id == anchor:
                    continue
                # Closer = higher score, decay with distance.
                proximity = 1.0 / (1.0 + distance)
                scores[node_id] = max(scores.get(node_id, 0.0), proximity)

        return scores
