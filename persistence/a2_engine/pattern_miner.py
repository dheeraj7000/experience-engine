"""Pattern Miner (Phase 2, proposal 8.5 / 8.6).

Clusters episodes by structural similarity — not just the coarse
failure_signature, but multi-dimensional features: failure type, action
sequences, observation keywords, task family, and outcome shape.

The miner operates on the OFFLINE path (consolidation), so cost is acceptable.
It produces clusters that the diagnoser can reason about causally, and detects
cross-cluster patterns (e.g. "the same root cause manifests across two
different failure types").
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

from schemas import Episode
from .taxonomy import FailureType, classify_failure, failure_features

_WORD = re.compile(r"[a-z0-9_]+")


def _bow(text: str) -> Counter:
    return Counter(_WORD.findall(text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


class PatternCluster:
    """A cluster of structurally-similar failure episodes."""

    def __init__(self, cluster_id: str, episodes: list[Episode]) -> None:
        self.cluster_id = cluster_id
        self.episodes = episodes
        self.failure_type = self._dominant_type()
        self.common_actions = self._common_actions()
        self.observation_keywords = self._top_keywords()
        self.size = len(episodes)

    def _dominant_type(self) -> FailureType:
        types = [classify_failure(e) for e in self.episodes]
        counts = Counter(types)
        return counts.most_common(1)[0][0] if counts else FailureType.unknown

    def _common_actions(self) -> list[str]:
        """Actions that appear in >50% of episodes in this cluster."""
        action_counts: Counter = Counter()
        for e in self.episodes:
            action_counts.update(set(s.action for s in e.steps))
        threshold = len(self.episodes) * 0.5
        return [a for a, c in action_counts.most_common() if c >= threshold]

    def _top_keywords(self, k: int = 5) -> list[str]:
        """Most frequent observation terms across the cluster."""
        terms: Counter = Counter()
        for e in self.episodes:
            for s in e.steps:
                terms.update(w for w in _WORD.findall(s.observation.lower()) if len(w) > 3)
        return [t for t, _ in terms.most_common(k)]

    def summary(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "failure_type": self.failure_type.value,
            "common_actions": self.common_actions,
            "observation_keywords": self.observation_keywords,
            "episode_ids": [e.episode_id for e in self.episodes],
        }


class PatternMiner:
    """Finds structural patterns in failure episodes.

    Two clustering strategies:
    1. Signature-based (fast, coarse) — same as Phase 1's failure_signature.
    2. Feature-based (richer) — multi-dimensional similarity using taxonomy
       features, action sequences, and observation overlap.
    """

    def __init__(self, similarity_threshold: float = 0.4,
                 min_cluster_size: int = 2) -> None:
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size

    def cluster_by_signature(self, episodes: list[Episode]) -> dict[str, list[Episode]]:
        """Phase 1-compatible: group by failure_signature string."""
        clusters: dict[str, list[Episode]] = defaultdict(list)
        for e in episodes:
            if e.outcome and e.outcome.task_success >= 1.0:
                continue
            clusters[e.failure_signature()].append(e)
        return {k: v for k, v in clusters.items() if len(v) >= self.min_cluster_size}

    def cluster_by_features(self, episodes: list[Episode]) -> list[PatternCluster]:
        """Phase 2: multi-dimensional clustering using structural features.

        Uses a simple agglomerative approach: start with each failure episode
        as its own cluster, then merge the most similar pair until no pair
        exceeds the threshold. This is O(n^2) but n is bounded by checkpoint
        intervals (typically 5-30 episodes).
        """
        failures = [e for e in episodes
                    if not (e.outcome and e.outcome.task_success >= 1.0)]
        if not failures:
            return []

        # Build feature vectors.
        features = [failure_features(e) for e in failures]

        # Start: each episode in its own cluster.
        clusters: list[list[int]] = [[i] for i in range(len(failures))]

        # Agglomerative merge.
        while True:
            best_sim = -1.0
            best_i, best_j = -1, -1
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    sim = self._cluster_similarity(clusters[i], clusters[j], features)
                    if sim > best_sim:
                        best_sim = sim
                        best_i, best_j = i, j
            if best_sim < self.similarity_threshold or best_i < 0:
                break
            # Merge j into i.
            clusters[best_i].extend(clusters[best_j])
            clusters.pop(best_j)

        # Build PatternCluster objects for clusters meeting min size.
        result: list[PatternCluster] = []
        for idx, members in enumerate(clusters):
            if len(members) < self.min_cluster_size:
                continue
            eps = [failures[m] for m in members]
            result.append(PatternCluster(f"cluster_{idx}", eps))
        return result

    def find_cross_cluster_patterns(self, clusters: list[PatternCluster]
                                    ) -> list[dict[str, Any]]:
        """Detect patterns that span multiple clusters — potential shared
        root causes that manifest differently on the surface."""
        patterns: list[dict[str, Any]] = []
        for i, c1 in enumerate(clusters):
            for c2 in clusters[i + 1:]:
                shared_kw = set(c1.observation_keywords) & set(c2.observation_keywords)
                shared_actions = set(c1.common_actions) & set(c2.common_actions)
                if len(shared_kw) >= 2 or len(shared_actions) >= 1:
                    patterns.append({
                        "clusters": [c1.cluster_id, c2.cluster_id],
                        "shared_keywords": list(shared_kw),
                        "shared_actions": list(shared_actions),
                        "hypothesis": (
                            f"Clusters share '{', '.join(shared_kw or shared_actions)}' — "
                            f"possibly same root cause across {c1.failure_type.value} "
                            f"and {c2.failure_type.value} failure types."
                        ),
                    })
        return patterns

    def _cluster_similarity(self, c1: list[int], c2: list[int],
                            features: list[dict]) -> float:
        """Average-linkage similarity between two clusters."""
        total = 0.0
        pairs = 0
        for i in c1:
            for j in c2:
                total += self._episode_similarity(features[i], features[j])
                pairs += 1
        return total / pairs if pairs else 0.0

    def _episode_similarity(self, f1: dict, f2: dict) -> float:
        """Multi-dimensional similarity between two failure feature dicts."""
        scores: list[float] = []

        # Same failure type: strong signal.
        scores.append(1.0 if f1["failure_type"] == f2["failure_type"] else 0.0)

        # Same task family.
        scores.append(1.0 if f1["task_family"] == f2["task_family"] else 0.0)

        # Action sequence overlap (Jaccard).
        a1 = set(f1.get("actions_taken", []))
        a2 = set(f2.get("actions_taken", []))
        scores.append(_jaccard(a1, a2))

        # Observation keyword overlap.
        obs1 = set(_WORD.findall(f1.get("last_observation_prefix", "").lower()))
        obs2 = set(_WORD.findall(f2.get("last_observation_prefix", "").lower()))
        scores.append(_jaccard(obs1, obs2))

        # Step count proximity (normalized).
        n1, n2 = f1.get("n_steps", 0), f2.get("n_steps", 0)
        max_n = max(n1, n2, 1)
        scores.append(1.0 - abs(n1 - n2) / max_n)

        # Weighted combination: failure_type and family matter most.
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        return sum(s * w for s, w in zip(scores, weights))
