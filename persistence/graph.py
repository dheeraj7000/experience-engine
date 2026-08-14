"""Experience Graph (Phase 3, proposal section 10).

A lightweight, in-process graph store that represents relationships between
episodes, experiences, skills, and policies. No external database required —
the graph is serializable to JSON and reconstructible from the ExperienceStore.

Node types: episode, experience, policy, task_family, failure_mode
Edge types: supports, contradicts, reinforces, generalizes, specializes,
            similar_to, transfers_to, supersedes, caused_by, promoted_to

The graph enables:
  - Evidence traversal (which episodes support an experience?)
  - Transfer detection (which experiences apply across task families?)
  - Conflict visualization (contradicts edges from Phase 2)
  - Policy provenance (experience → policy chain)
  - Similarity-based retrieval (graph walk from a query node)
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class NodeType(str, Enum):
    episode = "episode"
    experience = "experience"
    policy = "policy"
    task_family = "task_family"
    failure_mode = "failure_mode"


class EdgeType(str, Enum):
    supports = "supports"           # episode → experience
    contradicts = "contradicts"     # experience ↔ experience
    reinforces = "reinforces"       # episode → experience (additional evidence)
    generalizes = "generalizes"     # experience → experience (broader scope)
    specializes = "specializes"     # experience → experience (narrower scope)
    similar_to = "similar_to"       # experience ↔ experience
    transfers_to = "transfers_to"   # experience → experience (cross-family)
    supersedes = "supersedes"       # policy → policy
    caused_by = "caused_by"         # failure_mode → experience (root cause)
    promoted_to = "promoted_to"     # experience → policy
    belongs_to = "belongs_to"       # episode/experience → task_family


@dataclass
class GraphNode:
    node_id: str
    node_type: NodeType
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GraphNode":
        return cls(
            node_id=d["node_id"],
            node_type=NodeType(d["node_type"]),
            metadata=d.get("metadata", {}),
        )


@dataclass
class GraphEdge:
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GraphEdge":
        return cls(
            source=d["source"],
            target=d["target"],
            edge_type=EdgeType(d["edge_type"]),
            weight=d.get("weight", 1.0),
            metadata=d.get("metadata", {}),
        )


class ExperienceGraph:
    """In-process graph store for experience relationships.

    Supports:
      - Add/remove nodes and edges
      - Typed traversal (follow only specific edge types)
      - Neighborhood queries (all nodes within N hops)
      - Evidence aggregation (count supporting episodes for an experience)
      - Provenance chains (experience → policy lineage)
      - Similarity subgraph (experiences related to a query experience)
      - Serialization to/from JSON
    """

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        # Adjacency: source → list of edges
        self._outgoing: dict[str, list[GraphEdge]] = defaultdict(list)
        # Reverse adjacency: target → list of edges
        self._incoming: dict[str, list[GraphEdge]] = defaultdict(list)

    # ---- mutations --------------------------------------------------------
    def add_node(self, node_id: str, node_type: NodeType,
                 metadata: dict[str, Any] | None = None) -> GraphNode:
        """Add or update a node. Returns the node."""
        if node_id in self._nodes:
            # Update metadata on existing node.
            existing = self._nodes[node_id]
            if metadata:
                existing.metadata.update(metadata)
            return existing
        node = GraphNode(node_id=node_id, node_type=node_type,
                         metadata=metadata or {})
        self._nodes[node_id] = node
        return node

    def add_edge(self, source: str, target: str, edge_type: EdgeType,
                 weight: float = 1.0, metadata: dict[str, Any] | None = None
                 ) -> GraphEdge:
        """Add an edge. Deduplicates by (source, target, edge_type)."""
        # Check for existing identical edge.
        for e in self._outgoing.get(source, []):
            if e.target == target and e.edge_type == edge_type:
                # Update weight/metadata on existing edge.
                e.weight = weight
                if metadata:
                    e.metadata.update(metadata)
                return e
        edge = GraphEdge(source=source, target=target, edge_type=edge_type,
                         weight=weight, metadata=metadata or {})
        self._outgoing[source].append(edge)
        self._incoming[target].append(edge)
        return edge

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        self._nodes.pop(node_id, None)
        # Remove outgoing edges.
        for edge in self._outgoing.pop(node_id, []):
            self._incoming[edge.target] = [
                e for e in self._incoming[edge.target] if e.source != node_id
            ]
        # Remove incoming edges.
        for edge in self._incoming.pop(node_id, []):
            self._outgoing[edge.source] = [
                e for e in self._outgoing[edge.source] if e.target != node_id
            ]

    def remove_edge(self, source: str, target: str, edge_type: EdgeType) -> None:
        """Remove a specific edge."""
        self._outgoing[source] = [
            e for e in self._outgoing[source]
            if not (e.target == target and e.edge_type == edge_type)
        ]
        self._incoming[target] = [
            e for e in self._incoming[target]
            if not (e.source == source and e.edge_type == edge_type)
        ]

    # ---- queries ----------------------------------------------------------
    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self._outgoing.values())

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def nodes_by_type(self, node_type: NodeType) -> list[GraphNode]:
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def outgoing(self, node_id: str, edge_type: EdgeType | None = None
                 ) -> list[GraphEdge]:
        """Get outgoing edges, optionally filtered by type."""
        edges = self._outgoing.get(node_id, [])
        if edge_type is not None:
            return [e for e in edges if e.edge_type == edge_type]
        return list(edges)

    def incoming(self, node_id: str, edge_type: EdgeType | None = None
                 ) -> list[GraphEdge]:
        """Get incoming edges, optionally filtered by type."""
        edges = self._incoming.get(node_id, [])
        if edge_type is not None:
            return [e for e in edges if e.edge_type == edge_type]
        return list(edges)

    def neighbors(self, node_id: str, edge_types: Iterable[EdgeType] | None = None,
                  direction: str = "both") -> list[str]:
        """Get neighbor node IDs, optionally filtered by edge type and direction."""
        result: set[str] = set()
        type_set = set(edge_types) if edge_types else None

        if direction in ("out", "both"):
            for e in self._outgoing.get(node_id, []):
                if type_set is None or e.edge_type in type_set:
                    result.add(e.target)
        if direction in ("in", "both"):
            for e in self._incoming.get(node_id, []):
                if type_set is None or e.edge_type in type_set:
                    result.add(e.source)
        return list(result)

    def walk(self, start: str, edge_types: Iterable[EdgeType] | None = None,
             max_hops: int = 3, direction: str = "out") -> dict[str, int]:
        """BFS walk from `start`, returning {node_id: distance}.

        Useful for finding related experiences within N hops.
        """
        type_set = set(edge_types) if edge_types else None
        visited: dict[str, int] = {start: 0}
        frontier = [start]
        for depth in range(1, max_hops + 1):
            next_frontier: list[str] = []
            for node_id in frontier:
                for neighbor in self.neighbors(node_id, edge_types=type_set,
                                               direction=direction):
                    if neighbor not in visited:
                        visited[neighbor] = depth
                        next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        return visited

    # ---- higher-level queries --------------------------------------------
    def evidence_for(self, experience_id: str) -> list[str]:
        """Return episode IDs that support/reinforce an experience."""
        episode_ids: list[str] = []
        for e in self._incoming.get(experience_id, []):
            if e.edge_type in (EdgeType.supports, EdgeType.reinforces):
                node = self._nodes.get(e.source)
                if node and node.node_type == NodeType.episode:
                    episode_ids.append(e.source)
        return episode_ids

    def provenance(self, policy_id: str) -> list[str]:
        """Trace back from a policy to its supporting experiences."""
        exp_ids: list[str] = []
        for e in self._incoming.get(policy_id, []):
            if e.edge_type == EdgeType.promoted_to:
                exp_ids.append(e.source)
        return exp_ids

    def contradictions_of(self, experience_id: str) -> list[str]:
        """Find all experiences that contradict a given experience."""
        result: list[str] = []
        for e in self._outgoing.get(experience_id, []):
            if e.edge_type == EdgeType.contradicts:
                result.append(e.target)
        for e in self._incoming.get(experience_id, []):
            if e.edge_type == EdgeType.contradicts:
                result.append(e.source)
        return result

    def transfer_candidates(self, experience_id: str) -> list[str]:
        """Find experiences that transfer knowledge to/from this one."""
        result: list[str] = []
        for e in self._outgoing.get(experience_id, []):
            if e.edge_type in (EdgeType.similar_to, EdgeType.transfers_to):
                result.append(e.target)
        for e in self._incoming.get(experience_id, []):
            if e.edge_type in (EdgeType.similar_to, EdgeType.transfers_to):
                result.append(e.source)
        return result

    def subgraph(self, node_ids: Iterable[str]) -> "ExperienceGraph":
        """Extract a subgraph containing only the specified nodes and edges between them."""
        ids = set(node_ids)
        sub = ExperienceGraph()
        for nid in ids:
            node = self._nodes.get(nid)
            if node:
                sub.add_node(node.node_id, node.node_type, dict(node.metadata))
        for nid in ids:
            for edge in self._outgoing.get(nid, []):
                if edge.target in ids:
                    sub.add_edge(edge.source, edge.target, edge.edge_type,
                                 edge.weight, dict(edge.metadata))
        return sub

    # ---- serialization ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict()
                      for edges in self._outgoing.values()
                      for e in edges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperienceGraph":
        g = cls()
        for nd in data.get("nodes", []):
            node = GraphNode.from_dict(nd)
            g._nodes[node.node_id] = node
        for ed in data.get("edges", []):
            edge = GraphEdge.from_dict(ed)
            g._outgoing[edge.source].append(edge)
            g._incoming[edge.target].append(edge)
        return g

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ExperienceGraph":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)
