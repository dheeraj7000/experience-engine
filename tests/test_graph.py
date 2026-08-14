"""Tests for Phase 3: ExperienceGraph — nodes, edges, traversal, serialization."""
import json
import tempfile
from pathlib import Path

from persistence.graph import (
    ExperienceGraph, GraphNode, GraphEdge, NodeType, EdgeType,
)


def test_add_node_and_retrieve():
    g = ExperienceGraph()
    g.add_node("ep_1", NodeType.episode, {"goal": "fix bug"})
    assert g.node_count == 1
    node = g.get_node("ep_1")
    assert node is not None
    assert node.node_type == NodeType.episode
    assert node.metadata["goal"] == "fix bug"


def test_add_edge_and_query():
    g = ExperienceGraph()
    g.add_node("ep_1", NodeType.episode)
    g.add_node("exp_1", NodeType.experience)
    g.add_edge("ep_1", "exp_1", EdgeType.supports, weight=0.9)
    assert g.edge_count == 1
    out = g.outgoing("ep_1")
    assert len(out) == 1
    assert out[0].target == "exp_1"
    assert out[0].edge_type == EdgeType.supports


def test_edge_deduplication():
    g = ExperienceGraph()
    g.add_node("a", NodeType.experience)
    g.add_node("b", NodeType.experience)
    g.add_edge("a", "b", EdgeType.similar_to, weight=0.5)
    g.add_edge("a", "b", EdgeType.similar_to, weight=0.8)  # updates existing
    assert g.edge_count == 1
    assert g.outgoing("a")[0].weight == 0.8


def test_neighbors():
    g = ExperienceGraph()
    g.add_node("a", NodeType.experience)
    g.add_node("b", NodeType.experience)
    g.add_node("c", NodeType.policy)
    g.add_edge("a", "b", EdgeType.similar_to)
    g.add_edge("a", "c", EdgeType.promoted_to)
    neighbors = g.neighbors("a", direction="out")
    assert set(neighbors) == {"b", "c"}
    # Filter by type.
    sim_only = g.neighbors("a", edge_types=[EdgeType.similar_to], direction="out")
    assert sim_only == ["b"]


def test_walk_bfs():
    g = ExperienceGraph()
    g.add_node("a", NodeType.experience)
    g.add_node("b", NodeType.experience)
    g.add_node("c", NodeType.experience)
    g.add_node("d", NodeType.experience)
    g.add_edge("a", "b", EdgeType.similar_to)
    g.add_edge("b", "c", EdgeType.similar_to)
    g.add_edge("c", "d", EdgeType.similar_to)
    reachable = g.walk("a", max_hops=2, direction="out")
    assert reachable == {"a": 0, "b": 1, "c": 2}
    # d is 3 hops away, beyond max_hops=2.
    assert "d" not in reachable


def test_evidence_for():
    g = ExperienceGraph()
    g.add_node("ep_1", NodeType.episode)
    g.add_node("ep_2", NodeType.episode)
    g.add_node("exp_1", NodeType.experience)
    g.add_edge("ep_1", "exp_1", EdgeType.supports)
    g.add_edge("ep_2", "exp_1", EdgeType.reinforces)
    evidence = g.evidence_for("exp_1")
    assert set(evidence) == {"ep_1", "ep_2"}


def test_provenance():
    g = ExperienceGraph()
    g.add_node("exp_1", NodeType.experience)
    g.add_node("pol_1", NodeType.policy)
    g.add_edge("exp_1", "pol_1", EdgeType.promoted_to)
    assert g.provenance("pol_1") == ["exp_1"]


def test_contradictions_of():
    g = ExperienceGraph()
    g.add_node("exp_a", NodeType.experience)
    g.add_node("exp_b", NodeType.experience)
    g.add_edge("exp_a", "exp_b", EdgeType.contradicts)
    assert "exp_b" in g.contradictions_of("exp_a")
    assert "exp_a" in g.contradictions_of("exp_b")


def test_subgraph():
    g = ExperienceGraph()
    g.add_node("a", NodeType.experience)
    g.add_node("b", NodeType.experience)
    g.add_node("c", NodeType.experience)
    g.add_edge("a", "b", EdgeType.similar_to)
    g.add_edge("b", "c", EdgeType.similar_to)
    sub = g.subgraph(["a", "b"])
    assert sub.node_count == 2
    assert sub.edge_count == 1  # only a->b, not b->c


def test_remove_node():
    g = ExperienceGraph()
    g.add_node("a", NodeType.experience)
    g.add_node("b", NodeType.experience)
    g.add_edge("a", "b", EdgeType.similar_to)
    g.remove_node("a")
    assert g.node_count == 1
    assert g.edge_count == 0
    assert g.incoming("b") == []


def test_serialization_roundtrip():
    g = ExperienceGraph()
    g.add_node("ep_1", NodeType.episode, {"goal": "fix"})
    g.add_node("exp_1", NodeType.experience, {"confidence": 0.8})
    g.add_edge("ep_1", "exp_1", EdgeType.supports, weight=0.9)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "graph.json"
        g.save(path)
        loaded = ExperienceGraph.load(path)

    assert loaded.node_count == 2
    assert loaded.edge_count == 1
    assert loaded.get_node("exp_1").metadata["confidence"] == 0.8


def test_nodes_by_type():
    g = ExperienceGraph()
    g.add_node("ep_1", NodeType.episode)
    g.add_node("ep_2", NodeType.episode)
    g.add_node("exp_1", NodeType.experience)
    episodes = g.nodes_by_type(NodeType.episode)
    assert len(episodes) == 2
    experiences = g.nodes_by_type(NodeType.experience)
    assert len(experiences) == 1


def test_transfer_candidates():
    g = ExperienceGraph()
    g.add_node("exp_a", NodeType.experience)
    g.add_node("exp_b", NodeType.experience)
    g.add_edge("exp_a", "exp_b", EdgeType.transfers_to)
    assert "exp_b" in g.transfer_candidates("exp_a")
    assert "exp_a" in g.transfer_candidates("exp_b")
