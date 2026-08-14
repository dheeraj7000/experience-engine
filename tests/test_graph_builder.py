"""Tests for Phase 3: GraphBuilder — automatic graph construction from store."""
from persistence.graph import ExperienceGraph, NodeType, EdgeType
from persistence.graph_builder import GraphBuilder
from persistence.store import ExperienceStore
from schemas import Episode, ExperienceObject, PolicyObject, RewardVector
from schemas.episode import Step
from schemas.experience import ValidationStatus


def _episode(eid, family="test", success=False):
    return Episode(
        episode_id=eid, task_family=family, task_variant_id="v",
        goal="fix bug", initial_state={},
        outcome=RewardVector.from_success(success),
        steps=[Step(i=0, action="submit", observation="done")],
    )


def _experience(eid, family="test", source_episodes=None, lesson="fix the bug"):
    return ExperienceObject(
        experience_id=eid, task_family=family,
        source_episodes=source_episodes or [],
        lesson=lesson, root_cause="wrong logic",
        confidence=0.7, evidence_count=3,
        validation_status=ValidationStatus.active,
    )


def _policy(pid, scope="test", supporting=None):
    return PolicyObject(
        policy_id=pid, scope=scope,
        behavior="always check output",
        priority=70, confidence=0.8,
        supporting_experiences=supporting or [],
        validation_status=ValidationStatus.active,
    )


def test_build_full_creates_episode_nodes():
    store = ExperienceStore()
    store.add_episode(_episode("ep_1"))
    store.add_episode(_episode("ep_2"))
    builder = GraphBuilder()
    graph = builder.build_full(store)
    episodes = graph.nodes_by_type(NodeType.episode)
    assert len(episodes) == 2


def test_build_full_creates_experience_with_support_edges():
    store = ExperienceStore()
    store.add_episode(_episode("ep_1"))
    store.add_episode(_episode("ep_2"))
    store.add_experience(_experience("exp_1", source_episodes=["ep_1", "ep_2"]))
    builder = GraphBuilder()
    graph = builder.build_full(store)
    # Experience node exists.
    assert graph.get_node("exp_1") is not None
    # Support edges from episodes.
    evidence = graph.evidence_for("exp_1")
    assert set(evidence) == {"ep_1", "ep_2"}


def test_build_full_creates_policy_with_provenance():
    store = ExperienceStore()
    store.add_episode(_episode("ep_1"))
    store.add_experience(_experience("exp_1", source_episodes=["ep_1"]))
    store.add_policy(_policy("pol_1", supporting=["exp_1"]))
    builder = GraphBuilder()
    graph = builder.build_full(store)
    assert graph.get_node("pol_1") is not None
    assert graph.provenance("pol_1") == ["exp_1"]


def test_build_full_creates_family_nodes():
    store = ExperienceStore()
    store.add_episode(_episode("ep_1", family="bug_reproduction"))
    builder = GraphBuilder()
    graph = builder.build_full(store)
    family_nodes = graph.nodes_by_type(NodeType.task_family)
    assert any(n.node_id == "family:bug_reproduction" for n in family_nodes)


def test_similarity_edges_between_related_experiences():
    store = ExperienceStore()
    store.add_experience(_experience("exp_a", lesson="always verify the output before submitting"))
    store.add_experience(_experience("exp_b", lesson="verify output correctness before submission"))
    builder = GraphBuilder(similarity_threshold=0.3)
    graph = builder.build_full(store)
    # Should have a similar_to edge between the two.
    neighbors = graph.neighbors("exp_a", edge_types=[EdgeType.similar_to])
    assert "exp_b" in neighbors


def test_transfer_edges_across_families():
    store = ExperienceStore()
    store.add_experience(_experience(
        "exp_a", family="bug_reproduction",
        lesson="verify output correctness before submitting"))
    store.add_experience(_experience(
        "exp_b", family="toy_bug",
        lesson="verify output correctness before submitting the fix"))
    builder = GraphBuilder(similarity_threshold=0.3)
    graph = builder.build_full(store)
    # Cross-family similar experiences should get transfers_to edge.
    transfers = graph.transfer_candidates("exp_a")
    assert "exp_b" in transfers


def test_incremental_update_adds_new_nodes():
    store = ExperienceStore()
    store.add_episode(_episode("ep_1"))
    builder = GraphBuilder()
    graph = builder.build_full(store)
    assert graph.node_count == 2  # episode + family

    # Add another episode incrementally.
    new_ep = _episode("ep_2")
    store.add_episode(new_ep)
    builder.update_incremental(graph, store, new_episodes=[new_ep])
    assert graph.get_node("ep_2") is not None


def test_rejected_experiences_no_similarity_edges():
    store = ExperienceStore()
    active = _experience("exp_a", lesson="always verify output")
    rejected = _experience("exp_b", lesson="always verify output exactly the same")
    rejected.validation_status = ValidationStatus.rejected
    store.add_experience(active)
    store.add_experience(rejected)
    builder = GraphBuilder(similarity_threshold=0.3)
    graph = builder.build_full(store)
    # No similar_to edge to rejected experience.
    neighbors = graph.neighbors("exp_a", edge_types=[EdgeType.similar_to])
    assert "exp_b" not in neighbors
