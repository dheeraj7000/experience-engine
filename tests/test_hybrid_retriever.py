"""Tests for Phase 3: HybridRetriever — multi-signal ranked retrieval."""
from persistence.graph import ExperienceGraph, NodeType, EdgeType
from persistence.hybrid_retriever import HybridRetriever, RetrievalResult
from persistence.store import ExperienceStore
from schemas import Episode, ExperienceObject, PolicyObject, RewardVector
from schemas.episode import Step
from schemas.experience import ValidationStatus


def _episode(eid, family="test"):
    return Episode(
        episode_id=eid, task_family=family, task_variant_id="v",
        goal="fix bug", initial_state={},
        outcome=RewardVector.from_success(False),
        steps=[Step(i=0, action="submit", observation="failed")],
    )


def _experience(eid, family="test", lesson="verify output before submitting",
                confidence=0.7, conditions=None):
    return ExperienceObject(
        experience_id=eid, task_family=family,
        context_conditions=conditions or {"family": family},
        source_episodes=["ep_1", "ep_2"],
        lesson=lesson, root_cause="wrong logic",
        confidence=confidence, evidence_count=3,
        validation_status=ValidationStatus.active,
    )


def _policy(pid, scope="test"):
    return PolicyObject(
        policy_id=pid, scope=scope,
        behavior="always verify output",
        priority=70, confidence=0.8,
        supporting_experiences=["exp_1"],
        validation_status=ValidationStatus.active,
    )


def test_retrieve_returns_ranked_results():
    store = ExperienceStore()
    store.add_experience(_experience("exp_1", lesson="verify output before submitting"))
    store.add_experience(_experience("exp_2", lesson="check schema before querying"))
    retriever = HybridRetriever(store)
    results = retriever.retrieve({"family": "test", "goal": "verify the output"})
    assert len(results) >= 1
    # Results should be RetrievalResult instances.
    assert all(isinstance(r, RetrievalResult) for r in results)
    # First result should be most relevant.
    assert results[0].score >= results[-1].score


def test_retrieve_prefers_higher_confidence():
    store = ExperienceStore()
    store.add_experience(_experience("exp_lo", confidence=0.3,
                                     lesson="some advice"))
    store.add_experience(_experience("exp_hi", confidence=0.9,
                                     lesson="some advice"))
    retriever = HybridRetriever(store, min_confidence=0.2)
    results = retriever.retrieve({"family": "test", "goal": "some advice"})
    # Higher confidence should rank higher.
    ids = [r.item_id for r in results]
    assert ids.index("exp_hi") < ids.index("exp_lo")


def test_retrieve_filters_by_context():
    store = ExperienceStore()
    store.add_experience(_experience("exp_a", family="sql",
                                     conditions={"family": "sql"}))
    store.add_experience(_experience("exp_b", family="csv",
                                     conditions={"family": "csv"}))
    retriever = HybridRetriever(store)
    results = retriever.retrieve({"family": "sql", "goal": "fix query"})
    ids = [r.item_id for r in results]
    # Only exp_a matches context family=sql.
    assert "exp_a" in ids
    assert "exp_b" not in ids


def test_retrieve_includes_policies():
    store = ExperienceStore()
    store.add_experience(_experience("exp_1"))
    store.add_policy(_policy("pol_1"))
    retriever = HybridRetriever(store)
    results = retriever.retrieve({"family": "test", "goal": "verify"})
    types = [r.item_type for r in results]
    assert "policy" in types
    assert "experience" in types


def test_retrieve_text_returns_formatted_string():
    store = ExperienceStore()
    store.add_experience(_experience("exp_1"))
    retriever = HybridRetriever(store)
    text = retriever.retrieve_text({"family": "test", "goal": "verify"})
    assert "Experience" in text
    assert "verify output" in text


def test_retrieve_with_graph_boosts_connected_nodes():
    store = ExperienceStore()
    store.add_experience(_experience("exp_near", lesson="close to family"))
    store.add_experience(_experience("exp_far", lesson="distant unrelated"))

    graph = ExperienceGraph()
    graph.add_node("family:test", NodeType.task_family)
    graph.add_node("exp_near", NodeType.experience)
    graph.add_node("exp_far", NodeType.experience)
    # Only exp_near is connected to the family node.
    graph.add_edge("exp_near", "family:test", EdgeType.belongs_to)

    retriever = HybridRetriever(store, graph, w_graph=0.5)
    results = retriever.retrieve({"family": "test", "goal": "task"})
    # exp_near should score higher due to graph proximity.
    if len(results) >= 2:
        ids = [r.item_id for r in results]
        assert ids.index("exp_near") < ids.index("exp_far")


def test_retrieve_respects_max_results():
    store = ExperienceStore()
    for i in range(10):
        store.add_experience(_experience(f"exp_{i}", lesson=f"lesson {i}"))
    retriever = HybridRetriever(store, max_results=3)
    results = retriever.retrieve({"family": "test", "goal": "anything"})
    assert len(results) <= 3


def test_retrieve_empty_store():
    store = ExperienceStore()
    retriever = HybridRetriever(store)
    results = retriever.retrieve({"family": "test", "goal": "fix"})
    assert results == []


def test_signals_breakdown_present():
    store = ExperienceStore()
    store.add_experience(_experience("exp_1"))
    retriever = HybridRetriever(store)
    results = retriever.retrieve({"family": "test", "goal": "verify output"})
    assert results
    signals = results[0].signals
    assert "semantic" in signals
    assert "confidence" in signals
    assert "evidence" in signals
