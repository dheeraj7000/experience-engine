from .base import PersistenceLayer
from .graph import ExperienceGraph, GraphNode, GraphEdge, NodeType, EdgeType
from .graph_builder import GraphBuilder
from .hybrid_retriever import HybridRetriever, RetrievalResult
from .store import ExperienceStore
from .a0_none import NoPersistence
from .a1_memory import MemoryOnly
from .a2_engine.engine import ExperienceEngine

__all__ = [
    "PersistenceLayer", "ExperienceStore",
    "ExperienceGraph", "GraphNode", "GraphEdge", "NodeType", "EdgeType",
    "GraphBuilder", "HybridRetriever", "RetrievalResult",
    "NoPersistence", "MemoryOnly", "ExperienceEngine",
    "build_persistence",
]


def build_persistence(config: str, store: "ExperienceStore", offline_provider=None):
    """Factory: 'a0' | 'a1' | 'a2' -> a PersistenceLayer sharing one store."""
    config = config.lower()
    if config == "a0":
        return NoPersistence(store)
    if config == "a1":
        return MemoryOnly(store)
    if config == "a2":
        return ExperienceEngine(store, offline_provider=offline_provider)
    raise ValueError(f"unknown persistence config: {config!r}")
