from .base import PersistenceLayer
from .store import ExperienceStore
from .a0_none import NoPersistence
from .a1_memory import MemoryOnly
from .a2_engine.engine import ExperienceEngine

__all__ = [
    "PersistenceLayer", "ExperienceStore",
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
