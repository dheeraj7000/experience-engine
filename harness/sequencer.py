"""Sequencer — orders training variants into an episode stream.

A learning curve only exists if related tasks recur, so the sequencer presents
a family's variants in a controlled order (seed permutes it). Held-out variants
are kept apart and never recorded.
"""
from __future__ import annotations

import random
from typing import Iterable

from .environment import Variant
from .task_family import TaskFamily


class Sequencer:
    def order(self, family: TaskFamily, seed: int, n_episodes: int | None = None) -> list[Variant]:
        variants = list(family.train_variants())
        rng = random.Random(seed)
        rng.shuffle(variants)
        if n_episodes and n_episodes > len(variants):
            # Repeat the family to build a longer curve (with re-shuffles).
            out: list[Variant] = []
            while len(out) < n_episodes:
                chunk = variants[:]
                rng.shuffle(chunk)
                out.extend(chunk)
            return out[:n_episodes]
        return variants[:n_episodes] if n_episodes else variants

    @staticmethod
    def heldout(family: TaskFamily) -> list[Variant]:
        return list(family.heldout_variants())
