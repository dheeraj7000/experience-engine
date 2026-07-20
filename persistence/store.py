"""Experience store — Phase 0/1 backing: JSONL append log + in-memory index
with a dependency-light bag-of-words cosine search. Graph DB is a Phase-3
concern; this interface is stable enough to swap the backend later.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from schemas import Episode, ExperienceObject, PolicyObject
from schemas.experience import ValidationStatus

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


class ExperienceStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else None
        self.episodes: list[Episode] = []
        self.experiences: list[ExperienceObject] = []
        self.policies: list[PolicyObject] = []
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    # ---- writes -----------------------------------------------------------
    def add_episode(self, ep: Episode) -> None:
        self.episodes.append(ep)
        self._append("episodes.jsonl", ep.model_dump(mode="json"))

    def add_experience(self, exp: ExperienceObject) -> None:
        self.experiences.append(exp)
        self._append("experiences.jsonl", exp.model_dump(mode="json"))

    def add_policy(self, pol: PolicyObject) -> None:
        self.policies.append(pol)
        self._append("policies.jsonl", pol.model_dump(mode="json"))

    def _append(self, name: str, obj: dict) -> None:
        if not self.root:
            return
        with (self.root / name).open("a") as f:
            f.write(json.dumps(obj) + "\n")

    # ---- reads ------------------------------------------------------------
    def search_episodes(self, query: str, k: int = 3) -> list[Episode]:
        q = _bow(query)
        scored = [(_cosine(q, _bow(self._ep_text(e))), e) for e in self.episodes]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for s, e in scored[:k] if s > 0]

    def search_experiences(self, context: dict[str, Any], query: str, k: int = 3,
                           min_confidence: float = 0.0) -> list[ExperienceObject]:
        q = _bow(query + " " + " ".join(f"{v}" for v in context.values()))
        active = [e for e in self.experiences
                  if e.validation_status in (ValidationStatus.active, ValidationStatus.validated)
                  and e.confidence >= min_confidence
                  and e.matches(context)]
        scored = [(_cosine(q, _bow(e.lesson + " " + e.root_cause)), e) for e in active]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for s, e in scored[:k]]

    def find_mergeable(self, task_family: str, context_conditions: dict[str, Any]
                       ) -> ExperienceObject | None:
        """An existing (non-rejected) experience for the exact same family +
        conditions. New evidence should reinforce it, not spawn a near-
        duplicate — duplicates bloat injected context for zero added lesson."""
        for e in self.experiences:
            if (e.task_family == task_family
                    and e.context_conditions == context_conditions
                    and e.validation_status != ValidationStatus.rejected):
                return e
        return None

    def active_policies(self, scope: str) -> list[PolicyObject]:
        pols = [p for p in self.policies
                if p.validation_status == ValidationStatus.active
                and (p.scope == scope or p.scope == "")]
        pols.sort(key=lambda p: p.priority, reverse=True)
        return pols

    @staticmethod
    def _ep_text(e: Episode) -> str:
        return f"{e.goal} " + " ".join(s.observation for s in e.steps)

    # ---- reproducibility --------------------------------------------------
    def snapshot(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "episodes": [e.model_dump(mode="json") for e in self.episodes],
            "experiences": [e.model_dump(mode="json") for e in self.experiences],
            "policies": [p.model_dump(mode="json") for p in self.policies],
        }
        path.write_text(json.dumps(payload, indent=2))
