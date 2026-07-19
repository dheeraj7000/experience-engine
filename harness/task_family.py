"""A task family = many variants that recur, so a learning curve can exist.

Public benchmarks are mostly one-shot; the whole thesis needs REPETITION.
`heldout_variants` are graded during validation but their episodes must never
enter any experience store — that is both the validation signal and the
contamination control.
"""
from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from .environment import Environment, Variant


@runtime_checkable
class TaskFamily(Protocol):
    family_id: str

    def train_variants(self) -> Iterable[Variant]: ...
    def heldout_variants(self) -> Iterable[Variant]: ...
    def make_env(self) -> Environment: ...
