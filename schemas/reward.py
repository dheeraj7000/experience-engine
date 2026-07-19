"""Multi-dimensional, execution-grounded reward vector.

The load-bearing fields are `task_success` and `reproducible`: in the QA
domain these come from RUNNING THE TESTS, not from an LLM judge. That single
fact is what makes the downstream learning loop trustworthy.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class RewardVector(BaseModel):
    task_success: float = Field(0.0, ge=0.0, le=1.0)   # from test/CI execution
    partial_credit: float = Field(0.0, ge=0.0, le=1.0)
    efficiency: float = Field(0.0, ge=0.0, le=1.0)     # vs family baseline steps
    cost: float = Field(0.0, ge=0.0, le=1.0)
    latency: float = Field(0.0, ge=0.0, le=1.0)
    safety: float = Field(1.0, ge=0.0, le=1.0)
    reproducible: bool = True
    overall: float = Field(0.0, ge=0.0, le=1.0)

    # Weights for the scalar summary. Success dominates; the rest shape ties.
    _WEIGHTS = {
        "task_success": 0.55,
        "partial_credit": 0.10,
        "efficiency": 0.10,
        "cost": 0.08,
        "latency": 0.07,
        "safety": 0.10,
    }

    def compute_overall(self) -> "RewardVector":
        self.overall = round(
            sum(getattr(self, k) * w for k, w in self._WEIGHTS.items()), 4
        )
        return self

    @classmethod
    def from_success(cls, success: bool, **kw) -> "RewardVector":
        rv = cls(task_success=1.0 if success else 0.0,
                 partial_credit=1.0 if success else 0.0,
                 reproducible=success, **kw)
        return rv.compute_overall()
