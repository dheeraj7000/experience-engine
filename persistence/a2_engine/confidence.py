"""Confidence Manager (minimal, proposal 8.10).

Phase-1 form: evidence count + outcome consistency, penalized by
contradictions. Calibration (confidence vs actual improvement) is a Phase-2
concern; the shape here leaves room for it.
"""
from __future__ import annotations


def compute_confidence(evidence_count: int, consistency: float,
                       contradictions: int) -> float:
    """
    evidence_count : # supporting episodes
    consistency    : fraction of supporting episodes agreeing on the outcome [0,1]
    contradictions : # episodes contradicting the lesson
    """
    # More evidence -> more confidence, with diminishing returns.
    evidence_term = 1.0 - (1.0 / (1.0 + max(evidence_count, 0)))
    penalty = 0.15 * max(contradictions, 0)
    conf = evidence_term * max(min(consistency, 1.0), 0.0) - penalty
    return round(max(0.0, min(1.0, conf)), 4)
