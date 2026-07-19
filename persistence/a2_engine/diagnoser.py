"""Trace Debugger / Causal Reasoner (proposal 8.3).

The differentiator vs raw reflection: instead of 'the task failed', infer WHY
(root cause + critical step + counterfactual repair) so the induced lesson
targets the real cause, not a surface correlation.

Phase-1 implementation is heuristic with an optional LLM pass via the offline
provider. Real counterfactual replay is a Phase-2 upgrade.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from schemas import Episode


class CausalDiagnoser:
    def __init__(self, provider=None) -> None:
        self.provider = provider  # offline model role, optional in Phase 1

    def diagnose(self, episodes: list[Episode]) -> dict[str, Any]:
        """Given a cluster of same-signature failure episodes, return a
        structured diagnosis. Heuristic baseline; LLM refinement optional."""
        failing = [e for e in episodes
                   if not (e.outcome and e.outcome.task_success >= 1.0)]
        if not failing:
            return {"root_cause": "no failures in cluster", "critical_step": None,
                    "counterfactual_repair": "", "confidence": 0.0}

        # Heuristic: the most common last-observation keyword across failures.
        obs_terms: Counter = Counter()
        critical_actions: Counter = Counter()
        for e in failing:
            if e.steps:
                obs_terms.update(w for w in e.steps[-1].observation.lower().split()
                                 if len(w) > 4)
                critical_actions.update([e.steps[-1].action])
        top_term = obs_terms.most_common(1)[0][0] if obs_terms else "unknown"
        critical = critical_actions.most_common(1)[0][0] if critical_actions else None

        diagnosis = {
            "root_cause": f"recurring failure associated with '{top_term}'",
            "critical_step": critical,
            "counterfactual_repair": f"address '{top_term}' before the '{critical}' step",
            "confidence": min(0.5 + 0.1 * len(failing), 0.9),
        }
        if self.provider is not None:
            diagnosis = self._refine_with_llm(failing, diagnosis)
        return diagnosis

    def _refine_with_llm(self, failing: list[Episode], base: dict) -> dict:
        """Optional: ask the offline model to sharpen the root cause. Guarded so
        dry_run / unavailable providers fall back to the heuristic."""
        try:
            traces = "\n\n".join(
                f"Goal: {e.goal}\nSteps: " +
                " | ".join(f"{s.action}->{s.observation[:80]}" for s in e.steps)
                for e in failing[:5]
            )
            prompt = (
                "These agent episodes all failed similarly. In one sentence, state "
                "the most likely ROOT CAUSE (not the surface symptom):\n\n" + traces
            )
            resp = self.provider.complete(
                [{"role": "user", "content": prompt}], max_tokens=120)
            if resp.text.strip():
                base = dict(base, root_cause=resp.text.strip())
        except Exception:
            pass  # heuristic stands
        return base
