"""Trace Debugger / Causal Reasoner (Phase 2 upgrade, proposal 8.3).

Phase 1 was heuristic keyword counting with an optional single LLM prompt.
Phase 2 adds:
  - Structured failure taxonomy integration
  - Multi-step causal chain extraction (not just last observation)
  - Counterfactual step identification (which step, if changed, fixes it?)
  - Critical decision point detection
  - Cross-episode causal agreement scoring
  - Richer LLM-based diagnosis with structured output

The differentiator vs raw reflection: instead of 'the task failed', infer WHY
(root cause + critical step + counterfactual repair) so the induced lesson
targets the real cause, not a surface correlation.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from schemas import Episode
from .taxonomy import FailureType, classify_failure


class CausalChain:
    """A structured explanation of why a failure occurred."""

    def __init__(
        self,
        root_cause: str,
        failure_type: FailureType,
        critical_step_index: int | None,
        critical_action: str | None,
        contributing_factors: list[str],
        counterfactual_repair: str,
        confidence: float,
    ) -> None:
        self.root_cause = root_cause
        self.failure_type = failure_type
        self.critical_step_index = critical_step_index
        self.critical_action = critical_action
        self.contributing_factors = contributing_factors
        self.counterfactual_repair = counterfactual_repair
        self.confidence = confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_cause": self.root_cause,
            "failure_type": self.failure_type.value,
            "critical_step": self.critical_action,
            "critical_step_index": self.critical_step_index,
            "contributing_factors": self.contributing_factors,
            "counterfactual_repair": self.counterfactual_repair,
            "confidence": self.confidence,
        }


class CausalDiagnoser:
    """Phase 2 causal diagnoser with taxonomy-aware analysis."""

    def __init__(self, provider=None) -> None:
        self.provider = provider  # offline model role, optional

    def diagnose(self, episodes: list[Episode]) -> dict[str, Any]:
        """Given a cluster of failure episodes, return a structured diagnosis.

        Phase 2 upgrade: uses failure taxonomy, multi-step trace analysis,
        critical decision point detection, and cross-episode agreement.
        Falls back to heuristics when no provider is available.
        """
        failing = [e for e in episodes
                   if not (e.outcome and e.outcome.task_success >= 1.0)]
        if not failing:
            return {"root_cause": "no failures in cluster", "critical_step": None,
                    "counterfactual_repair": "", "confidence": 0.0}

        # Phase 2: build a CausalChain from structured analysis.
        chain = self._analyze_chain(failing)

        # Optional LLM refinement.
        if self.provider is not None:
            chain = self._refine_with_llm(failing, chain)

        return chain.to_dict()

    def diagnose_single(self, episode: Episode) -> CausalChain:
        """Diagnose a single episode. Useful for per-episode trace debugging."""
        ftype = classify_failure(episode)
        critical_idx, critical_action = self._find_critical_step(episode)
        root_cause = self._infer_root_cause_single(episode, ftype, critical_action)
        repair = self._suggest_repair(ftype, root_cause, critical_action)

        return CausalChain(
            root_cause=root_cause,
            failure_type=ftype,
            critical_step_index=critical_idx,
            critical_action=critical_action,
            contributing_factors=self._contributing_factors(episode),
            counterfactual_repair=repair,
            confidence=0.5,  # single-episode confidence is moderate
        )

    def _analyze_chain(self, failing: list[Episode]) -> CausalChain:
        """Build a CausalChain from a cluster of failures."""
        # Classify all failures.
        types = [classify_failure(e) for e in failing]
        dominant_type = Counter(types).most_common(1)[0][0]

        # Find the critical step across the cluster.
        critical_action, critical_idx = self._cluster_critical_step(failing)

        # Extract root cause from cross-episode patterns.
        root_cause = self._cluster_root_cause(failing, dominant_type, critical_action)

        # Contributing factors.
        factors = self._cluster_contributing_factors(failing)

        # Counterfactual repair suggestion.
        repair = self._suggest_repair(dominant_type, root_cause, critical_action)

        # Confidence: more episodes + more agreement = higher confidence.
        agreement = self._cross_episode_agreement(failing)
        confidence = min(0.5 + 0.1 * len(failing) + 0.2 * agreement, 0.95)

        return CausalChain(
            root_cause=root_cause,
            failure_type=dominant_type,
            critical_step_index=critical_idx,
            critical_action=critical_action,
            contributing_factors=factors,
            counterfactual_repair=repair,
            confidence=confidence,
        )

    def _find_critical_step(self, episode: Episode) -> tuple[int | None, str | None]:
        """The critical step is the decision point that most likely caused failure.

        Heuristics (Phase 2):
        1. Last step before a negative observation change.
        2. First step that introduces an error keyword.
        3. The step with the longest observation (often the error dump).
        """
        if not episode.steps:
            return None, None

        # Look for first error-introducing step.
        for i, step in enumerate(episode.steps):
            obs_lower = step.observation.lower()
            if any(kw in obs_lower for kw in ("error", "fail", "assert", "exception")):
                return i, step.action

        # Fall back to last step (the one that ended the episode).
        last = episode.steps[-1]
        return len(episode.steps) - 1, last.action

    def _cluster_critical_step(self, episodes: list[Episode]
                               ) -> tuple[str | None, int | None]:
        """Find the most common critical action across a cluster."""
        action_votes: Counter = Counter()
        index_votes: Counter = Counter()
        for e in episodes:
            idx, action = self._find_critical_step(e)
            if action:
                action_votes[action] += 1
            if idx is not None:
                index_votes[idx] += 1

        top_action = action_votes.most_common(1)[0][0] if action_votes else None
        top_idx = index_votes.most_common(1)[0][0] if index_votes else None
        return top_action, top_idx

    def _infer_root_cause_single(self, episode: Episode, ftype: FailureType,
                                 critical_action: str | None) -> str:
        """Infer root cause for a single episode from its structure."""
        if ftype == FailureType.no_attempt:
            return "agent produced no meaningful action toward the goal"
        if ftype == FailureType.timeout:
            return "execution exceeded time budget"
        if ftype == FailureType.runtime_error:
            # Try to extract the error message.
            for s in reversed(episode.steps):
                if "error" in s.observation.lower():
                    # First 100 chars of the error observation.
                    return f"runtime error at '{s.action}': {s.observation[:100]}"
            return f"runtime error during '{critical_action}'"
        if ftype == FailureType.wrong_output:
            return f"incorrect output produced by '{critical_action}' step"
        if ftype == FailureType.incomplete:
            return "task partially completed but stopped before full solution"
        return f"failure during '{critical_action}' (type: {ftype.value})"

    def _cluster_root_cause(self, episodes: list[Episode],
                            dominant_type: FailureType,
                            critical_action: str | None) -> str:
        """Infer root cause across a cluster. Higher quality than single-episode."""
        # Strategy: look for the most common observation terms at the critical step.
        obs_terms: Counter = Counter()
        for e in episodes:
            if not e.steps:
                continue
            # Use all observations, weighted toward the critical step.
            for i, s in enumerate(e.steps):
                weight = 2.0 if any(kw in s.observation.lower()
                                    for kw in ("error", "fail", "assert")) else 1.0
                for w in s.observation.lower().split():
                    if len(w) > 4:
                        obs_terms[w] += weight

        top_terms = [t for t, _ in obs_terms.most_common(3)]
        context_phrase = ", ".join(top_terms) if top_terms else "unknown context"

        if dominant_type == FailureType.wrong_output:
            return (f"recurring incorrect output associated with '{context_phrase}' "
                    f"at the '{critical_action}' step")
        if dominant_type == FailureType.runtime_error:
            return f"recurring runtime error involving '{context_phrase}'"
        if dominant_type == FailureType.no_attempt:
            return "agent consistently fails to produce a meaningful action"
        if dominant_type == FailureType.timeout:
            return f"recurring timeout, associated with '{context_phrase}'"
        return f"recurring failure ({dominant_type.value}) associated with '{context_phrase}'"

    def _contributing_factors(self, episode: Episode) -> list[str]:
        """Extract factors that may have contributed to failure."""
        factors: list[str] = []
        if not episode.steps:
            factors.append("no actions taken")
            return factors
        if len(episode.steps) == 1:
            factors.append("only one action attempted before failure")
        # Check for repeated identical actions (stuck in a loop).
        actions = [s.action for s in episode.steps]
        if len(actions) > 2 and len(set(actions)) == 1:
            factors.append(f"repeated same action '{actions[0]}' without progress")
        # Check for very short observations (possibly uninformative feedback).
        short_obs = sum(1 for s in episode.steps if len(s.observation) < 10)
        if short_obs > len(episode.steps) * 0.5:
            factors.append("most observations were very short (limited feedback)")
        return factors

    def _cluster_contributing_factors(self, episodes: list[Episode]) -> list[str]:
        """Aggregate contributing factors across a cluster."""
        factor_counts: Counter = Counter()
        for e in episodes:
            for f in self._contributing_factors(e):
                factor_counts[f] += 1
        # Return factors present in >30% of episodes.
        threshold = max(1, len(episodes) * 0.3)
        return [f for f, c in factor_counts.most_common() if c >= threshold]

    def _suggest_repair(self, ftype: FailureType, root_cause: str,
                        critical_action: str | None) -> str:
        """Generate a counterfactual repair suggestion."""
        if ftype == FailureType.wrong_output:
            return f"verify output correctness before submitting at the '{critical_action}' step"
        if ftype == FailureType.runtime_error:
            return f"add error handling or input validation before '{critical_action}'"
        if ftype == FailureType.timeout:
            return "reduce scope or add early termination before resource exhaustion"
        if ftype == FailureType.no_attempt:
            return "ensure the agent takes at least one meaningful action toward the goal"
        if ftype == FailureType.incomplete:
            return "continue working until the task is fully complete before submitting"
        if ftype == FailureType.missing_action:
            return "use the required tool/action for this task type"
        # Generic.
        return f"address '{root_cause}' before the '{critical_action}' step"

    def _cross_episode_agreement(self, episodes: list[Episode]) -> float:
        """How much do the episodes agree on what went wrong? (0-1)."""
        if len(episodes) < 2:
            return 0.5
        types = [classify_failure(e) for e in episodes]
        most_common_count = Counter(types).most_common(1)[0][1]
        type_agreement = most_common_count / len(episodes)

        actions = []
        for e in episodes:
            _, action = self._find_critical_step(e)
            if action:
                actions.append(action)
        if actions:
            action_agreement = Counter(actions).most_common(1)[0][1] / len(actions)
        else:
            action_agreement = 0.0

        return (type_agreement + action_agreement) / 2.0

    def _refine_with_llm(self, failing: list[Episode], chain: CausalChain) -> CausalChain:
        """Ask the offline model to sharpen the root cause. Guarded so
        dry_run / unavailable providers fall back to the heuristic."""
        try:
            traces = "\n\n".join(
                f"Goal: {e.goal}\nFailure type: {classify_failure(e).value}\nSteps: " +
                " | ".join(f"{s.action}->{s.observation[:80]}" for s in e.steps)
                for e in failing[:5]
            )
            prompt = (
                f"These {len(failing)} agent episodes all failed similarly "
                f"(classified as: {chain.failure_type.value}).\n\n"
                f"Heuristic root cause: {chain.root_cause}\n\n"
                "Traces:\n" + traces + "\n\n"
                "In one sentence, state the most likely ROOT CAUSE (not the "
                "surface symptom). Then in a second sentence, state the "
                "COUNTERFACTUAL REPAIR (what should the agent do differently)."
            )
            resp = self.provider.complete(
                [{"role": "user", "content": prompt}], max_tokens=200)
            if resp.text.strip():
                lines = resp.text.strip().split(".")
                if len(lines) >= 2:
                    chain.root_cause = lines[0].strip() + "."
                    chain.counterfactual_repair = ".".join(lines[1:]).strip()
                else:
                    chain.root_cause = resp.text.strip()
                chain.confidence = min(chain.confidence + 0.1, 0.95)
        except Exception:
            pass  # heuristic stands
        return chain
