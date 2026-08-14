"""Skill Compiler (Phase 4, proposal 8.8).

Converts repeated successful behaviors into reusable skills. A skill is more
than advice — it's a structured workflow with preconditions, ordered steps,
and postconditions that the agent can reuse without reasoning from scratch.

Induction criteria:
  - Multiple successful episodes share a common action sequence
  - The sequence recurs across different variants in the same family
  - Confidence is above threshold

The compiler works OFFLINE during consolidation, alongside the experience
inducer. While the inducer extracts LESSONS from failures, the compiler
extracts PROCEDURES from successes.
"""
from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from typing import Any

from schemas import Episode, SkillObject
from schemas.experience import ValidationStatus
from .confidence import compute_confidence


# Minimum successful episodes sharing a workflow before we compile.
MIN_SKILL_EVIDENCE = 3


class SkillCompiler:
    """Induces reusable skills from clusters of successful episodes."""

    def __init__(self, min_evidence: int = MIN_SKILL_EVIDENCE,
                 min_confidence: float = 0.5) -> None:
        self.min_evidence = min_evidence
        self.min_confidence = min_confidence

    def compile_skills(self, episodes: list[Episode],
                       existing_skills: list[SkillObject] | None = None
                       ) -> list[SkillObject]:
        """Find recurring successful workflows and compile them into skills.

        Returns newly compiled skills (does not include existing ones).
        """
        existing = existing_skills or []
        existing_ids = {s.skill_id for s in existing}

        # Group successes by task family.
        family_successes: dict[str, list[Episode]] = defaultdict(list)
        for ep in episodes:
            if ep.outcome and ep.outcome.task_success >= 1.0:
                family_successes[ep.task_family].append(ep)

        new_skills: list[SkillObject] = []
        for family, eps in family_successes.items():
            if len(eps) < self.min_evidence:
                continue

            # Find common action workflows.
            workflows = self._extract_workflows(eps)
            for workflow_key, (workflow_eps, steps) in workflows.items():
                if len(workflow_eps) < self.min_evidence:
                    continue

                # Check if this workflow is already captured in an existing skill.
                if self._already_compiled(steps, existing, family):
                    continue

                skill = self._induce_skill(family, workflow_eps, steps)
                if skill.confidence >= self.min_confidence:
                    new_skills.append(skill)

        return new_skills

    def validate_skill(self, skill: SkillObject, replay_fn) -> bool:
        """Validate a skill by replaying with its workflow injected.

        replay_fn(skill) -> (baseline_success, with_skill_success)
        """
        if replay_fn is None:
            skill.validation_status = ValidationStatus.candidate
            return False

        baseline, improved = replay_fn(skill)
        delta = improved - baseline
        if delta > 0:
            skill.validation_status = ValidationStatus.active
            return True
        skill.validation_status = ValidationStatus.candidate
        return False

    def reinforce_skill(self, skill: SkillObject, episode: Episode) -> None:
        """Strengthen a skill with additional evidence from a new success."""
        skill.evidence_count += 1
        if episode.episode_id not in skill.source_episodes:
            skill.source_episodes.append(episode.episode_id)
        skill.confidence = min(0.95, skill.confidence + 0.02)

    def _extract_workflows(self, episodes: list[Episode]
                           ) -> dict[str, tuple[list[Episode], list[str]]]:
        """Group episodes by their action sequence (workflow fingerprint)."""
        workflows: dict[str, tuple[list[Episode], list[str]]] = {}
        for ep in episodes:
            if not ep.steps:
                continue
            # Workflow = ordered sequence of actions.
            actions = [s.action for s in ep.steps if s.action]
            if not actions:
                continue
            key = "|".join(actions)
            if key not in workflows:
                workflows[key] = ([], actions)
            workflows[key][0].append(ep)
        return workflows

    def _induce_skill(self, family: str, episodes: list[Episode],
                      steps: list[str]) -> SkillObject:
        """Create a SkillObject from a cluster of successful episodes."""
        # Extract preconditions from shared context.
        preconditions = self._extract_preconditions(episodes)
        # Extract postconditions from outcomes.
        postconditions = self._extract_postconditions(episodes)
        # Compute confidence.
        confidence = compute_confidence(
            evidence_count=len(episodes),
            consistency=1.0,  # all are successes
            contradictions=0,
        )
        # Build human-readable workflow description.
        workflow = self._describe_workflow(steps, episodes)

        return SkillObject(
            skill_id=f"skill_{uuid.uuid4().hex[:8]}",
            name=self._name_skill(family, steps),
            description=f"Workflow for {family} tasks using {len(steps)} step(s).",
            scope=family,
            preconditions=preconditions,
            workflow=workflow,
            postconditions=postconditions,
            source_experiences=[],
            source_episodes=[ep.episode_id for ep in episodes],
            confidence=confidence,
            evidence_count=len(episodes),
            validation_status=ValidationStatus.candidate,
        )

    def _extract_preconditions(self, episodes: list[Episode]) -> list[str]:
        """Infer when this workflow should be applied."""
        # Common initial state keys across successful episodes.
        if not episodes:
            return []
        common_keys = set(episodes[0].initial_state.keys())
        for ep in episodes[1:]:
            common_keys &= set(ep.initial_state.keys())
        conditions: list[str] = []
        for k in sorted(common_keys):
            vals = {str(ep.initial_state.get(k)) for ep in episodes}
            if len(vals) == 1:
                conditions.append(f"{k}={list(vals)[0]}")
        if episodes[0].task_family:
            conditions.insert(0, f"task_family={episodes[0].task_family}")
        return conditions or ["applicable"]

    def _extract_postconditions(self, episodes: list[Episode]) -> list[str]:
        """What should be true after the skill executes successfully."""
        conditions = ["task_success >= 1.0"]
        # Check if all episodes had specific outcome patterns.
        if all(ep.outcome and ep.outcome.efficiency >= 0.8 for ep in episodes):
            conditions.append("efficiency >= 0.8")
        return conditions

    def _describe_workflow(self, actions: list[str],
                           episodes: list[Episode]) -> list[str]:
        """Turn raw action names into descriptive workflow steps."""
        # Enrich action names with observation context from episodes.
        descriptions: list[str] = []
        for i, action in enumerate(actions):
            # Collect observations at this step across episodes.
            obs_words: Counter = Counter()
            for ep in episodes:
                if i < len(ep.steps):
                    for w in ep.steps[i].observation.lower().split():
                        if len(w) > 3:
                            obs_words[w] += 1
            top_context = [w for w, _ in obs_words.most_common(2)]
            context_hint = f" ({', '.join(top_context)})" if top_context else ""
            descriptions.append(f"{action}{context_hint}")
        return descriptions

    @staticmethod
    def _name_skill(family: str, steps: list[str]) -> str:
        """Generate a human-readable skill name."""
        if len(steps) == 1:
            return f"{family}_{steps[0]}"
        return f"{family}_{steps[0]}_to_{steps[-1]}"

    @staticmethod
    def _already_compiled(steps: list[str], existing: list[SkillObject],
                          family: str) -> bool:
        """Check if an equivalent skill already exists."""
        for skill in existing:
            if skill.scope != family:
                continue
            # Compare workflow steps (actions, not descriptions).
            existing_actions = [s.split(" (")[0] for s in skill.workflow]
            if existing_actions == steps:
                return True
        return False
