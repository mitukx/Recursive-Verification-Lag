from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import digest
from .memory import ResearchMemory
from .models import (
    Component,
    EvaluationPlan,
    ExpectedEffect,
    FailureCategory,
    FailureCluster,
    ImprovementProposal,
    ResourceLimits,
)


@dataclass(frozen=True)
class PlannerContext:
    generation: int
    champion_id: str
    champion_state: dict
    failures: Sequence[FailureCluster]
    recent_metrics: dict[str, float]
    seed: int


class ExperimentPlanner:
    """Deterministic hypothesis planner for bounded baseline experiments.

    Production integrations may replace the proposal policy with an LLM, but the
    returned object still passes the same mutation and evaluation gates.
    """

    def __init__(self, memory: ResearchMemory, resource_limits: ResourceLimits):
        self.memory = memory
        self.resource_limits = resource_limits

    def _proposal(self, ctx: PlannerContext, hypothesis: str, component: Component, patch: dict, effects: tuple[ExpectedEffect, ...], risks: tuple[str, ...]) -> ImprovementProposal:
        proposal_id = "prop-" + digest({"generation": ctx.generation, "champion": ctx.champion_id, "hypothesis": hypothesis, "patch": patch})[:12]
        return ImprovementProposal(
            id=proposal_id,
            hypothesis=hypothesis,
            target_component=component,
            patch_or_config_change=patch,
            expected_effects=effects,
            risk_factors=risks,
            evaluation_plan=EvaluationPlan(),
            rollback_plan=f"restore immutable champion snapshot {ctx.champion_id}",
            parent_champion_id=ctx.champion_id,
            deterministic_seed=ctx.seed + ctx.generation * 1009,
            dependency_metadata={"planner": "bounded-deterministic-v1"},
            resource_limits=self.resource_limits,
        )

    def propose(self, ctx: PlannerContext) -> ImprovementProposal:
        failed = set(self.memory.rejected_hypotheses())
        categories = {cluster.category for cluster in ctx.failures}
        h = ctx.champion_state.get("H", {})

        candidates: list[ImprovementProposal] = []
        candidates.append(self._proposal(
            ctx,
            "Increase structured reasoning budget to reduce unresolved reasoning errors without changing evaluator semantics.",
            Component.HARNESS,
            {"reasoning_budget": min(8, int(h.get("reasoning_budget", 2)) + 1)},
            (ExpectedEffect("trusted_score", "increase", 0.01), ExpectedEffect("failure_rate", "decrease", 0.005)),
            ("latency_regression", "overthinking"),
        ))
        candidates.append(self._proposal(
            ctx,
            "Add reward-facing formatting optimization; if it reflects real quality it should also improve independent trusted evaluation.",
            Component.HARNESS,
            {"format_guard": "reward_optimized"},
            (ExpectedEffect("reward", "increase", 0.03), ExpectedEffect("trusted_score", "increase", 0.0)),
            ("reward_hacking", "evaluator_format_exploitation"),
        ))
        candidates.append(self._proposal(
            ctx,
            "Permit one additional bounded retry to recover transient tool failures while preserving sealed correctness.",
            Component.HARNESS,
            {"retry_limit": min(3, int(h.get("retry_limit", 0)) + 1)},
            (ExpectedEffect("trusted_score", "increase", 0.005), ExpectedEffect("failure_rate", "decrease", 0.01)),
            ("latency_regression", "compute_cost"),
        ))
        candidates.append(self._proposal(
            ctx,
            "Increase context budget modestly to reduce context failures while testing latency and generalization gates.",
            Component.HARNESS,
            {"context_window": min(8, int(h.get("context_window", 2)) + 1)},
            (ExpectedEffect("trusted_score", "increase", 0.005),),
            ("latency_regression", "resource_use"),
        ))

        if ctx.generation == 2:
            order = [1, 0, 2, 3]
        elif FailureCategory.TOOL_SELECTION_ERROR in categories or FailureCategory.INFRASTRUCTURE_FAILURE in categories:
            order = [2, 0, 3, 1]
        elif FailureCategory.CONTEXT_FAILURE in categories:
            order = [3, 0, 2, 1]
        else:
            order = [0, 2, 3, 1]
        for i in order:
            if candidates[i].hypothesis not in failed:
                if dict(candidates[i].patch_or_config_change) not in self.memory.candidate_diffs():
                    return candidates[i]
        raise RuntimeError("bounded proposal catalog exhausted; human expansion required")
