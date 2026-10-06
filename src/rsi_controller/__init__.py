"""Bounded recursive self-improvement experiment controller.

The controller intentionally keeps promotion rules, sealed evaluations, resource
ceilings and audit logs outside candidate mutation surfaces.
"""

from .models import (
    Candidate,
    ChampionSnapshot,
    FailureCategory,
    FailureCluster,
    GenerationRecord,
    ImprovementProposal,
    PromotionDecision,
    RSIMode,
)

__all__ = [
    "Candidate", "ChampionSnapshot", "FailureCategory", "FailureCluster",
    "GenerationRecord", "ImprovementProposal", "PromotionDecision", "RSIMode",
]
