"""Minimal RLVR systems stack for Recursive Verification Lag experiments."""

from .grpo import GRPOConfig, compute_group_advantages
from .pipeline import RLVRPipeline, PipelineConfig
from .refresh import AdaptiveRefreshController
from .rollout import AsyncRolloutEngine, RolloutRequest
from .telemetry import Telemetry
from .types import Generation, VerifiedGeneration
from .verifier import ExactMatchVerifier

__all__ = [
    "AdaptiveRefreshController",
    "AsyncRolloutEngine",
    "ExactMatchVerifier",
    "GRPOConfig",
    "Generation",
    "PipelineConfig",
    "RLVRPipeline",
    "RolloutRequest",
    "Telemetry",
    "VerifiedGeneration",
    "compute_group_advantages",
]
