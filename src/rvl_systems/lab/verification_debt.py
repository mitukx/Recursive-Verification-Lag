"""Verification-debt accounting and bounded rollout admission."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationDebtSignals:
    pending: int = 0
    verifying: int = 0
    stale_rewards: int = 0
    max_policy_lag: int = 0
    max_verifier_lag: int = 0
    oldest_unverified_age_s: float = 0.0

    def __post_init__(self):
        ints = (
            self.pending,self.verifying,self.stale_rewards,
            self.max_policy_lag,self.max_verifier_lag,
        )
        if any(v < 0 for v in ints) or self.oldest_unverified_age_s < 0:
            raise ValueError("verification debt signals must be non-negative")


@dataclass(frozen=True)
class VerificationDebtConfig:
    pending_weight: float = 1.0
    verifying_weight: float = 0.75
    stale_reward_weight: float = 1.5
    policy_lag_weight: float = 0.20
    verifier_lag_weight: float = 1.0
    age_weight_per_s: float = 0.02
    soft_limit: float = 2.5
    hard_limit: float = 5.0

    def __post_init__(self):
        weights = (
            self.pending_weight,self.verifying_weight,self.stale_reward_weight,
            self.policy_lag_weight,self.verifier_lag_weight,self.age_weight_per_s,
        )
        if any(v < 0 for v in weights):
            raise ValueError("verification debt weights must be non-negative")
        if self.soft_limit < 0 or self.hard_limit <= self.soft_limit:
            raise ValueError("verification debt limits must satisfy 0 <= soft < hard")


@dataclass(frozen=True)
class VerificationDebtAssessment:
    score: float
    level: str
    action: str
    signals: VerificationDebtSignals


class VerificationDebtController:
    """Conservative rollout admission controller.

    This is a systems backpressure heuristic, not a statistical safety
    certificate. It prevents rollout production from outrunning verification
    indefinitely and exposes a scalar only for control/telemetry.
    """

    def __init__(self, config: VerificationDebtConfig | None = None):
        self.config = config or VerificationDebtConfig()

    def score(self, signals: VerificationDebtSignals) -> float:
        c = self.config
        return (
            c.pending_weight*signals.pending
            + c.verifying_weight*signals.verifying
            + c.stale_reward_weight*signals.stale_rewards
            + c.policy_lag_weight*signals.max_policy_lag
            + c.verifier_lag_weight*signals.max_verifier_lag
            + c.age_weight_per_s*signals.oldest_unverified_age_s
        )

    def assess(self, signals: VerificationDebtSignals) -> VerificationDebtAssessment:
        score = self.score(signals)
        if score >= self.config.hard_limit:
            return VerificationDebtAssessment(score,"high","pause_generation",signals)
        if score >= self.config.soft_limit:
            return VerificationDebtAssessment(score,"elevated","throttle_generation",signals)
        return VerificationDebtAssessment(score,"normal","admit_generation",signals)
