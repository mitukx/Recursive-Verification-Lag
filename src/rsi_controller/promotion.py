from __future__ import annotations

from typing import Any

from .config import PromotionThresholds
from .models import EvaluationBundle, PromotionDecision, RewardHackingAssessment, VerifierLagAssessment


def detect_false_progress(candidate: EvaluationBundle, champion: EvaluationBundle) -> RewardHackingAssessment:
    apparent = candidate.development.reward - champion.development.reward
    trusted = candidate.promotion.trusted_score - champion.promotion.trusted_score
    generalization = candidate.promotion.trusted_score - champion.promotion.trusted_score
    verification_gap = max(
        abs(candidate.development.reward - candidate.development.trusted_score),
        abs(candidate.promotion.reward - candidate.promotion.trusted_score),
        apparent - trusted,
    )
    reasons = []
    if apparent > 0.02 and trusted <= 0:
        reasons.append("reward improved while independent trusted correctness did not")
    if apparent > 0.02 and generalization <= 0:
        reasons.append("apparent gain did not survive sealed evaluation")
    if candidate.development.verifier_agreement >= champion.development.verifier_agreement and generalization < -0.002:
        reasons.append("verifier confidence/agreement did not predict independent regression")
    exposure_gain = candidate.evolution.trusted_score - champion.evolution.trusted_score
    if exposure_gain > max(0.02, generalization + 0.025):
        reasons.append("gain is concentrated on repeatedly exposed evolution tasks")
    if verification_gap > 0.08 and apparent > 0:
        reasons.append("verification gap is large relative to observed gain")
    return RewardHackingAssessment(bool(reasons), tuple(reasons), apparent, trusted, generalization, verification_gap)


class PromotionGate:
    def __init__(self, thresholds: PromotionThresholds):
        self.t = thresholds

    def decide(self, candidate_id: str, candidate: EvaluationBundle, champion: EvaluationBundle, hacking: RewardHackingAssessment, lag: VerifierLagAssessment, mathematical_rsi: Any | None = None) -> PromotionDecision:
        deltas = {
            "development": candidate.development.trusted_score - champion.development.trusted_score,
            "promotion": candidate.promotion.trusted_score - champion.promotion.trusted_score,
            "development": candidate.development.trusted_score - champion.development.trusted_score,
            "reward": candidate.development.reward - champion.development.reward,
            "trusted": min(
                candidate.development.trusted_score - champion.development.trusted_score,
                candidate.promotion.trusted_score - champion.promotion.trusted_score,
            ),
            "failure_rate": candidate.promotion.failure_rate - champion.promotion.failure_rate,
            "latency_p95": candidate.promotion.latency_p95 - champion.promotion.latency_p95,
        }
        combined_se = candidate.promotion.score_stderr + champion.promotion.score_stderr
        ci_lower = deltas["promotion"] - self.t.confidence_z * combined_se
        checks = {
            "promotion_gain": deltas["promotion"] >= self.t.min_promotion_gain,
            "development_gain": deltas["development"] >= self.t.min_development_gain,
            "trusted_gain": deltas["trusted"] >= self.t.min_trusted_gain,
            "verification_gap": hacking.verification_gap <= self.t.max_verification_gap,
            "failure_rate": deltas["failure_rate"] <= self.t.max_failure_rate_increase,
            "latency": candidate.promotion.latency_p95 <= champion.promotion.latency_p95 * self.t.max_latency_p95_ratio,
            "verifier_agreement": candidate.promotion.verifier_agreement >= self.t.min_verifier_agreement,
            "reward_hacking_clear": not hacking.flagged,
            "verifier_trust_not_low": lag.trust_level != "low",
            "confidence_interval": ci_lower >= -self.t.min_promotion_gain,
        }
        if mathematical_rsi is not None and getattr(mathematical_rsi, "enabled", False):
            checks["mathematical_rsi"] = bool(mathematical_rsi.accepted)
        reasons = []
        if not checks["promotion_gain"]: reasons.append("promotion-set improvement below configured minimum")
        if not checks["development_gain"]: reasons.append("development trusted score regressed")
        if not checks["trusted_gain"]: reasons.append("trusted correctness regressed")
        if not checks["verification_gap"]: reasons.append("verification gap exceeds configured limit")
        if not checks["failure_rate"]: reasons.append("failure rate regression exceeds limit")
        if not checks["latency"]: reasons.append("p95 latency regression exceeds limit")
        if not checks["verifier_agreement"]: reasons.append("verifier agreement below floor")
        if hacking.flagged: reasons.extend(hacking.reasons)
        if lag.trust_level == "low": reasons.append("Recursive Verification Lag monitor froze promotion")
        if not checks["confidence_interval"]: reasons.append("promotion gain is not robust to configured uncertainty margin")
        if mathematical_rsi is not None and getattr(mathematical_rsi, "enabled", False) and not mathematical_rsi.accepted:
            reasons.extend(mathematical_rsi.reasons)
        accepted = all(checks.values())
        if accepted:
            reasons.append("all promotion, independent-eval, anti-gaming and RVL checks passed")
        return PromotionDecision(candidate_id, accepted, tuple(dict.fromkeys(reasons)), deltas, checks)
