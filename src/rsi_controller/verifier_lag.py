from __future__ import annotations

from .config import VerifierTrustThresholds
from .models import EvaluationBundle, VerifierLagAssessment


class RecursiveVerificationLagMonitor:
    def __init__(self, thresholds: VerifierTrustThresholds):
        self.t = thresholds

    def assess(self, candidate: EvaluationBundle, champion: EvaluationBundle) -> VerifierLagAssessment:
        disagreement = max(0.0, 1.0 - candidate.promotion.verifier_agreement)
        trusted_disagreement = abs(candidate.promotion.reward - candidate.promotion.trusted_score)
        reward_truth_gap = max(
            abs(candidate.development.reward - candidate.development.trusted_score),
            abs(candidate.promotion.reward - candidate.promotion.trusted_score),
        )
        policy_gain = candidate.promotion.trusted_score - champion.promotion.trusted_score
        shift = abs(candidate.development.trusted_score - champion.development.trusted_score) + 0.025 * candidate.policy_verifier_age
        age = candidate.policy_verifier_age
        verifier_refresh_rate = 1.0 / (1.0 + age)

        low_reasons = []
        uncertain_reasons = []
        if disagreement >= self.t.low_disagreement:
            low_reasons.append("verifier disagreement exceeds low-trust threshold")
        elif disagreement >= self.t.uncertain_disagreement:
            uncertain_reasons.append("verifier disagreement elevated")
        if reward_truth_gap >= self.t.low_reward_truth_gap:
            low_reasons.append("reward/trusted divergence exceeds low-trust threshold")
        elif reward_truth_gap >= self.t.uncertain_reward_truth_gap:
            uncertain_reasons.append("reward/trusted divergence elevated")
        if shift >= self.t.low_distribution_shift:
            low_reasons.append("policy/verifier distribution shift too large")
        elif shift >= self.t.uncertain_distribution_shift:
            uncertain_reasons.append("policy/verifier distribution shift elevated")
        if age >= self.t.low_stale_age:
            low_reasons.append("verifier version is too stale")
        elif age >= self.t.uncertain_stale_age:
            uncertain_reasons.append("verifier refresh is becoming stale")

        if low_reasons:
            trust = "low"
            intervention = "freeze_promotion_acquire_trusted_labels_refresh_and_reevaluate"
            reasons = tuple(low_reasons + uncertain_reasons)
        elif uncertain_reasons:
            trust = "uncertain"
            intervention = "increase_trusted_evaluation_reduce_promotion_rate_and_refresh_verifier"
            reasons = tuple(uncertain_reasons)
        else:
            trust = "high"
            intervention = "continue_normal_improvement"
            reasons = ("all configured verifier-trust diagnostics within bounds",)
        return VerifierLagAssessment(
            trust_level=trust,
            policy_improvement_rate=policy_gain,
            verifier_refresh_rate=verifier_refresh_rate,
            verifier_disagreement_rate=disagreement,
            trusted_label_disagreement=trusted_disagreement,
            reward_ground_truth_divergence=reward_truth_gap,
            distribution_shift=shift,
            stale_verifier_age=age,
            intervention=intervention,
            reasons=reasons,
        )
