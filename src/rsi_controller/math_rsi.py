from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .config import MathematicalRSIConfig, PromotionThresholds
from .models import Candidate, EvaluationBundle, RewardHackingAssessment, VerifierLagAssessment


@dataclass(frozen=True)
class ConstraintResult:
    name: str
    passed: bool
    critical: bool
    value: float
    threshold: float
    relation: str


@dataclass(frozen=True)
class CodedVerificationAssessment:
    passed: bool
    pass_fraction: float
    allowed_noncritical_failures: int
    failed_constraints: tuple[str, ...]
    critical_failures: tuple[str, ...]
    constraints: tuple[ConstraintResult, ...]


@dataclass(frozen=True)
class ProbeCoverageAssessment:
    covered: bool
    complexity: int
    max_complexity: int
    probe_ids: tuple[str, ...]
    uncovered_mutations: tuple[str, ...]


@dataclass(frozen=True)
class InformationBudgetAssessment:
    passed: bool
    effective_dimension: int
    target_error: float
    required_trusted_samples: int
    available_trusted_samples: int
    scaling_proxy: float


@dataclass(frozen=True)
class ReconstructionAssessment:
    enforced: bool
    passed: bool
    branching_factor: int
    lambda_hat: float
    criticality: float
    threshold: float
    interpretation: str


@dataclass(frozen=True)
class MathematicalRSIAssessment:
    enabled: bool
    accepted: bool
    checks: dict[str, bool]
    reasons: tuple[str, ...]
    coded: CodedVerificationAssessment
    probes: ProbeCoverageAssessment
    information_budget: InformationBudgetAssessment
    reconstruction: ReconstructionAssessment

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# These are fixed *engineering probes*, not a theorem-level universal hitting set.
# The #116 result motivates the contract: every bounded mutation must declare a
# fixed probe family before optimization starts, and leaving the covered class
# forces rejection/refresh rather than silently reusing old verification.
_FIXED_PROBES: dict[tuple[str, str], tuple[str, ...]] = {
    ("H", "reasoning_budget"): ("reasoning-heldout", "long-chain-consistency"),
    ("H", "retry_limit"): ("tool-recovery", "duplicate-action"),
    ("H", "context_window"): ("context-overflow", "distractor-context"),
    ("H", "memory_slots"): ("memory-retrieval", "stale-memory"),
    ("H", "tool_policy"): ("tool-selection", "tool-side-effect"),
    ("H", "sampling_temperature"): ("sampling-stability", "tail-risk"),
    ("H", "curriculum_level"): ("difficulty-transfer", "curriculum-regression"),
    ("H", "format_guard"): ("format-invariance", "reward-hacking"),
    ("F", "learning_rate"): ("update-stability", "heldout-regression"),
    ("F", "batch_size"): ("batch-sensitivity", "heldout-regression"),
    ("F", "adapter_rank"): ("capacity-shift", "heldout-regression"),
    ("F", "clip_ratio"): ("policy-ratio", "off-policy-stability"),
    ("V", "threshold"): ("threshold-sweep", "calibration"),
    ("V", "ensemble_size"): ("verifier-disagreement", "ensemble-ablation"),
    ("V", "refresh_cadence"): ("staleness-sweep", "freshness-control"),
    ("V", "trusted_fraction"): ("audit-budget", "trusted-label-subsample"),
    ("theta", "adapter_checkpoint"): ("checkpoint-heldout", "capability-regression"),
    ("theta", "policy_checkpoint"): ("checkpoint-heldout", "capability-regression"),
}


def _component_section(candidate: Candidate) -> str:
    return {
        "harness": "H",
        "training": "F",
        "verifier": "V",
        "policy": "theta",
    }[candidate.target_component.value]


def _relation(value: float, threshold: float, relation: str) -> bool:
    if relation == ">=":
        return value >= threshold
    if relation == "<=":
        return value <= threshold
    raise ValueError(f"unknown relation: {relation}")


class MathematicalRSIGate:
    """OAI-math-inspired bounded RSI gate.

    The gate deliberately separates theorem-derived *design principles* from
    theorem guarantees. None of the OpenAI Math theorems directly proves this
    controller safe. The implementation uses:
      - #136 PCP-for-PPAD: redundant local obligations with a bounded fraction
        of noncritical failures while critical invariants remain mandatory;
      - #116 universal hitting: a fixed probe registry for a bounded mutation
        class, with fail-closed behavior outside that class;
      - #140 memory/sample lower bounds: d*log(1/epsilon) as an explicitly
        heuristic trusted-information budget proxy;
      - #229 reconstruction threshold: d*lambda^2 as a diagnostic or enforced
        gate only when the experiment explicitly declares the tree-channel
        assumptions applicable.
    """

    def __init__(self, config: MathematicalRSIConfig, promotion: PromotionThresholds):
        self.cfg = config
        self.promotion = promotion

    def _coded(
        self,
        candidate: EvaluationBundle,
        champion: EvaluationBundle,
        hacking: RewardHackingAssessment,
        lag: VerifierLagAssessment,
    ) -> CodedVerificationAssessment:
        p = self.promotion
        latency_ratio = candidate.promotion.latency_p95 / max(
            champion.promotion.latency_p95, 1e-12
        )
        throughput_ratio = candidate.promotion.throughput / max(
            champion.promotion.throughput, 1e-12
        )
        constraints = (
            ConstraintResult(
                "promotion_trusted_nonregression",
                candidate.promotion.trusted_score >= champion.promotion.trusted_score,
                True,
                candidate.promotion.trusted_score - champion.promotion.trusted_score,
                0.0,
                ">=",
            ),
            ConstraintResult(
                "development_trusted_nonregression",
                candidate.development.trusted_score >= champion.development.trusted_score,
                True,
                candidate.development.trusted_score - champion.development.trusted_score,
                0.0,
                ">=",
            ),
            ConstraintResult(
                "reward_hacking_clear",
                not hacking.flagged,
                True,
                0.0 if hacking.flagged else 1.0,
                1.0,
                ">=",
            ),
            ConstraintResult(
                "rvl_not_low",
                lag.trust_level != "low",
                True,
                0.0 if lag.trust_level == "low" else 1.0,
                1.0,
                ">=",
            ),
            ConstraintResult(
                "evolution_transfer_nonregression",
                candidate.evolution.trusted_score
                >= champion.evolution.trusted_score - self.cfg.local_slack,
                False,
                candidate.evolution.trusted_score - champion.evolution.trusted_score,
                -self.cfg.local_slack,
                ">=",
            ),
            ConstraintResult(
                "promotion_verifier_agreement",
                candidate.promotion.verifier_agreement >= p.min_verifier_agreement,
                False,
                candidate.promotion.verifier_agreement,
                p.min_verifier_agreement,
                ">=",
            ),
            ConstraintResult(
                "verification_gap",
                hacking.verification_gap <= p.max_verification_gap,
                False,
                hacking.verification_gap,
                p.max_verification_gap,
                "<=",
            ),
            ConstraintResult(
                "failure_rate_delta",
                candidate.promotion.failure_rate - champion.promotion.failure_rate
                <= p.max_failure_rate_increase,
                False,
                candidate.promotion.failure_rate - champion.promotion.failure_rate,
                p.max_failure_rate_increase,
                "<=",
            ),
            ConstraintResult(
                "latency_ratio",
                latency_ratio <= p.max_latency_p95_ratio,
                False,
                latency_ratio,
                p.max_latency_p95_ratio,
                "<=",
            ),
            ConstraintResult(
                "throughput_floor",
                throughput_ratio >= self.cfg.min_throughput_ratio,
                False,
                throughput_ratio,
                self.cfg.min_throughput_ratio,
                ">=",
            ),
        )
        critical_failures = tuple(c.name for c in constraints if c.critical and not c.passed)
        redundant = [c for c in constraints if not c.critical]
        redundant_failures = [c.name for c in redundant if not c.passed]
        allowed = math.floor(self.cfg.max_corrupt_fraction * len(redundant))
        passed = not critical_failures and len(redundant_failures) <= allowed
        pass_fraction = sum(c.passed for c in constraints) / len(constraints)
        return CodedVerificationAssessment(
            passed,
            pass_fraction,
            allowed,
            tuple(c.name for c in constraints if not c.passed),
            critical_failures,
            constraints,
        )

    def _probes(self, candidate: Candidate) -> ProbeCoverageAssessment:
        section = _component_section(candidate)
        diff = dict(candidate.diff)
        code_patch = diff.pop("__code_patch__", None)
        complexity = len(diff)
        uncovered: list[str] = []
        probes: set[str] = set()

        if code_patch is not None:
            # Candidate-controlled code is intentionally outside the fixed
            # bounded probe class until a separate theorem/coverage contract is
            # supplied for that code surface.
            files = code_patch if isinstance(code_patch, dict) else {}
            complexity += len(files)
            uncovered.append(f"{section}.__code_patch__")

        for key in sorted(diff):
            probe_ids = _FIXED_PROBES.get((section, key))
            if probe_ids is None:
                uncovered.append(f"{section}.{key}")
            else:
                probes.update(probe_ids)

        if complexity > self.cfg.max_probe_complexity:
            uncovered.append(
                f"complexity:{complexity}>{self.cfg.max_probe_complexity}"
            )
        return ProbeCoverageAssessment(
            covered=not uncovered,
            complexity=complexity,
            max_complexity=self.cfg.max_probe_complexity,
            probe_ids=tuple(sorted(probes)),
            uncovered_mutations=tuple(uncovered),
        )

    def _information_budget(
        self,
        candidate: Candidate,
        evaluation: EvaluationBundle,
        probes: ProbeCoverageAssessment,
    ) -> InformationBudgetAssessment:
        # The factor d*log(1/eps) is a research proxy inspired by #140, not a
        # transferred lower bound. We expose the exact number used in every
        # decision so future work can replace this heuristic with a proved
        # reduction or a calibrated empirical law.
        d = max(
            1,
            probes.complexity
            + len(probes.uncovered_mutations)
            + (1 if candidate.target_component.value == "verifier" else 0),
        )
        eps = self.cfg.trusted_target_error
        scaling = d * math.log(1.0 / eps)
        required = max(
            self.cfg.min_trusted_samples,
            math.ceil(self.cfg.trusted_budget_multiplier * scaling),
        )
        required = min(required, self.cfg.max_trusted_samples)
        available = int(evaluation.promotion.samples)
        return InformationBudgetAssessment(
            passed=available >= required,
            effective_dimension=d,
            target_error=eps,
            required_trusted_samples=required,
            available_trusted_samples=available,
            scaling_proxy=scaling,
        )

    def _reconstruction(self, evaluation: EvaluationBundle) -> ReconstructionAssessment:
        agreement = min(
            evaluation.development.verifier_agreement,
            evaluation.promotion.verifier_agreement,
        )
        # Under a binary symmetric-channel abstraction, corr = 2*agreement-1.
        # We preserve the sign because #229 allows signed lambda, then square it.
        lam = max(-1.0, min(1.0, 2.0 * agreement - 1.0))
        d = self.cfg.reconstruction_branching_factor
        criticality = d * lam * lam
        passed = criticality > self.cfg.reconstruction_threshold
        if not self.cfg.reconstruction_assumptions_met:
            interpretation = (
                "diagnostic_only: current controller has not established the "
                "independent homogeneous tree-channel assumptions required to "
                "treat d*lambda^2 as a promotion theorem"
            )
        elif passed:
            interpretation = "declared tree-channel experiment is supercritical"
        else:
            interpretation = "declared tree-channel experiment is non-reconstructible/subcritical"
        return ReconstructionAssessment(
            enforced=self.cfg.reconstruction_assumptions_met,
            passed=passed,
            branching_factor=d,
            lambda_hat=lam,
            criticality=criticality,
            threshold=self.cfg.reconstruction_threshold,
            interpretation=interpretation,
        )

    def assess(
        self,
        candidate: Candidate,
        evaluation: EvaluationBundle,
        champion: EvaluationBundle,
        hacking: RewardHackingAssessment,
        lag: VerifierLagAssessment,
    ) -> MathematicalRSIAssessment:
        coded = self._coded(evaluation, champion, hacking, lag)
        probes = self._probes(candidate)
        budget = self._information_budget(candidate, evaluation, probes)
        reconstruction = self._reconstruction(evaluation)

        if not self.cfg.enabled:
            return MathematicalRSIAssessment(
                enabled=False,
                accepted=True,
                checks={},
                reasons=("OAI-math-inspired RSI gate disabled",),
                coded=coded,
                probes=probes,
                information_budget=budget,
                reconstruction=reconstruction,
            )

        checks = {
            "coded_verification": coded.passed if self.cfg.require_coded_verification else True,
            "bounded_probe_coverage": probes.covered if self.cfg.require_probe_coverage else True,
            "trusted_information_budget": budget.passed if self.cfg.require_information_budget else True,
            "reconstruction_criticality": (
                reconstruction.passed
                if reconstruction.enforced and self.cfg.require_reconstruction_supercritical
                else True
            ),
        }
        reasons: list[str] = []
        if not checks["coded_verification"]:
            reasons.append(
                "OAI#136-inspired coded verification failed: "
                + ",".join(coded.failed_constraints)
            )
        if not checks["bounded_probe_coverage"]:
            reasons.append(
                "OAI#116-inspired bounded probe class exceeded: "
                + ",".join(probes.uncovered_mutations)
            )
        if not checks["trusted_information_budget"]:
            reasons.append(
                "OAI#140-inspired trusted-information budget proxy unmet: "
                f"{budget.available_trusted_samples}<{budget.required_trusted_samples}"
            )
        if not checks["reconstruction_criticality"]:
            reasons.append(
                "OAI#229-inspired declared reconstruction experiment is subcritical: "
                f"d*lambda^2={reconstruction.criticality:.6f}"
            )
        if all(checks.values()):
            reasons.append(
                "all enabled OAI-math-inspired bounded RSI checks passed"
            )
        return MathematicalRSIAssessment(
            enabled=True,
            accepted=all(checks.values()),
            checks=checks,
            reasons=tuple(reasons),
            coded=coded,
            probes=probes,
            information_budget=budget,
            reconstruction=reconstruction,
        )


def reconstruction_phase_grid(
    branching_factors: tuple[int, ...],
    agreements: tuple[float, ...],
    threshold: float = 1.0,
) -> list[dict[str, float | int | bool]]:
    """Deterministic #229-inspired phase table for declared BSC tree models."""
    rows: list[dict[str, float | int | bool]] = []
    for d in branching_factors:
        if d <= 0:
            raise ValueError("branching factors must be positive")
        for agreement in agreements:
            if not 0.0 <= agreement <= 1.0:
                raise ValueError("agreement must lie in [0,1]")
            lam = 2.0 * agreement - 1.0
            criticality = d * lam * lam
            rows.append(
                {
                    "branching_factor": d,
                    "agreement": agreement,
                    "lambda": lam,
                    "criticality": criticality,
                    "supercritical": criticality > threshold,
                }
            )
    return rows
