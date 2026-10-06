from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class Component(str, Enum):
    POLICY = "theta"
    TRAINING = "F"
    VERIFIER = "V"
    HARNESS = "H"


class RSIMode(str, Enum):
    HARNESS = "harness"
    ADAPTER = "adapter"
    RL = "rl"


class FailureCategory(str, Enum):
    REASONING_ERROR = "reasoning_error"
    TOOL_SELECTION_ERROR = "tool_selection_error"
    VERIFICATION_FALSE_POSITIVE = "verification_false_positive"
    VERIFICATION_FALSE_NEGATIVE = "verification_false_negative"
    REWARD_HACKING_CANDIDATE = "reward_hacking_candidate"
    TIMEOUT = "timeout"
    CONTEXT_FAILURE = "context_failure"
    CURRICULUM_MISMATCH = "curriculum_mismatch"
    POLICY_VERIFIER_DISTRIBUTION_SHIFT = "policy_verifier_distribution_shift"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ExpectedEffect:
    metric: str
    direction: str
    minimum_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.direction not in {"increase", "decrease", "stable"}:
            raise ValueError(f"invalid direction: {self.direction}")


@dataclass(frozen=True)
class EvaluationPlan:
    development: bool = True
    promotion: bool = True
    sealed: bool = True
    require_independent_trusted: bool = True


@dataclass(frozen=True)
class ResourceLimits:
    wall_time_s: float = 30.0
    cpu_time_s: int = 20
    memory_mb: int = 1024
    max_output_bytes: int = 1_000_000
    network_allowed: bool = False

    def __post_init__(self) -> None:
        if self.wall_time_s <= 0 or self.cpu_time_s <= 0 or self.memory_mb <= 0:
            raise ValueError("resource limits must be positive")


@dataclass(frozen=True)
class ImprovementProposal:
    id: str
    hypothesis: str
    target_component: Component
    patch_or_config_change: Mapping[str, Any]
    expected_effects: tuple[ExpectedEffect, ...]
    risk_factors: tuple[str, ...]
    evaluation_plan: EvaluationPlan
    rollback_plan: str
    parent_champion_id: str
    deterministic_seed: int
    dependency_metadata: Mapping[str, str] = field(default_factory=dict)
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["target_component"] = self.target_component.value
        return raw


@dataclass(frozen=True)
class FailureEvidence:
    source: str
    detail: str
    value: float | str | None = None


@dataclass(frozen=True)
class FailureCluster:
    category: FailureCategory
    count: int
    severity: float
    evidence: tuple[FailureEvidence, ...]
    task_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    proposal_id: str
    parent_champion_id: str
    target_component: Component
    diff: Mapping[str, Any]
    full_state: Mapping[str, Any]
    seed: int
    resource_limits: ResourceLimits
    dependency_metadata: Mapping[str, str]


@dataclass(frozen=True)
class SplitMetrics:
    score: float
    trusted_score: float
    reward: float
    verifier_agreement: float
    failure_rate: float
    latency_p50: float
    latency_p95: float
    throughput: float
    compute_cost: float
    samples: int
    score_stderr: float = 0.0


@dataclass(frozen=True)
class EvaluationBundle:
    evolution: SplitMetrics
    development: SplitMetrics
    promotion: SplitMetrics
    sealed: SplitMetrics
    verifier_version: int
    policy_version: int
    policy_verifier_age: int
    suite_digests: Mapping[str, str]

    @property
    def apparent_gain_basis(self) -> float:
        return self.development.reward


@dataclass(frozen=True)
class RewardHackingAssessment:
    flagged: bool
    reasons: tuple[str, ...]
    apparent_gain: float
    trusted_gain: float
    generalization_gain: float
    verification_gap: float


@dataclass(frozen=True)
class VerifierLagAssessment:
    trust_level: str
    policy_improvement_rate: float
    verifier_refresh_rate: float
    verifier_disagreement_rate: float
    trusted_label_disagreement: float
    reward_ground_truth_divergence: float
    distribution_shift: float
    stale_verifier_age: int
    intervention: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PromotionDecision:
    candidate_id: str
    accepted: bool
    reasons: tuple[str, ...]
    metric_deltas: Mapping[str, float]
    safety_checks: Mapping[str, bool]


@dataclass(frozen=True)
class ChampionSnapshot:
    champion_id: str
    generation: int
    state: Mapping[str, Any]
    policy_version: int
    verifier_version: int
    parent_champion_id: str | None = None


@dataclass(frozen=True)
class GenerationRecord:
    generation: int
    champion_id: str
    candidate_id: str
    development_score: float
    promotion_score: float
    sealed_score: float
    reward: float
    trusted_score: float
    verification_gap: float
    verifier_version: int
    policy_version: int
    policy_verifier_age: int
    latency_p50: float
    latency_p95: float
    throughput: float
    failure_rate: float
    compute_cost: float
    promotion_decision: str
