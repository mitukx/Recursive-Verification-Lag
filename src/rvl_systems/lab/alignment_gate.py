from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .contracts import digest


VALID_SAMPLING_SCHEMES = {"iid_policy"}


@dataclass(frozen=True)
class AlignmentAuditObservation:
    """One trusted audit sampled from the declared policy distribution."""

    trajectory_id: str
    policy_version: int
    proxy_score: float
    trusted_reward: float

    def __post_init__(self) -> None:
        if not self.trajectory_id:
            raise ValueError("trajectory_id is required")
        if self.policy_version < 0:
            raise ValueError("policy_version must be nonnegative")
        if not math.isfinite(self.proxy_score):
            raise ValueError("proxy_score must be finite")
        if not math.isfinite(self.trusted_reward) or not 0.0 <= self.trusted_reward <= 1.0:
            raise ValueError("trusted_reward must be a finite probability")


@dataclass(frozen=True)
class AlignmentGatePolicy:
    """Distribution-free local sign gate for verifier/truth covariance.

    The confidence statement is valid only for i.i.d. samples from one policy
    version. Adaptive priority sampling is intentionally rejected rather than
    silently treated as i.i.d.
    """

    min_samples: int = 16
    delta: float = 0.05

    def __post_init__(self) -> None:
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be in (0,1)")


@dataclass(frozen=True)
class AlignmentGateDecision:
    decision: str
    policy_version: int
    samples: int
    estimate: float
    radius: float
    lower: float
    upper: float
    proxy_mean: float
    proxy_min: float
    proxy_max: float
    sampling_scheme: str
    evidence_sha256: str

    @property
    def blocked(self) -> bool:
        return self.decision == "block"

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"


def evaluate_alignment_gate(
    observations: list[AlignmentAuditObservation],
    *,
    proxy_mean: float,
    proxy_min: float,
    proxy_max: float,
    sampling_scheme: str,
    policy: AlignmentGatePolicy | None = None,
) -> AlignmentGateDecision:
    """Estimate Cov(y,v) and make a fail-closed local-direction decision.

    proxy_mean, proxy_min, and proxy_max must refer to the same target policy
    distribution that generated the observations. The gate certifies only the
    sign of the local exponential-update derivative, not a finite post-update
    reward guarantee.
    """

    policy = policy or AlignmentGatePolicy()
    if sampling_scheme not in VALID_SAMPLING_SCHEMES:
        raise ValueError(
            "alignment confidence bound requires iid_policy sampling; "
            "adaptive/priority audits need a propensity-aware estimator"
        )
    if len(observations) < policy.min_samples:
        raise ValueError("insufficient trusted alignment samples")
    if len({row.trajectory_id for row in observations}) != len(observations):
        raise ValueError("duplicate alignment audit identities")
    versions = {row.policy_version for row in observations}
    if len(versions) != 1:
        raise ValueError("alignment audits must come from one policy version")
    if not all(math.isfinite(x) for x in (proxy_mean, proxy_min, proxy_max)):
        raise ValueError("proxy support statistics must be finite")
    if proxy_min > proxy_mean or proxy_mean > proxy_max:
        raise ValueError("proxy_mean must lie inside declared proxy support")

    centered_min = proxy_min - proxy_mean
    centered_max = proxy_max - proxy_mean
    x_min = min(0.0, centered_min)
    x_max = max(0.0, centered_max)
    x_range = x_max - x_min

    values = [
        row.trusted_reward * (row.proxy_score - proxy_mean)
        for row in observations
    ]
    if any(value < x_min - 1e-12 or value > x_max + 1e-12 for value in values):
        raise ValueError("observation lies outside declared proxy support")

    estimate = sum(values) / len(values)
    radius = (
        0.0
        if x_range == 0.0
        else x_range
        * math.sqrt(math.log(2.0 / policy.delta) / (2.0 * len(values)))
    )
    lower = estimate - radius
    upper = estimate + radius
    if upper < 0.0:
        decision = "block"
    elif lower > 0.0:
        decision = "allow"
    else:
        decision = "inconclusive"

    evidence = {
        "policy": asdict(policy),
        "sampling_scheme": sampling_scheme,
        "policy_version": next(iter(versions)),
        "proxy_mean": proxy_mean,
        "proxy_min": proxy_min,
        "proxy_max": proxy_max,
        "observations": [asdict(row) for row in observations],
        "summary": {
            "estimate": estimate,
            "radius": radius,
            "lower": lower,
            "upper": upper,
            "decision": decision,
        },
    }
    return AlignmentGateDecision(
        decision=decision,
        policy_version=next(iter(versions)),
        samples=len(values),
        estimate=estimate,
        radius=radius,
        lower=lower,
        upper=upper,
        proxy_mean=proxy_mean,
        proxy_min=proxy_min,
        proxy_max=proxy_max,
        sampling_scheme=sampling_scheme,
        evidence_sha256=digest(evidence),
    )
