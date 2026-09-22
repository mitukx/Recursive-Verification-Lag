"""Observable refresh controllers for recursive verification experiments.

The controller is intentionally agnostic to the generator.  An experiment loop
feeds it observable movement diagnostics accumulated since the last trusted
verifier refresh.  This makes it usable for exponential reweighting, Best-of-N,
RL updates, or precomputed candidate banks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class MovementState:
    rounds_since_refresh: int
    cumulative_kl: float | None = None
    max_log_density_ratio: float | None = None
    restricted_geometry: float | None = None
    proxy_margin: float | None = None


class RefreshController:
    """Base interface for verifier-refresh decisions."""

    name = "base"

    def should_refresh(self, state: MovementState) -> bool:
        raise NotImplementedError


@dataclass
class FixedCadence(RefreshController):
    interval: int
    name = "fixed_cadence"

    def should_refresh(self, state: MovementState) -> bool:
        if self.interval <= 0:
            raise ValueError("interval must be positive")
        return state.rounds_since_refresh >= self.interval


@dataclass
class ScalarThreshold(RefreshController):
    metric: str
    threshold: float
    name = "scalar_threshold"

    def should_refresh(self, state: MovementState) -> bool:
        if self.threshold < 0:
            raise ValueError("threshold must be nonnegative")
        if not hasattr(state, self.metric):
            raise ValueError(f"unknown movement metric: {self.metric}")
        value = getattr(state, self.metric)
        if value is None:
            return False
        return float(value) >= self.threshold


@dataclass
class GeometryMarginController(RefreshController):
    """Refresh when verification-relevant movement exhausts a safety margin.

    This is the operational version of the paper's intended law: movement alone
    is insufficient, so movement is normalized by an observable proxy margin.
    The controller does *not* use trusted true gain at decision time.
    """

    threshold: float
    epsilon: float = 1e-8
    name = "geometry_margin"

    def should_refresh(self, state: MovementState) -> bool:
        if state.restricted_geometry is None or state.proxy_margin is None:
            return False
        margin = max(abs(float(state.proxy_margin)), self.epsilon)
        risk = float(state.restricted_geometry) / (margin * margin)
        return risk >= self.threshold


def controller_from_dict(config: Mapping[str, object]) -> RefreshController:
    kind = str(config.get("kind", ""))
    if kind == "fixed":
        return FixedCadence(interval=int(config["interval"]))
    if kind == "threshold":
        return ScalarThreshold(metric=str(config["metric"]), threshold=float(config["threshold"]))
    if kind == "geometry_margin":
        return GeometryMarginController(threshold=float(config["threshold"]))
    raise ValueError(f"unknown controller kind: {kind}")
