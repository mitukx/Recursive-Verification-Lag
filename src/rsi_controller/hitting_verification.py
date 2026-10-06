"""Reusable fixed-probe verification for bounded linear verifier-error classes.

Let v be a proxy verifier, y trusted truth, and e = v-y. Assume

    e = Phi beta + r,     ||r||_infinity <= rho.

Choose a fixed probe set H whose rows span the row space of Phi. Trusted labels
on H reveal e_H. Define

    A = Phi Phi_H^+
    e_hat = A e_H.

For any future policy contrast delta, the trusted-gain estimate is

    g_hat = delta^T (v - e_hat).

Because e_H = Phi_H beta + r_H and A Phi_H beta = Phi beta,

    |g_true - g_hat|
      <= rho ( ||delta||_1 + ||delta^T A||_1 ).

Thus one fixed probe set can be reused for arbitrarily many future policy
contrasts as long as the declared bounded error class remains valid.

This is an elementary finite-dimensional analogue inspired by OpenAI Math
family #116. It is not a consequence of that noncommutative-formula theorem.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_matrix(value) -> np.ndarray:
    x = np.asarray(value, dtype=float)
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError("finite nonempty matrix required")
    return x


def select_hitting_rows(
    features,
    ids: list[str] | tuple[str, ...],
    *,
    tolerance: float = 1e-10,
) -> tuple[int, ...]:
    """Deterministically choose a well-conditioned row-space hitting set.

    Selection is label-blind. At each step, among rows that increase rank,
    choose the row maximizing the smallest singular value of the selected
    matrix; lexical ID breaks numerical ties.
    """
    x = _as_matrix(features)
    if len(ids) != len(x) or len(set(ids)) != len(ids):
        raise ValueError("unique row IDs must align with feature rows")
    if tolerance <= 0:
        raise ValueError("positive tolerance required")
    target_rank = int(np.linalg.matrix_rank(x, tol=tolerance))
    if target_rank == 0:
        raise ValueError("zero-rank feature class has no informative probes")

    selected: list[int] = []
    rank = 0
    remaining = set(range(len(x)))
    while rank < target_rank:
        candidates = []
        for i in remaining:
            trial = x[selected + [i]]
            trial_rank = int(np.linalg.matrix_rank(trial, tol=tolerance))
            if trial_rank <= rank:
                continue
            singular = np.linalg.svd(trial, compute_uv=False)
            positive = singular[singular > tolerance]
            score = float(positive.min()) if len(positive) else 0.0
            candidates.append((-score, str(ids[i]), i))
        if not candidates:
            raise ArithmeticError("failed to construct full-rank hitting rows")
        _, _, chosen = min(candidates)
        selected.append(chosen)
        remaining.remove(chosen)
        rank += 1

    probes = tuple(selected)
    if int(np.linalg.matrix_rank(x[list(probes)], tol=tolerance)) != target_rank:
        raise AssertionError("probe rows do not span feature row space")
    return probes


@dataclass(frozen=True)
class HittingGainInterval:
    estimate: float
    radius: float
    lower: float
    upper: float
    proxy_gain: float
    probe_count: int
    feature_rank: int
    contrast_l1: float
    transport_l1: float

    @property
    def decision(self) -> str:
        if self.lower >= 0.0:
            return "allow"
        if self.upper < 0.0:
            return "block"
        return "refresh"


class LinearErrorHittingVerifier:
    """Fixed trusted probes reusable across future policy contrasts."""

    def __init__(
        self,
        features,
        proxy_scores,
        probe_indices: tuple[int, ...] | list[int],
        *,
        tolerance: float = 1e-10,
    ):
        self.features = _as_matrix(features)
        self.proxy = np.asarray(proxy_scores, dtype=float)
        if self.proxy.shape != (len(self.features),) or not np.isfinite(self.proxy).all():
            raise ValueError("proxy scores must align with feature rows")
        probes = tuple(int(i) for i in probe_indices)
        if not probes or len(set(probes)) != len(probes):
            raise ValueError("nonempty unique probes required")
        if min(probes) < 0 or max(probes) >= len(self.features):
            raise ValueError("probe index outside feature bank")
        rank = int(np.linalg.matrix_rank(self.features, tol=tolerance))
        probe_rank = int(np.linalg.matrix_rank(self.features[list(probes)], tol=tolerance))
        if probe_rank != rank:
            raise ValueError("probe rows do not span declared feature class")
        self.probes = probes
        self.feature_rank = rank
        self._pinv = np.linalg.pinv(self.features[list(probes)], rcond=tolerance)
        self.transport = self.features @ self._pinv
        # A Phi_H = Phi on the row-space class.
        residual = self.transport @ self.features[list(probes)] - self.features
        if np.max(np.abs(residual)) > 1e-7 * (1.0 + np.max(np.abs(self.features))):
            raise ArithmeticError("probe transport does not reconstruct feature class")

    def probe_errors(self, trusted_probe_scores) -> np.ndarray:
        y = np.asarray(trusted_probe_scores, dtype=float)
        if y.shape != (len(self.probes),) or not np.isfinite(y).all():
            raise ValueError("trusted probe scores must align with probes")
        if np.any((y < 0.0) | (y > 1.0)):
            raise ValueError("trusted scores outside [0,1]")
        return self.proxy[list(self.probes)] - y

    def reconstructed_error(self, trusted_probe_scores) -> np.ndarray:
        return self.transport @ self.probe_errors(trusted_probe_scores)

    def interval(
        self,
        delta,
        trusted_probe_scores,
        *,
        residual_radius: float,
    ) -> HittingGainInterval:
        d = np.asarray(delta, dtype=float)
        if d.shape != self.proxy.shape or not np.isfinite(d).all():
            raise ValueError("finite policy contrast must align with bank")
        if not np.isfinite(residual_radius) or residual_radius < 0:
            raise ValueError("nonnegative residual radius required")
        e_hat = self.reconstructed_error(trusted_probe_scores)
        proxy_gain = float(d @ self.proxy)
        estimate = float(d @ (self.proxy - e_hat))
        contrast_l1 = float(np.abs(d).sum())
        transported = d @ self.transport
        transport_l1 = float(np.abs(transported).sum())
        radius = float(residual_radius * (contrast_l1 + transport_l1))
        return HittingGainInterval(
            estimate=estimate,
            radius=radius,
            lower=estimate - radius,
            upper=estimate + radius,
            proxy_gain=proxy_gain,
            probe_count=len(self.probes),
            feature_rank=self.feature_rank,
            contrast_l1=contrast_l1,
            transport_l1=transport_l1,
        )


def evaluator_residual_diagnostic(features, proxy_scores, trusted_scores) -> dict[str, float]:
    """Offline misspecification diagnostic; never an online controller input."""
    x = _as_matrix(features)
    v = np.asarray(proxy_scores, dtype=float)
    y = np.asarray(trusted_scores, dtype=float)
    if v.shape != (len(x),) or y.shape != (len(x),):
        raise ValueError("scores must align with feature matrix")
    e = v - y
    beta, *_ = np.linalg.lstsq(x, e, rcond=None)
    residual = e - x @ beta
    return {
        "lstsq_residual_inf": float(np.max(np.abs(residual))),
        "lstsq_residual_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "error_rmse": float(np.sqrt(np.mean(e ** 2))),
    }
