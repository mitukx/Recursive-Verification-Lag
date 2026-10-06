"""Finite-KL path identities for exponential verifier optimization."""
from __future__ import annotations

import math
import numpy as np


def _validate(p, y, v):
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    v = np.asarray(v, float)
    if (
        p.ndim != 1
        or y.shape != p.shape
        or v.shape != p.shape
        or len(p) < 2
        or not np.isfinite(p).all()
        or not np.isfinite(y).all()
        or not np.isfinite(v).all()
        or np.any(p <= 0)
        or not np.isclose(p.sum(), 1.0, atol=1e-12, rtol=0)
    ):
        raise ValueError("finite aligned vectors and positive normalized baseline required")
    return p, y, v


def exponential_tilt(p, v, beta):
    p = np.asarray(p, float)
    v = np.asarray(v, float)
    beta = float(beta)
    if beta < 0 or not np.isfinite(beta):
        raise ValueError("finite beta>=0 required")
    logits = np.log(p) + beta * v
    logits -= logits.max()
    q = np.exp(logits)
    q /= q.sum()
    return q


def covariance(q, a, b):
    q = np.asarray(q, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(q @ (a * b) - (q @ a) * (q @ b))


def direct_quantities(p, y, v, beta):
    p, y, v = _validate(p, y, v)
    q = exponential_tilt(p, v, beta)
    true_progress = float(q @ y - p @ y)
    proxy_progress = float(q @ v - p @ v)
    kl = float(q @ (np.log(q) - np.log(p)))
    return {
        "policy": q,
        "true_progress": true_progress,
        "proxy_progress": proxy_progress,
        "kl": max(0.0, kl),
    }


def path_integrals(p, y, v, beta, *, quadrature_order=96):
    """Gauss-Legendre evaluation of the exact exponential-family identities."""
    p, y, v = _validate(p, y, v)
    beta = float(beta)
    if beta < 0 or not np.isfinite(beta):
        raise ValueError("finite beta>=0 required")
    if quadrature_order < 8:
        raise ValueError("quadrature_order must be >=8")
    if beta == 0:
        return {
            "true_progress_integral": 0.0,
            "proxy_progress_integral": 0.0,
            "kl_integral": 0.0,
            "minimum_alignment_covariance": covariance(p, y, v),
            "maximum_alignment_covariance": covariance(p, y, v),
        }

    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    times = 0.5 * beta * (nodes + 1.0)
    scaled_weights = 0.5 * beta * weights
    covariances = []
    variances = []
    for t in times:
        q = exponential_tilt(p, v, float(t))
        covariances.append(covariance(q, y, v))
        variances.append(covariance(q, v, v))
    covariances = np.asarray(covariances)
    variances = np.asarray(variances)
    return {
        "true_progress_integral": float(scaled_weights @ covariances),
        "proxy_progress_integral": float(scaled_weights @ variances),
        "kl_integral": float(scaled_weights @ (times * variances)),
        "minimum_alignment_covariance": float(covariances.min()),
        "maximum_alignment_covariance": float(covariances.max()),
    }


def verify_path_identities(p, y, v, beta, *, atol=1e-10):
    direct = direct_quantities(p, y, v, beta)
    integral = path_integrals(p, y, v, beta)
    errors = {
        "true_progress_error": abs(
            direct["true_progress"] - integral["true_progress_integral"]
        ),
        "proxy_progress_error": abs(
            direct["proxy_progress"] - integral["proxy_progress_integral"]
        ),
        "kl_error": abs(direct["kl"] - integral["kl_integral"]),
    }
    return {
        **direct,
        **integral,
        **errors,
        "passed": all(value <= atol for value in errors.values()),
    }


def error_alignment_decomposition(q, y, verifier):
    """Cov_q(y,v)=Var_q(y)+Cov_q(y,e), where e=v-y."""
    q = np.asarray(q, float)
    y = np.asarray(y, float)
    verifier = np.asarray(verifier, float)
    if q.shape != y.shape or verifier.shape != y.shape:
        raise ValueError("aligned arrays required")
    error = verifier - y
    signal = covariance(q, y, y)
    error_alignment = covariance(q, y, error)
    total = covariance(q, y, verifier)
    return {
        "signal_variance": signal,
        "error_alignment_covariance": error_alignment,
        "total_true_progress_derivative": total,
        "decomposition_error": abs(total - (signal + error_alignment)),
    }
