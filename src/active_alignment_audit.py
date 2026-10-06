"""Active trusted-label audit for verifier/truth alignment covariance.

The estimator targets Cov_p(y, v) = E_p[y (v - E_p[v])] with y in [0, 1].
The active proposal r_i proportional to p_i |v_i - E_p[v]| minimizes the
worst-case absolute importance coefficient over proposal distributions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.alignment_covariance_audit import build_verifier
from src.equal_kl_alignment_stress import make_task


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(p, y, verifier):
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    verifier = np.asarray(verifier, float)
    if (
        p.ndim != 1
        or y.shape != p.shape
        or verifier.shape != p.shape
        or len(p) < 2
        or not np.isfinite(p).all()
        or not np.isfinite(y).all()
        or not np.isfinite(verifier).all()
        or np.any(p <= 0)
        or not np.isclose(p.sum(), 1.0, atol=1e-12, rtol=0)
        or np.any(y < 0)
        or np.any(y > 1)
    ):
        raise ValueError("positive normalized p, aligned finite verifier, and y in [0,1] required")
    return p, y, verifier


def active_proposal(p, verifier):
    """Return the minimax proposal for the covariance importance estimator."""
    p = np.asarray(p, float)
    verifier = np.asarray(verifier, float)
    if (
        p.ndim != 1
        or verifier.shape != p.shape
        or not np.isfinite(p).all()
        or not np.isfinite(verifier).all()
        or np.any(p <= 0)
        or not np.isclose(p.sum(), 1.0, atol=1e-12, rtol=0)
    ):
        raise ValueError("valid positive normalized policy and verifier required")
    centered = verifier - float(p @ verifier)
    weights = p * np.abs(centered)
    scale = float(weights.sum())
    if scale <= 1e-15:
        return {
            "proposal": p.copy(),
            "centered_verifier": centered,
            "active_scale": 0.0,
            "constant_verifier": True,
        }
    proposal = weights / scale
    return {
        "proposal": proposal,
        "centered_verifier": centered,
        "active_scale": scale,
        "constant_verifier": False,
    }


def estimator_range(p, centered_verifier, proposal):
    """Distribution-free single-sample estimator range for unknown y in [0,1]."""
    p = np.asarray(p, float)
    centered = np.asarray(centered_verifier, float)
    proposal = np.asarray(proposal, float)
    support = proposal > 0
    if not support.any():
        raise ValueError("proposal has empty support")
    if np.any((~support) & (np.abs(p * centered) > 1e-15)):
        raise ValueError("proposal omits a nonzero covariance coefficient")
    coefficient = p[support] * centered[support] / proposal[support]
    lower = min(0.0, float(coefficient.min()))
    upper = max(0.0, float(coefficient.max()))
    return {
        "lower": lower,
        "upper": upper,
        "width": upper - lower,
        "max_abs": max(abs(lower), abs(upper)),
    }


def covariance_audit(p, y, verifier, *, n: int, delta: float, seed: int, mode: str):
    """Unbiased passive or active covariance audit with a Hoeffding interval."""
    p, y, verifier = _validate(p, y, verifier)
    if n <= 0 or not 0 < delta < 1:
        raise ValueError("n>0 and delta in (0,1) required")
    if mode not in {"passive", "active"}:
        raise ValueError("mode must be passive or active")

    active = active_proposal(p, verifier)
    centered = active["centered_verifier"]
    proposal = p.copy() if mode == "passive" else active["proposal"]
    exact = float(p @ (y * centered))

    if active["constant_verifier"]:
        return {
            "mode": mode,
            "estimate": 0.0,
            "radius": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "decision": "inconclusive",
            "exact_covariance": exact,
            "wrong_sign_decision": 0,
            "range_width": 0.0,
            "max_abs_importance_coefficient": 0.0,
            "active_scale": 0.0,
        }

    bounds = estimator_range(p, centered, proposal)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(p), size=n, replace=True, p=proposal)
    observed = (p[indices] / proposal[indices]) * y[indices] * centered[indices]
    estimate = float(observed.mean())
    radius = bounds["width"] * math.sqrt(math.log(2.0 / delta) / (2.0 * n))
    lower, upper = estimate - radius, estimate + radius
    if upper < 0:
        decision = "block"
    elif lower > 0:
        decision = "allow"
    else:
        decision = "inconclusive"
    wrong = int(
        (decision == "block" and exact >= 0)
        or (decision == "allow" and exact <= 0)
    )
    return {
        "mode": mode,
        "estimate": estimate,
        "radius": radius,
        "lower": lower,
        "upper": upper,
        "decision": decision,
        "exact_covariance": exact,
        "wrong_sign_decision": wrong,
        "range_width": bounds["width"],
        "max_abs_importance_coefficient": bounds["max_abs"],
        "active_scale": active["active_scale"],
    }


def run(protocol: Path, output: Path):
    lock = json.loads(protocol.read_text())
    if not lock["status"].startswith(
        "prospective active covariance audit locked before evaluation outcomes"
    ):
        raise ValueError("locked prospective protocol required")
    support = int(lock["support_size"])
    start = int(lock["evaluation_task_seeds"]["start"])
    count = int(lock["evaluation_task_seeds"]["count"])
    sigma = float(lock["verifier_geometry"]["sigma"])
    arms = {str(k): float(v) for k, v in lock["verifier_geometry"]["arms"].items()}
    budgets = [int(x) for x in lock["trusted_label_budgets"]]
    delta = float(lock["confidence_delta"])
    primary_budget = int(lock["primary"]["budget"])
    if set(arms) != {"harmful", "benign"}:
        raise ValueError("harmful and benign arms required")
    if support < 4 or count < 1 or any(n <= 0 for n in budgets):
        raise ValueError("invalid experiment grid")

    rows = []
    for task_seed in range(start, start + count):
        p, y, y_std, u = make_task(task_seed, support)
        for arm, rho in arms.items():
            verifier, exact = build_verifier(
                p, y, y_std, u, sigma=sigma, rho=rho
            )
            for mode_index, mode in enumerate(("passive", "active")):
                for budget_index, budget in enumerate(budgets):
                    audit_seed = (
                        50_000_000
                        + task_seed * 1000
                        + mode_index * 100
                        + budget_index
                    )
                    result = covariance_audit(
                        p,
                        y,
                        verifier,
                        n=budget,
                        delta=delta,
                        seed=audit_seed,
                        mode=mode,
                    )
                    rows.append(
                        {
                            "task_seed": task_seed,
                            "arm": arm,
                            "rho": rho,
                            "budget": budget,
                            "mode": mode,
                            **result,
                            "exact_covariance_reference": exact,
                        }
                    )

    frame = pd.DataFrame(rows)
    if not np.allclose(
        frame.exact_covariance,
        frame.exact_covariance_reference,
        atol=1e-12,
        rtol=0,
    ):
        raise AssertionError("exact covariance implementations disagree")
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "rows.csv.gz", index=False, compression="gzip")

    summary = (
        frame.assign(
            decisive=(frame.decision != "inconclusive").astype(int),
            block=(frame.decision == "block").astype(int),
            allow=(frame.decision == "allow").astype(int),
        )
        .groupby(["arm", "mode", "budget"], sort=True)
        .agg(
            tasks=("task_seed", "count"),
            decisive_rate=("decisive", "mean"),
            block_rate=("block", "mean"),
            allow_rate=("allow", "mean"),
            wrong_sign_decisive_rate=("wrong_sign_decision", "mean"),
            mean_radius=("radius", "mean"),
            mean_range_width=("range_width", "mean"),
            mean_max_abs_importance_coefficient=(
                "max_abs_importance_coefficient",
                "mean",
            ),
            mean_exact_covariance=("exact_covariance", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output / "summary.csv", index=False)

    primary = summary[
        (summary.arm == "harmful") & (summary.budget == primary_budget)
    ].set_index("mode")
    active = primary.loc["active"]
    passive = primary.loc["passive"]
    conditions = {
        "active_harmful_block_rate": float(active.block_rate)
        >= float(lock["primary"]["active_block_rate_min"]),
        "passive_harmful_block_rate": float(passive.block_rate)
        <= float(lock["primary"]["passive_block_rate_max"]),
        "active_wrong_sign_rate": float(active.wrong_sign_decisive_rate)
        <= float(lock["primary"]["wrong_sign_rate_max"]),
        "active_range_strictly_smaller": float(active.mean_range_width)
        < float(passive.mean_range_width),
    }

    manifest = {
        "status": lock["status"],
        "protocol_sha256": sha256(protocol),
        "code_sha256": sha256(Path(__file__)),
        "support_size": support,
        "evaluation_task_count": count,
        "development_seed_policy": lock["development_seed_policy"],
        "primary_budget": primary_budget,
        "primary_conditions": conditions,
        "primary_prediction_passed": bool(all(conditions.values())),
        "harmful_primary_active_block_rate": float(active.block_rate),
        "harmful_primary_passive_block_rate": float(passive.block_rate),
        "harmful_primary_active_mean_range_width": float(active.mean_range_width),
        "harmful_primary_passive_mean_range_width": float(passive.mean_range_width),
        "new_pretrained_model_runs": 0,
        "heldout_benchmark_access": 0,
        "scope": lock["scope"],
        "output_sha256": {
            p.name: sha256(p)
            for p in sorted(output.iterdir())
            if p.is_file() and p.name != "manifest.json"
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/active_alignment_audit_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.output)
