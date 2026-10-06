"""Prospective propensity-weighted verifier/truth alignment audit study."""
from __future__ import annotations

import argparse
import gzip
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


def allocate_propensities(
    p,
    verifier,
    *,
    expected_budget: float,
    method: str,
    minimum_propensity: float,
):
    """Allocate Bernoulli inclusion probabilities without trusted labels."""
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
        raise ValueError("finite aligned positive-support inputs required")
    n = len(p)
    budget = float(expected_budget)
    floor = float(minimum_propensity)
    if not 0 < floor <= 1:
        raise ValueError("minimum_propensity must be in (0,1]")
    if budget < n * floor - 1e-12 or budget > n + 1e-12:
        raise ValueError("expected budget incompatible with propensity bounds")

    if method == "uniform":
        result = np.full(n, budget / n, dtype=float)
    elif method == "proxy_leverage":
        mean_v = float(p @ verifier)
        leverage = p * np.abs(verifier - mean_v)
        if np.all(leverage <= 1e-18):
            result = np.full(n, budget / n, dtype=float)
        elif budget >= n - 1e-12:
            result = np.ones(n, dtype=float)
        else:
            base = np.power(leverage, 2.0 / 3.0)
            lo, hi = 0.0, 1.0
            def total(scale):
                return float(np.clip(scale * base, floor, 1.0).sum())
            while total(hi) < budget:
                hi *= 2.0
                if hi > 1e18:
                    raise ArithmeticError("failed to bracket propensity scale")
            for _ in range(120):
                mid = 0.5 * (lo + hi)
                if total(mid) < budget:
                    lo = mid
                else:
                    hi = mid
            result = np.clip(0.5 * (lo + hi) * base, floor, 1.0)
            # Tiny floating mismatch from clipping/bisection is harmless, but
            # keep the declared expected budget numerically auditable.
            residual = budget - float(result.sum())
            if abs(residual) > 1e-9:
                free = np.where((result > floor + 1e-12) & (result < 1 - 1e-12))[0]
                if len(free):
                    result[free[0]] += residual
    else:
        raise ValueError("unknown propensity method")

    if (
        np.any(result < floor - 1e-10)
        or np.any(result > 1 + 1e-10)
        or not np.isclose(result.sum(), budget, atol=2e-8, rtol=0)
    ):
        raise AssertionError("propensity allocation constraints failed")
    return result


def propensity_covariance_audit(
    p,
    y,
    verifier,
    propensities,
    *,
    delta: float,
    seed: int,
):
    """Horvitz-Thompson covariance estimate with conditional Hoeffding radius."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    verifier = np.asarray(verifier, float)
    pi = np.asarray(propensities, float)
    if (
        y.shape != p.shape
        or verifier.shape != p.shape
        or pi.shape != p.shape
        or np.any(pi <= 0)
        or np.any(pi > 1)
        or not 0 < delta < 1
    ):
        raise ValueError("valid aligned propensity-audit inputs required")

    mean_v = float(p @ verifier)
    centered_proxy = verifier - mean_v
    x = y * centered_proxy
    exact = float(p @ x)
    rng = np.random.default_rng(seed)
    included = rng.random(len(p)) < pi
    estimate = float(np.sum(included * p * x / pi))
    leverage_bound = p * np.abs(centered_proxy) / pi
    radius = math.sqrt(
        0.5 * math.log(2.0 / delta) * float(leverage_bound @ leverage_bound)
    )
    lower, upper = estimate - radius, estimate + radius
    if upper < 0:
        decision = "block"
    elif lower > 0:
        decision = "allow"
    else:
        decision = "inconclusive"
    sign = -1 if exact < 0 else (1 if exact > 0 else 0)
    wrong = int(
        (decision == "block" and sign >= 0)
        or (decision == "allow" and sign <= 0)
    )
    return {
        "estimate": estimate,
        "radius": radius,
        "lower": lower,
        "upper": upper,
        "decision": decision,
        "exact_covariance": exact,
        "true_sign": sign,
        "wrong_sign_decision": wrong,
        "realized_labels": int(included.sum()),
        "selected_indices": np.flatnonzero(included).astype(int).tolist(),
    }


def _deterministic_gzip_text(path: Path, text: str) -> None:
    path.write_bytes(gzip.compress(text.encode("utf-8"), compresslevel=9, mtime=0))


def run(protocol: Path, output: Path):
    lock = json.loads(protocol.read_text())
    if not lock["status"].startswith(
        "prospective post-v1 efficiency follow-up locked before execution"
    ):
        raise ValueError("locked propensity-audit protocol required")

    support = int(lock["support_size"])
    start = int(lock["task_seeds"]["start"])
    count = int(lock["task_seeds"]["count"])
    sigma = float(lock["verifier_geometry"]["sigma"])
    arms = {k: float(v) for k, v in lock["verifier_geometry"]["arms"].items()}
    methods = list(lock["selection"]["methods"])
    floor = float(lock["selection"]["minimum_propensity"])
    budgets = [float(x) for x in lock["expected_label_budgets"]]
    delta = float(lock["confidence_delta"])
    primary_budget = float(lock["primary_budget"])

    rows = []
    records = []
    for task_seed in range(start, start + count):
        p, y, y_std, u = make_task(task_seed, support)
        for arm, rho in arms.items():
            verifier, exact_cov = build_verifier(
                p, y, y_std, u, sigma=sigma, rho=rho
            )
            expected_sign = -1 if arm == "harmful" else 1
            if np.sign(exact_cov) != expected_sign:
                raise AssertionError("declared geometry arm has wrong exact sign")
            for budget in budgets:
                for method_index, method in enumerate(methods):
                    pi = allocate_propensities(
                        p,
                        verifier,
                        expected_budget=budget,
                        method=method,
                        minimum_propensity=floor,
                    )
                    audit = propensity_covariance_audit(
                        p,
                        y,
                        verifier,
                        pi,
                        delta=delta,
                        seed=(
                            task_seed * 10_000_019
                            + int(round(budget * 1000)) * 101
                            + method_index * 1009
                            + (0 if arm == "harmful" else 1)
                        ),
                    )
                    rows.append(
                        {
                            "task_seed": task_seed,
                            "arm": arm,
                            "rho": rho,
                            "sigma": sigma,
                            "expected_labels": budget,
                            "method": method,
                            "realized_labels": audit["realized_labels"],
                            "estimate": audit["estimate"],
                            "radius": audit["radius"],
                            "lower": audit["lower"],
                            "upper": audit["upper"],
                            "decision": audit["decision"],
                            "exact_covariance": audit["exact_covariance"],
                            "wrong_sign_decision": audit["wrong_sign_decision"],
                        }
                    )
                    records.append(
                        {
                            "task_seed": task_seed,
                            "arm": arm,
                            "expected_labels": budget,
                            "method": method,
                            "propensities": pi.tolist(),
                            "selected_indices": audit["selected_indices"],
                        }
                    )

    output.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame(rows)
    csv_text = frame.to_csv(index=False)
    _deterministic_gzip_text(output / "rows.csv.gz", csv_text)
    _deterministic_gzip_text(
        output / "selection_records.jsonl.gz",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
    )

    summary_rows = []
    for (budget, method, arm), cohort in frame.groupby(
        ["expected_labels", "method", "arm"], sort=True
    ):
        summary_rows.append(
            {
                "expected_labels": budget,
                "method": method,
                "arm": arm,
                "tasks": len(cohort),
                "mean_realized_labels": float(cohort.realized_labels.mean()),
                "mean_radius": float(cohort.radius.mean()),
                "block_rate": float((cohort.decision == "block").mean()),
                "allow_rate": float((cohort.decision == "allow").mean()),
                "inconclusive_rate": float(
                    (cohort.decision == "inconclusive").mean()
                ),
                "decisive_rate": float(
                    (cohort.decision != "inconclusive").mean()
                ),
                "wrong_sign_decision_rate": float(
                    cohort.wrong_sign_decision.mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "summary.csv", index=False)

    primary = summary[summary.expected_labels == primary_budget].copy()
    pivot_radius = primary.pivot(index="arm", columns="method", values="mean_radius")
    pivot_decisive = primary.pivot(
        index="arm", columns="method", values="decisive_rate"
    )
    radius_better = bool(
        (
            pivot_radius["proxy_leverage"]
            < pivot_radius["uniform"]
        ).all()
    )
    pooled = (
        primary.groupby("method", sort=True)
        .apply(lambda x: np.average(x.decisive_rate, weights=x.tasks))
        .to_dict()
    )
    decisive_not_worse = bool(
        pooled["proxy_leverage"] >= pooled["uniform"] - 1e-12
    )
    wrong_ok = bool((primary.wrong_sign_decision_rate <= 0.08).all())
    passed = radius_better and decisive_not_worse and wrong_ok

    manifest = {
        "status": lock["status"],
        "protocol_sha256": sha256(protocol),
        "code_sha256": sha256(Path(__file__)),
        "task_count": count,
        "support_size": support,
        "row_count": len(frame),
        "primary_budget": primary_budget,
        "primary_radius_better_both_arms": radius_better,
        "primary_pooled_decisive_uniform": float(pooled["uniform"]),
        "primary_pooled_decisive_proxy_leverage": float(
            pooled["proxy_leverage"]
        ),
        "primary_decisive_not_worse": decisive_not_worse,
        "primary_wrong_sign_rates_ok": wrong_ok,
        "primary_prediction_passed": passed,
        "trusted_reward_used_for_propensity_allocation": False,
        "new_pretrained_model_runs": 0,
        "heldout_benchmark_access": 0,
        "scope": lock["scope"],
        "output_sha256": {
            name: sha256(output / name)
            for name in (
                "rows.csv.gz",
                "selection_records.jsonl.gz",
                "summary.csv",
            )
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
        default=Path("configs/propensity_alignment_audit_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.output)
