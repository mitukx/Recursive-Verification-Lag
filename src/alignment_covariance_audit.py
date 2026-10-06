"""Prospective trusted-label audit for verifier/truth covariance."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.equal_kl_alignment_stress import make_task


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_verifier(p, y, y_std, u, sigma: float, rho: float):
    nuisance = math.sqrt(max(0.0, 1.0 - rho * rho))
    error = sigma * (rho * y_std + nuisance * u)
    verifier = y_std + error
    mean_v = float(p @ verifier)
    centered_y = y - float(p @ y)
    exact_cov = float(p @ (centered_y * (verifier - mean_v)))
    return verifier, exact_cov


def covariance_audit(
    p,
    y,
    verifier,
    *,
    n: int,
    delta: float,
    seed: int,
):
    """Unbiased covariance estimate with a distribution-free Hoeffding interval."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    verifier = np.asarray(verifier, float)
    if (
        p.ndim != 1
        or y.shape != p.shape
        or verifier.shape != p.shape
        or n <= 0
        or not 0 < delta < 1
        or not np.isfinite(p).all()
        or not np.isfinite(y).all()
        or not np.isfinite(verifier).all()
    ):
        raise ValueError("valid finite aligned audit inputs required")
    mean_v = float(p @ verifier)
    x_support = y * (verifier - mean_v)
    x_range = float(x_support.max() - x_support.min())
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(p), size=n, replace=True, p=p)
    observed = x_support[indices]
    estimate = float(observed.mean())
    if x_range == 0:
        radius = 0.0
    else:
        radius = x_range * math.sqrt(math.log(2.0 / delta) / (2.0 * n))
    lower, upper = estimate - radius, estimate + radius
    if upper < 0:
        decision = "block"
    elif lower > 0:
        decision = "allow"
    else:
        decision = "inconclusive"
    exact_cov = float(p @ x_support)
    true_sign = -1 if exact_cov < 0 else (1 if exact_cov > 0 else 0)
    wrong = int(
        (decision == "block" and true_sign >= 0)
        or (decision == "allow" and true_sign <= 0)
    )
    return {
        "estimate": estimate,
        "radius": radius,
        "lower": lower,
        "upper": upper,
        "decision": decision,
        "exact_covariance": exact_cov,
        "true_sign": true_sign,
        "wrong_sign_decision": wrong,
        "x_range": x_range,
    }


def run(protocol: Path, output: Path):
    lock = json.loads(protocol.read_text())
    if not lock["status"].startswith(
        "prospective geometry-aware audit experiment locked before execution"
    ):
        raise ValueError("locked prospective audit protocol required")
    support = int(lock["support_size"])
    start = int(lock["task_seeds"]["start"])
    count = int(lock["task_seeds"]["count"])
    sigma = float(lock["verifier_geometry"]["sigma"])
    arms = {
        str(k): float(v)
        for k, v in lock["verifier_geometry"]["arms"].items()
    }
    budgets = [int(x) for x in lock["trusted_label_budgets"]]
    delta = float(lock["confidence_delta"])
    if set(arms) != {"harmful", "benign"}:
        raise ValueError("exact harmful/benign arms required")

    rows = []
    for task_seed in range(start, start + count):
        p, y, y_std, u = make_task(task_seed, support)
        for arm, rho in arms.items():
            verifier, exact_cov = build_verifier(
                p, y, y_std, u, sigma=sigma, rho=rho
            )
            expected_sign = -1 if arm == "harmful" else 1
            if np.sign(exact_cov) != expected_sign:
                raise AssertionError(
                    f"geometry construction sign mismatch: {task_seed} {arm}"
                )
            for n in budgets:
                audit = covariance_audit(
                    p,
                    y,
                    verifier,
                    n=n,
                    delta=delta,
                    seed=(
                        task_seed * 1_000_003
                        + n * 101
                        + (0 if arm == "harmful" else 1)
                    ),
                )
                rows.append(
                    {
                        "task_seed": task_seed,
                        "arm": arm,
                        "rho": rho,
                        "sigma": sigma,
                        "labels": n,
                        **audit,
                    }
                )

    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "rows.csv.gz", index=False, compression="gzip")

    summaries = []
    for (labels, arm), cohort in frame.groupby(["labels", "arm"], sort=True):
        summaries.append(
            {
                "labels": int(labels),
                "arm": arm,
                "tasks": len(cohort),
                "mean_exact_covariance": float(cohort.exact_covariance.mean()),
                "block_rate": float((cohort.decision == "block").mean()),
                "allow_rate": float((cohort.decision == "allow").mean()),
                "inconclusive_rate": float(
                    (cohort.decision == "inconclusive").mean()
                ),
                "wrong_sign_decision_rate": float(
                    cohort.wrong_sign_decision.mean()
                ),
                "coverage_rate": float(
                    (
                        (cohort.lower <= cohort.exact_covariance)
                        & (cohort.exact_covariance <= cohort.upper)
                    ).mean()
                ),
                "mean_radius": float(cohort.radius.mean()),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)

    primary_budget = int(lock["primary_budget"])
    primary = summary[summary.labels == primary_budget].set_index("arm")
    harmful_block = float(primary.loc["harmful", "block_rate"])
    benign_allow = float(primary.loc["benign", "allow_rate"])
    harmful_wrong = float(
        primary.loc["harmful", "wrong_sign_decision_rate"]
    )
    benign_wrong = float(primary.loc["benign", "wrong_sign_decision_rate"])
    passed = (
        harmful_block >= 0.8
        and benign_allow >= 0.8
        and harmful_wrong <= 0.08
        and benign_wrong <= 0.08
    )

    manifest = {
        "status": lock["status"],
        "protocol_sha256": sha256(protocol),
        "code_sha256": sha256(Path(__file__)),
        "task_count": count,
        "support_size": support,
        "row_count": len(frame),
        "primary_budget": primary_budget,
        "primary_harmful_block_rate": harmful_block,
        "primary_benign_allow_rate": benign_allow,
        "primary_harmful_wrong_sign_rate": harmful_wrong,
        "primary_benign_wrong_sign_rate": benign_wrong,
        "primary_prediction_passed": passed,
        "policy_kl_used_for_decision": False,
        "new_pretrained_model_runs": 0,
        "heldout_benchmark_access": 0,
        "scope": lock["scope"],
        "output_sha256": {
            "rows.csv.gz": sha256(output / "rows.csv.gz"),
            "summary.csv": sha256(output / "summary.csv"),
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
        default=Path("configs/alignment_covariance_audit_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.output)
