"""Prospective equal-KL verifier-error alignment stress test.

The protocol is locked in configs/equal_kl_alignment_stress_v1.json. This is a
controlled finite-support mechanism experiment: no pretrained model, learned
verifier, heldout benchmark, or deployment claim is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weighted_inner(p, a, b):
    return float(np.asarray(p, float) @ (np.asarray(a, float) * np.asarray(b, float)))


def make_task(seed: int, support_size: int):
    """Deterministic baseline, binary reward and p-orthogonal nuisance direction."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.full(support_size, 2.0))
    while True:
        y = rng.binomial(1, 0.5, size=support_size).astype(float)
        if y.min() < y.max():
            break
    mean = float(p @ y)
    centered = y - mean
    sd = math.sqrt(float(p @ (centered * centered)))
    y_std = centered / sd

    for _ in range(100):
        u = rng.normal(size=support_size)
        u = u - float(p @ u)
        u = u - weighted_inner(p, u, y_std) * y_std
        norm = math.sqrt(weighted_inner(p, u, u))
        if norm > 1e-12:
            u = u / norm
            break
    else:
        raise ArithmeticError("failed to construct orthogonal nuisance direction")

    if abs(float(p @ u)) > 1e-10 or abs(weighted_inner(p, u, y_std)) > 1e-10:
        raise AssertionError("orthogonal construction failed")
    if abs(weighted_inner(p, y_std, y_std) - 1.0) > 1e-10:
        raise AssertionError("reward standardization failed")
    return p, y, y_std, u


def matched_kl_tilt(p, score, target):
    """Return exponential tilt at exact KL, or an explicit unattainable record."""
    p = np.asarray(p, float)
    s = np.asarray(score, float)
    if (
        p.ndim != 1
        or s.shape != p.shape
        or not np.isfinite(p).all()
        or not np.isfinite(s).all()
        or np.any(p <= 0)
        or not np.isclose(p.sum(), 1.0, atol=1e-12, rtol=0)
        or not np.isfinite(target)
        or target <= 0
    ):
        raise ValueError("finite positive-support baseline, score and positive KL required")

    span = float(s.max() - s.min())
    if span == 0.0:
        return {"attainable": False, "limiting_kl": 0.0}

    maximum = s.max()
    top = np.isclose(s, maximum, atol=1e-14, rtol=0)
    limiting_kl = float(-math.log(float(p[top].sum())))
    if target >= limiting_kl:
        return {"attainable": False, "limiting_kl": limiting_kl}

    logp = np.log(p)

    def tilt(beta):
        logits = logp + beta * s
        logits -= logits.max()
        q = np.exp(logits)
        q /= q.sum()
        kl = float(q @ (np.log(q) - logp))
        return q, max(0.0, kl)

    lo, hi = 0.0, 1.0
    for _ in range(200):
        if tilt(hi)[1] >= target:
            break
        hi *= 2.0
    else:
        raise ArithmeticError("failed to bracket KL target")

    for _ in range(120):
        beta = 0.5 * (lo + hi)
        q, kl = tilt(beta)
        if abs(kl - target) <= 1e-12:
            break
        if kl < target:
            lo = beta
        else:
            hi = beta
    else:
        raise ArithmeticError("KL inversion failed")

    return {
        "attainable": True,
        "limiting_kl": limiting_kl,
        "beta": beta,
        "kl": kl,
        "policy": q,
    }


def run(protocol: Path, output: Path):
    lock = json.loads(protocol.read_text())
    if lock["status"].split(";")[0].strip() != (
        "prospective controlled-mechanism experiment locked before outcome execution"
    ):
        raise ValueError("prospective protocol status required")
    support_size = int(lock["support_size"])
    start = int(lock["task_seeds"]["start"])
    count = int(lock["task_seeds"]["count"])
    sigmas = [float(x) for x in lock["error_scales"]]
    rhos = [float(x) for x in lock["alignment_rhos"]]
    budgets = [float(x) for x in lock["matched_kl_budgets"]]
    if support_size < 4 or count < 1 or any(s <= 0 for s in sigmas):
        raise ValueError("invalid experiment grid")
    if any(abs(r) > 1 for r in rhos) or any(k <= 0 for k in budgets):
        raise ValueError("invalid alignment or KL grid")

    rows = []
    for seed in range(start, start + count):
        p, y, y_std, u = make_task(seed, support_size)
        base_reward = float(p @ y)
        for sigma in sigmas:
            for rho in rhos:
                nuisance = math.sqrt(max(0.0, 1.0 - rho * rho))
                error = sigma * (rho * y_std + nuisance * u)
                score = y_std + error
                score_mean = float(p @ score)
                cov_y_score = float(p @ ((y - base_reward) * (score - score_mean)))
                local_margin = 1.0 + sigma * rho
                for budget in budgets:
                    result = matched_kl_tilt(p, score, budget)
                    record = {
                        "task_seed": seed,
                        "sigma": sigma,
                        "rho": rho,
                        "requested_kl": budget,
                        "local_margin": local_margin,
                        "cov_y_score": cov_y_score,
                        "attainable": int(result["attainable"]),
                        "limiting_kl": result["limiting_kl"],
                    }
                    if not result["attainable"]:
                        record.update(
                            beta=np.nan,
                            actual_kl=np.nan,
                            proxy_progress=np.nan,
                            true_progress=np.nan,
                            false_progress=np.nan,
                        )
                    else:
                        q = result["policy"]
                        proxy = float(q @ score - score_mean)
                        true = float(q @ y - base_reward)
                        record.update(
                            beta=result["beta"],
                            actual_kl=result["kl"],
                            proxy_progress=proxy,
                            true_progress=true,
                            false_progress=int(proxy > 1e-12 and true < -1e-12),
                        )
                    rows.append(record)

    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "rows.csv.gz", index=False, compression="gzip")

    available = frame[frame.attainable == 1].copy()
    if (available.proxy_progress < -1e-10).any():
        raise AssertionError("exponential optimizer produced negative proxy progress")
    if not np.allclose(
        available.actual_kl, available.requested_kl, atol=2e-11, rtol=0
    ):
        raise AssertionError("matched-KL invariant failed")

    cells = (
        available.groupby(["requested_kl", "sigma", "rho"], sort=True)
        .agg(
            tasks=("task_seed", "count"),
            mean_true_progress=("true_progress", "mean"),
            mean_proxy_progress=("proxy_progress", "mean"),
            false_progress_rate=("false_progress", "mean"),
            mean_actual_kl=("actual_kl", "mean"),
        )
        .reset_index()
    )
    cells["predicted_sign"] = np.sign(1.0 + cells.sigma * cells.rho)
    cells["observed_mean_sign"] = np.sign(cells.mean_true_progress)
    cells["phase_agreement"] = (
        (cells.predicted_sign == cells.observed_mean_sign)
        | (cells.predicted_sign == 0)
    ).astype(int)
    cells.to_csv(output / "cells.csv", index=False)

    primary_budget = float(lock["primary_budget"])
    primary = cells[
        (cells.requested_kl == primary_budget) & (cells.predicted_sign != 0)
    ].copy()
    phase_accuracy = float(primary.phase_agreement.mean())
    proxy_nonnegative = bool((primary.mean_proxy_progress >= -1e-12).all())
    passed = phase_accuracy >= 0.9 and proxy_nonnegative

    budget_summary = []
    for budget, cohort in cells[cells.predicted_sign != 0].groupby(
        "requested_kl", sort=True
    ):
        harmful = cohort[cohort.predicted_sign < 0]
        benign = cohort[cohort.predicted_sign > 0]
        budget_summary.append(
            {
                "requested_kl": budget,
                "phase_accuracy": float(cohort.phase_agreement.mean()),
                "harmful_cell_false_progress_rate": float(harmful.false_progress_rate.mean()),
                "benign_cell_false_progress_rate": float(benign.false_progress_rate.mean()),
                "harmful_cell_mean_true_progress": float(harmful.mean_true_progress.mean()),
                "benign_cell_mean_true_progress": float(benign.mean_true_progress.mean()),
                "minimum_cell_mean_proxy_progress": float(cohort.mean_proxy_progress.min()),
                "cells": len(cohort),
            }
        )
    pd.DataFrame(budget_summary).to_csv(output / "summary.csv", index=False)

    manifest = {
        "status": lock["status"],
        "protocol_sha256": sha256(protocol),
        "code_sha256": sha256(Path(__file__)),
        "support_size": support_size,
        "task_count": count,
        "total_grid_rows": len(frame),
        "attainable_rows": len(available),
        "unattainable_rows": int((frame.attainable == 0).sum()),
        "primary_budget": primary_budget,
        "primary_phase_accuracy": phase_accuracy,
        "primary_proxy_nonnegative": proxy_nonnegative,
        "primary_prediction_passed": passed,
        "new_pretrained_model_runs": 0,
        "learned_verifier_updates": 0,
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
        default=Path("configs/equal_kl_alignment_stress_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.output)
