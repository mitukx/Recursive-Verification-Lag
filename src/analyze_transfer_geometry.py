"""Post-hoc score-geometry audit for the archived matched-KL transfer study.

This module never constructs policies from evaluation rewards. It analyzes
already-persisted frozen/refreshed decisions after verifying their archived
hashes. The purpose is to separate positive-affine score calibration, ranking
changes, and within-ranking score-spacing changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.transfer_calibration import score_family


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_score_geometry(base, frozen_scores, refreshed_scores):
    """Positive-affine-invariant score displacement on baseline support."""
    p, frozen, _ = score_family(base, frozen_scores)
    p2, refreshed, _ = score_family(base, refreshed_scores)
    if not np.allclose(p, p2, atol=0, rtol=0):
        raise AssertionError("score-family baseline changed")
    diff = refreshed - frozen
    return {
        "canonical_weighted_rms": float(np.sqrt(p @ (diff * diff))),
        "canonical_max_abs": float(np.max(np.abs(diff))),
    }


def policy_geometry(frozen, refreshed):
    """Symmetric policy displacement metrics."""
    p = np.asarray(frozen, float)
    q = np.asarray(refreshed, float)
    if (
        p.ndim != 1
        or q.shape != p.shape
        or not np.isfinite(p).all()
        or not np.isfinite(q).all()
        or np.any(p < 0)
        or np.any(q < 0)
        or not np.isclose(p.sum(), 1.0, atol=1e-12, rtol=0)
        or not np.isclose(q.sum(), 1.0, atol=1e-12, rtol=0)
    ):
        raise ValueError("finite normalized aligned policies required")
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0, p * np.log(p / m), 0.0).sum()
        kl_qm = np.where(q > 0, q * np.log(q / m), 0.0).sum()
    return {
        "policy_total_variation": float(0.5 * np.abs(p - q).sum()),
        "policy_jensen_shannon": float(0.5 * (kl_pm + kl_qm)),
    }


def _spearman(x, y):
    x = pd.Series(np.asarray(x, float)).rank(method="average")
    y = pd.Series(np.asarray(y, float)).rank(method="average")
    value = x.corr(y)
    return float(value) if np.isfinite(value) else float("nan")


def _load_archive(policies_path: Path, ranking_path: Path, bank_path: Path, source_manifest: Path):
    manifest = json.loads(source_manifest.read_text())
    expected = manifest.get("output_sha256", {})
    for path in (policies_path, ranking_path):
        if path.name not in expected or sha256(path) != expected[path.name]:
            raise ValueError(f"archived output hash mismatch: {path.name}")
    if sha256(bank_path) != manifest.get("bank_sha256"):
        raise ValueError("archived bank hash mismatch")

    decisions = json.loads(policies_path.read_text())
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("nonempty policy archive required")
    index = {}
    for row in decisions:
        key = (str(row["task_id"]), row["arm"], float(row["requested_kl"]))
        if key in index:
            raise ValueError("duplicate archived policy decision")
        index[key] = row

    rewards = {}
    for line in bank_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        candidate = row["candidate_id"]
        value = float(row["trusted_score"])
        if candidate in rewards and rewards[candidate] != value:
            raise ValueError("inconsistent candidate reward")
        rewards[candidate] = value

    ranking = pd.read_csv(ranking_path)
    exact = ranking[
        (ranking.representation == "all") & (ranking.absolute_tie_tolerance == 0.0)
    ].copy()
    if exact.task_id.astype(str).duplicated().any():
        raise ValueError("duplicate exact ranking audit row")
    reversals = dict(zip(exact.task_id.astype(str), exact.strict_reversals.astype(int)))
    return manifest, decisions, index, rewards, reversals


def analyze(
    policies_path: Path,
    ranking_path: Path,
    bank_path: Path,
    source_manifest: Path,
    output: Path,
):
    """Analyze archived matched-KL decisions without changing any policy."""
    manifest, decisions, index, rewards, reversals = _load_archive(
        policies_path, ranking_path, bank_path, source_manifest
    )
    tasks = sorted({str(row["task_id"]) for row in decisions})
    budgets = sorted({float(row["requested_kl"]) for row in decisions})
    rows = []
    public_max_tv = 0.0

    for task in tasks:
        if task not in reversals:
            raise ValueError(f"missing ranking audit for task {task}")
        for budget in budgets:
            frozen = index[(task, "all_frozen", budget)]
            refreshed = index[(task, "all_refreshed", budget)]
            if (
                frozen["candidate_ids"] != refreshed["candidate_ids"]
                or not np.allclose(frozen["baseline"], refreshed["baseline"], atol=0, rtol=0)
                or not np.isclose(frozen["target_kl"], refreshed["target_kl"], atol=3e-12, rtol=0)
                or not np.isclose(frozen["kl"], refreshed["kl"], atol=3e-12, rtol=0)
            ):
                raise ValueError("matched-KL comparison is not aligned")
            y = np.array([rewards[c] for c in frozen["candidate_ids"]], float)
            geometry = canonical_score_geometry(
                frozen["baseline"], frozen["scores"], refreshed["scores"]
            )
            policy = policy_geometry(frozen["policy"], refreshed["policy"])
            qf = np.asarray(frozen["policy"], float)
            qr = np.asarray(refreshed["policy"], float)
            rows.append(
                {
                    "task_id": task,
                    "requested_kl": budget,
                    "target_kl": float(frozen["target_kl"]),
                    "strict_reversals": int(reversals[task]),
                    **geometry,
                    **policy,
                    "trusted_reward_delta_evaluation_only": float((qr - qf) @ y),
                    "task_has_any_success_evaluation_only": int(np.any(y > 0)),
                }
            )

            pf = index[(task, "public_frozen", budget)]
            pr = index[(task, "public_refreshed", budget)]
            if pf["candidate_ids"] != pr["candidate_ids"]:
                raise ValueError("public control candidate mismatch")
            public_max_tv = max(
                public_max_tv,
                policy_geometry(pf["policy"], pr["policy"])["policy_total_variation"],
            )

    frame = pd.DataFrame(rows)
    if public_max_tv > 1e-10:
        raise AssertionError("public positive-affine control changed at matched KL")
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "task_budget.csv", index=False)

    summaries = []
    for budget, cohort in frame.groupby("requested_kl", sort=True):
        summaries.append(
            {
                "requested_kl": budget,
                "tasks": len(cohort),
                "mean_canonical_weighted_rms": cohort.canonical_weighted_rms.mean(),
                "mean_policy_total_variation": cohort.policy_total_variation.mean(),
                "mean_trusted_reward_delta_evaluation_only": cohort.trusted_reward_delta_evaluation_only.mean(),
                "tasks_with_strict_reversal": int((cohort.strict_reversals > 0).sum()),
                "tasks_with_policy_change": int((cohort.policy_total_variation > 1e-12).sum()),
                "tasks_without_reversal_but_policy_change": int(
                    ((cohort.strict_reversals == 0) & (cohort.policy_total_variation > 1e-12)).sum()
                ),
            }
        )
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)

    active = frame[frame.target_kl > 0].copy()
    correlations = pd.DataFrame(
        [
            {
                "x": "canonical_weighted_rms",
                "y": "policy_total_variation",
                "rows": len(active),
                "spearman": _spearman(active.canonical_weighted_rms, active.policy_total_variation),
            },
            {
                "x": "canonical_weighted_rms",
                "y": "abs_trusted_reward_delta_evaluation_only",
                "rows": len(active),
                "spearman": _spearman(
                    active.canonical_weighted_rms,
                    active.trusted_reward_delta_evaluation_only.abs(),
                ),
            },
            {
                "x": "strict_reversals",
                "y": "policy_total_variation",
                "rows": len(active),
                "spearman": _spearman(active.strict_reversals, active.policy_total_variation),
            },
        ]
    )
    correlations.to_csv(output / "correlations.csv", index=False)

    by_task = frame.sort_values("requested_kl").groupby("task_id", as_index=False).first()
    audit = {
        "status": "post-hoc archived development diagnostic; not confirmatory heldout evidence",
        "source_manifest_sha256": sha256(source_manifest),
        "policies_sha256": sha256(policies_path),
        "ranking_sha256": sha256(ranking_path),
        "bank_sha256": sha256(bank_path),
        "code_sha256": sha256(Path(__file__)),
        "tasks": len(tasks),
        "budgets": budgets,
        "strict_reversals_total_unique_pairs": int(by_task.strict_reversals.sum()),
        "tasks_with_strict_reversal": int((by_task.strict_reversals > 0).sum()),
        "public_control_max_policy_total_variation": public_max_tv,
        "decision_time_evaluation_task_trusted_queries": manifest.get(
            "decision_time_evaluation_task_trusted_queries"
        ),
        "new_policy_decisions": 0,
        "new_physical_candidate_executions": 0,
        "generator_parameter_updates": 0,
        "interpretation": (
            "canonical score geometry is positive-affine invariant; correlations are descriptive "
            "on the inspected development archive and are not a safety certificate"
        ),
    }
    (output / "manifest.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policies",
        type=Path,
        default=Path("results/transfer_calibration_v1/policies.json"),
    )
    parser.add_argument(
        "--ranking",
        type=Path,
        default=Path("results/transfer_calibration_v1/ranking_changes.csv"),
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=Path("data/mbppplus_qwen15b_development_v2_scored.jsonl"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("results/transfer_calibration_v1/manifest.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.policies, args.ranking, args.bank, args.source_manifest, args.output)
