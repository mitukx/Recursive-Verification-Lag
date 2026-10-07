"""Build a hiring-facing post-training capability scorecard from raw evidence.

This script does not run training. It combines independently retained real-model
artifacts from the oracle RLVR capability track and the learned reward-model
fresh/stale/shuffled track. Missing or underpowered evidence remains explicit.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


ARMS = ("oracle", "fresh", "stale", "shuffled")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def summarize_rlvr(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "missing"}
    payload = _load(path)
    metrics = payload.get("metrics", payload)
    before = metrics.get("before_accuracy")
    after = metrics.get("after_accuracy")
    delta = metrics.get("accuracy_delta")
    if not all(_finite(x) for x in (before, after, delta)):
        raise ValueError("RLVR benchmark lacks finite before/after/delta metrics")
    return {
        "status": "available",
        "before_accuracy": float(before),
        "after_accuracy": float(after),
        "accuracy_delta": float(delta),
        "eval_examples": int(metrics.get("eval_examples", 0)),
        "steps": int(metrics.get("steps", 0)),
        "model": payload.get("model") or payload.get("config", {}).get("model"),
    }


def summarize_learned_verifier(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "missing"}
    summary = _load(path)
    seeds = list(summary.get("seed_results", []))
    eligible = [row for row in seeds if row.get("eligible_primary_seed")]
    if not seeds:
        raise ValueError("learned-verifier summary has no seed_results")
    arm_values: dict[str, list[float]] = {arm: [] for arm in ARMS}
    geometry: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for row in eligible:
        shifts = row.get("mean_preference_shift", {})
        geo = row.get("pre_update_geometry", {})
        for arm in ARMS:
            if _finite(shifts.get(arm)):
                arm_values[arm].append(float(shifts[arm]))
            if _finite(geo.get(arm, {}).get("cov_y_v")):
                geometry[arm].append(float(geo[arm]["cov_y_v"]))
    mean_shift = {
        arm: (fmean(values) if values else None) for arm, values in arm_values.items()
    }
    mean_cov = {
        arm: (fmean(values) if values else None) for arm, values in geometry.items()
    }
    comparisons = {}
    if mean_shift["fresh"] is not None and mean_shift["stale"] is not None:
        comparisons["fresh_minus_stale_trusted_shift"] = (
            mean_shift["fresh"] - mean_shift["stale"]
        )
    if mean_shift["fresh"] is not None and mean_shift["shuffled"] is not None:
        comparisons["fresh_minus_shuffled_trusted_shift"] = (
            mean_shift["fresh"] - mean_shift["shuffled"]
        )
    if mean_shift["oracle"] is not None and mean_shift["fresh"] is not None:
        comparisons["oracle_minus_fresh_trusted_shift"] = (
            mean_shift["oracle"] - mean_shift["fresh"]
        )
    return {
        "status": "available" if eligible else "underpowered",
        "declared_seeds": len(seeds),
        "eligible_seeds": len(eligible),
        "required_eligible_seeds": int(summary.get("required_eligible_seeds", 0)),
        "primary_evidence_sufficient": bool(summary.get("primary_evidence_sufficient", False)),
        "primary_direction_passed": summary.get("primary_direction_passed"),
        "mean_geometry_spearman": summary.get("mean_geometry_spearman"),
        "mean_effect_k3_spearman": summary.get("mean_effect_k3_spearman"),
        "mean_trusted_preference_shift": mean_shift,
        "mean_preupdate_cov_y_v": mean_cov,
        "comparisons": comparisons,
    }


def build_scorecard(
    *,
    rlvr_benchmark: Path | None = None,
    learned_verifier_summary: Path | None = None,
) -> dict[str, Any]:
    rlvr = summarize_rlvr(rlvr_benchmark)
    learned = summarize_learned_verifier(learned_verifier_summary)

    claims = {
        "real_model_capability_gain_observed": (
            rlvr.get("status") == "available"
            and float(rlvr["accuracy_delta"]) > 0.0
        ),
        "learned_verifier_evidence_sufficient": (
            learned.get("status") == "available"
            and bool(learned.get("primary_evidence_sufficient"))
        ),
        "fresh_beats_stale_on_trusted_shift": None,
        "fresh_beats_shuffled_on_trusted_shift": None,
    }
    comparisons = learned.get("comparisons", {})
    if "fresh_minus_stale_trusted_shift" in comparisons:
        claims["fresh_beats_stale_on_trusted_shift"] = (
            comparisons["fresh_minus_stale_trusted_shift"] > 0.0
        )
    if "fresh_minus_shuffled_trusted_shift" in comparisons:
        claims["fresh_beats_shuffled_on_trusted_shift"] = (
            comparisons["fresh_minus_shuffled_trusted_shift"] > 0.0
        )

    return {
        "schema_version": 1,
        "oracle_rlvr": rlvr,
        "learned_reward_model": learned,
        "claims": claims,
        "claim_boundary": (
            "A positive scorecard field is descriptive evidence from retained real-model "
            "runs, not proof of frontier-scale post-training performance. Missing, failed, "
            "null and underpowered evidence must remain visible."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlvr-benchmark", type=Path)
    parser.add_argument("--learned-verifier-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scorecard = build_scorecard(
        rlvr_benchmark=args.rlvr_benchmark,
        learned_verifier_summary=args.learned_verifier_summary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n")
    print(json.dumps(scorecard, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
