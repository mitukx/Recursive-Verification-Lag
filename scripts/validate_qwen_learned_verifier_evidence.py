"""Fail-closed semantic validator for Issue #43 learned-verifier evidence.

The validator is intentionally separate from the GPU runner. It recomputes the
scientific quantities needed for the locked primary analysis from retained raw
artifacts and accepts negative/null/underpowered outcomes as valid evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.validate_qwen_alignment_bridge_evidence import (
    _candidate_bank,
    _close,
    _require,
    _validate_eval_logprobs,
    load_json,
    load_jsonl,
    sha256,
    validate_manifest,
    validate_token_drift_rows,
)


ARMS = ("oracle", "fresh", "stale", "shuffled")
FULL_SKIP_REASONS = {
    "stale_verifier_fit_underpowered",
    "fresh_verifier_fit_underpowered",
}
CALIBRATION_SKIP_REASON = "matched_drift_calibration_failed"


def _finite_or_nan_close(a: float, b: float) -> bool:
    a, b = float(a), float(b)
    return (math.isnan(a) and math.isnan(b)) or _close(a, b)


def _rankdata(values: list[float]) -> np.ndarray:
    x = np.asarray(values, float)
    _require(x.ndim == 1 and x.size > 0 and np.isfinite(x).all(), "finite rank data required")
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return ranks


def spearman(values_a: list[float], values_b: list[float]) -> float:
    a = _rankdata(values_a)
    b = _rankdata(values_b)
    if len(a) < 2 or float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _effect_rows(seed_root: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(seed_root / "effect_verifier_scores.jsonl")
    _require(bool(rows), "effect verifier scores are empty")
    seen: set[str] = set()
    for row in rows:
        cid = str(row["candidate_id"])
        _require(cid not in seen, f"duplicate effect candidate {cid}")
        seen.add(cid)
        _require(float(row["trusted_reward"]) in (0.0, 1.0), f"nonbinary trusted reward {cid}")
        for arm in ARMS:
            score = float(row[f"{arm}_score"])
            _require(math.isfinite(score) and 0.0 <= score <= 1.0, f"invalid {arm} score {cid}")
    return sorted(
        rows,
        key=lambda row: (str(row["task_id"]), int(row["candidate_index"])),
    )


def recompute_effect_geometry(
    rows: list[dict[str, Any]],
    *,
    advantage_eps: float,
    clip_advantage: float,
) -> dict[str, dict[str, Any]]:
    _require(advantage_eps > 0 and clip_advantage > 0, "positive GRPO normalization constants required")
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task_id"]), []).append(row)
    for task_rows in by_task.values():
        task_rows.sort(key=lambda row: int(row["candidate_index"]))

    result: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        y_all = np.asarray([float(row["trusted_reward"]) for row in rows], float)
        v_all = np.asarray([float(row[f"{arm}_score"]) for row in rows], float)
        e_all = v_all - y_all
        cov_y_v = float(np.mean((y_all - y_all.mean()) * (v_all - v_all.mean())))
        cov_y_e = float(np.mean((y_all - y_all.mean()) * (e_all - e_all.mean())))
        prompt_rows: list[dict[str, Any]] = []
        within_weighted = 0.0
        advantage_weighted = 0.0
        total = 0
        for task_id, task_rows in by_task.items():
            y = np.asarray([float(row["trusted_reward"]) for row in task_rows], float)
            v = np.asarray([float(row[f"{arm}_score"]) for row in task_rows], float)
            e = v - y
            yc = y - y.mean()
            vc = v - v.mean()
            ec = e - e.mean()
            variance_v = float(np.mean(vc * vc))
            advantage = np.clip(
                vc / math.sqrt(variance_v + advantage_eps),
                -clip_advantage,
                clip_advantage,
            )
            ac = advantage - advantage.mean()
            cov_prompt = float(np.mean(yc * vc))
            cov_error = float(np.mean(yc * ec))
            cov_adv = float(np.mean(yc * ac))
            n = int(y.size)
            total += n
            within_weighted += n * cov_prompt
            advantage_weighted += n * cov_adv
            prompt_rows.append(
                {
                    "task_id": task_id,
                    "occurrences": n,
                    "trusted_mean": float(y.mean()),
                    "score_mean": float(v.mean()),
                    "score_variance": variance_v,
                    "cov_y_v": cov_prompt,
                    "cov_y_e": cov_error,
                    "cov_y_grpo_advantage": cov_adv,
                    "informative_trusted_labels": bool(np.any(y > 0.5) and np.any(y <= 0.5)),
                    "nonconstant_verifier_scores": bool(variance_v > 0.0),
                }
            )
        within = float(within_weighted / total)
        between = 0.0
        for row in prompt_rows:
            weight = float(row["occurrences"]) / total
            between += weight * (
                (float(row["trusted_mean"]) - float(y_all.mean()))
                * (float(row["score_mean"]) - float(v_all.mean()))
            )
        result[arm] = {
            "cov_y_v": cov_y_v,
            "cov_y_e": cov_y_e,
            "brier": float(np.mean((v_all - y_all) ** 2)),
            "score_mean": float(v_all.mean()),
            "score_std": float(v_all.std()),
            "trusted_mean": float(y_all.mean()),
            "grpo_aligned": {
                "pooled_cov_y_v": cov_y_v,
                "within_prompt_cov_y_v_occurrence_weighted": within,
                "between_prompt_cov_y_v": float(between),
                "pooled_cov_reconstruction_error": float(cov_y_v - within - between),
                "mean_prompt_cov_y_v": float(np.mean([row["cov_y_v"] for row in prompt_rows])),
                "mean_prompt_cov_y_e": float(np.mean([row["cov_y_e"] for row in prompt_rows])),
                "mean_prompt_cov_y_grpo_advantage": float(
                    np.mean([row["cov_y_grpo_advantage"] for row in prompt_rows])
                ),
                "occurrence_weighted_cov_y_grpo_advantage": float(advantage_weighted / total),
                "trusted_informative_prompts": int(
                    sum(bool(row["informative_trusted_labels"]) for row in prompt_rows)
                ),
                "nonconstant_score_prompts": int(
                    sum(bool(row["nonconstant_verifier_scores"]) for row in prompt_rows)
                ),
                "advantage_eps": float(advantage_eps),
                "clip_advantage": float(clip_advantage),
                "per_prompt": prompt_rows,
            },
        }
    return result


def _validate_geometry_record(
    recorded: dict[str, Any],
    recomputed: dict[str, Any],
    *,
    arm: str,
) -> None:
    for key in ("cov_y_v", "cov_y_e", "brier", "score_mean", "score_std", "trusted_mean"):
        _require(_close(recorded[key], recomputed[key]), f"{arm} geometry mismatch: {key}")
    r = recorded["grpo_aligned"]
    c = recomputed["grpo_aligned"]
    for key in (
        "pooled_cov_y_v",
        "within_prompt_cov_y_v_occurrence_weighted",
        "between_prompt_cov_y_v",
        "pooled_cov_reconstruction_error",
        "mean_prompt_cov_y_v",
        "mean_prompt_cov_y_e",
        "mean_prompt_cov_y_grpo_advantage",
        "occurrence_weighted_cov_y_grpo_advantage",
        "advantage_eps",
        "clip_advantage",
    ):
        _require(_close(r[key], c[key]), f"{arm} GRPO geometry mismatch: {key}")
    for key in ("trusted_informative_prompts", "nonconstant_score_prompts"):
        _require(int(r[key]) == int(c[key]), f"{arm} GRPO count mismatch: {key}")
    _require(len(r["per_prompt"]) == len(c["per_prompt"]), f"{arm} per-prompt geometry count mismatch")
    by_task = {str(row["task_id"]): row for row in r["per_prompt"]}
    _require(len(by_task) == len(r["per_prompt"]), f"{arm} duplicate per-prompt geometry")
    for expected in c["per_prompt"]:
        task_id = str(expected["task_id"])
        _require(task_id in by_task, f"{arm} missing prompt geometry {task_id}")
        actual = by_task[task_id]
        for key in (
            "trusted_mean",
            "score_mean",
            "score_variance",
            "cov_y_v",
            "cov_y_e",
            "cov_y_grpo_advantage",
        ):
            _require(_close(actual[key], expected[key]), f"{arm}/{task_id} geometry mismatch: {key}")
        for key in ("occurrences", "informative_trusted_labels", "nonconstant_verifier_scores"):
            _require(actual[key] == expected[key], f"{arm}/{task_id} geometry mismatch: {key}")


def _recompute_policy_shift(seed_root: Path) -> dict[str, float]:
    rows = load_jsonl(seed_root / "pre_update_policy_shift_tokens.jsonl")
    _require(bool(rows), "pre-update policy-shift token evidence is empty")
    all_kl: list[float] = []
    all_k3: list[float] = []
    all_abs: list[float] = []
    seen: set[str] = set()
    for row in rows:
        cid = str(row["candidate_id"])
        _require(cid not in seen, f"duplicate policy-shift candidate {cid}")
        seen.add(cid)
        current = np.asarray(row["current_behavior_token_logprobs"], float)
        pre = np.asarray(row["pre_shift_token_logprobs"], float)
        _require(current.ndim == 1 and current.size > 0 and pre.shape == current.shape, f"policy-shift shape mismatch {cid}")
        _require(np.isfinite(current).all() and np.isfinite(pre).all(), f"nonfinite policy-shift logprobs {cid}")
        raw = current - pre
        reverse = np.clip(pre - current, -20.0, 20.0)
        k3 = np.exp(reverse) - 1.0 - reverse
        np.testing.assert_allclose(row["log_ratio_current_over_pre_shift"], raw, atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(row["token_k3_current_vs_pre_shift"], k3, atol=1e-12, rtol=1e-10)
        all_kl.extend(raw.tolist())
        all_k3.extend(k3.tolist())
        all_abs.extend(np.abs(raw).tolist())
    return {
        "sampled_current_vs_pre_shift_kl": float(np.mean(all_kl)),
        "sampled_current_vs_pre_shift_k3": float(np.mean(all_k3)),
        "max_abs_log_ratio": float(np.max(all_abs)),
        "mean_abs_log_ratio": float(np.mean(all_abs)),
        "tokens": len(all_kl),
    }


def _validate_calibration_and_effect(
    seed_root: Path,
    seed_summary: dict[str, Any],
    lock: dict[str, Any],
) -> dict[str, float]:
    calibration = load_json(seed_root / "drift_calibration.json")
    _require("selected" in calibration, "completed seed lacks matched-drift selection")
    declared_lrs = {float(x) for x in lock["drift_calibration"]["learning_rate_grid"]}
    effect_k3: dict[str, float] = {}
    for arm in ARMS:
        _require(arm in calibration["grid"], f"missing calibration arm {arm}")
        for row in calibration["grid"][arm]:
            lr = float(row["lr"])
            _require(lr in declared_lrs, f"undeclared calibration LR {arm}/{lr}")
            path = seed_root / str(row["token_drift_file"])
            _require(path.exists(), f"missing calibration token evidence {path}")
            recomputed = validate_token_drift_rows(load_jsonl(path))
            for key, value in recomputed.items():
                _require(_close(row[key], value), f"calibration drift mismatch {arm}/{key}")
        selected_lr = float(seed_summary["selected_lrs"][arm])
        _require(selected_lr in declared_lrs, f"selected LR outside grid {arm}")
        _require(_close(selected_lr, calibration["selected"][arm]["lr"], atol=0, rtol=0), f"selected LR mismatch {arm}")
        arm_record = seed_summary["arms"][arm]
        _require(_close(arm_record["lr"], selected_lr, atol=0, rtol=0), f"arm LR mismatch {arm}")
        grad_norm = float(arm_record["train_metrics"]["grad_norm"])
        _require(math.isfinite(grad_norm), f"nonfinite effect gradient norm {arm}")
        effect_path = seed_root / str(arm_record["effect_token_drift_file"])
        _require(effect_path.exists(), f"missing effect drift evidence {arm}")
        recomputed = validate_token_drift_rows(load_jsonl(effect_path))
        for key, value in recomputed.items():
            _require(_close(arm_record["effect_drift"][key], value), f"effect drift mismatch {arm}/{key}")
        effect_k3[arm] = float(recomputed["post_update_k3"])
    return effect_k3


def _validate_terminal_preferences(
    seed_root: Path,
    seed_summary: dict[str, Any],
) -> tuple[dict[str, float], int]:
    bank = _candidate_bank(seed_root)
    labels = load_jsonl(seed_root / "evaluation_candidate_trusted_labels.jsonl")
    label_by_id: dict[str, float] = {}
    for row in labels:
        cid = str(row["candidate_id"])
        _require(cid not in label_by_id, f"duplicate evaluation trusted label {cid}")
        reward = float(row["trusted_reward"])
        _require(reward in (0.0, 1.0), f"nonbinary evaluation reward {cid}")
        label_by_id[cid] = reward
    _require(set(label_by_id) == set(bank), "evaluation trusted-label coverage mismatch")

    eval_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARMS:
        path = seed_root / str(seed_summary["arms"][arm]["evaluation_candidate_logprob_file"])
        _require(path.exists(), f"missing evaluation logprob evidence {arm}")
        eval_rows[arm] = _validate_eval_logprobs(path, arm, bank)

    grouped: dict[str, list[str]] = {}
    for cid in bank:
        grouped.setdefault(cid.rsplit(":candidate-", 1)[0], []).append(cid)
    preference_rows = load_jsonl(seed_root / "evaluation_preference.jsonl")
    recorded: dict[tuple[str, str], dict[str, Any]] = {}
    for row in preference_rows:
        key = (str(row["task_id"]), str(row["arm"]))
        _require(key not in recorded, f"duplicate evaluation preference {key}")
        _require(key[1] in ARMS, f"unknown evaluation arm {key[1]}")
        recorded[key] = row
    expected_keys = {(task_id, arm) for task_id in grouped for arm in ARMS}
    _require(set(recorded) == expected_keys, "evaluation preference coverage mismatch")

    shifts: dict[str, list[float]] = {arm: [] for arm in ARMS}
    informative_prompts = 0
    for task_id, ids in grouped.items():
        ids = sorted(ids, key=lambda cid: int(cid.rsplit(":candidate-", 1)[1]))
        y = np.asarray([label_by_id[cid] for cid in ids], float)
        informative = bool(np.any(y > 0.5) and np.any(y <= 0.5))
        informative_prompts += int(informative)
        positive = y > 0.5
        for arm in ARMS:
            row = recorded[(task_id, arm)]
            np.testing.assert_allclose(row["trusted_rewards"], y, atol=0, rtol=0)
            _require(bool(row["informative"]) == informative, f"informative flag mismatch {task_id}/{arm}")
            if not informative:
                for field in ("baseline_margin", "post_margin", "preference_shift"):
                    _require(row.get(field) is None, f"uninformative row has {field}: {task_id}/{arm}")
                continue
            base = np.asarray([eval_rows[arm][cid]["baseline_sequence_logprob"] for cid in ids], float)
            post = np.asarray([eval_rows[arm][cid]["post_update_sequence_logprob"] for cid in ids], float)
            base_margin = float(base[positive].mean() - base[~positive].mean())
            post_margin = float(post[positive].mean() - post[~positive].mean())
            shift = post_margin - base_margin
            _require(_close(row["baseline_margin"], base_margin), f"baseline margin mismatch {task_id}/{arm}")
            _require(_close(row["post_margin"], post_margin), f"post margin mismatch {task_id}/{arm}")
            _require(_close(row["preference_shift"], shift), f"preference shift mismatch {task_id}/{arm}")
            shifts[arm].append(shift)

    means = {
        arm: float(np.mean(shifts[arm])) if shifts[arm] else float("nan")
        for arm in ARMS
    }
    for arm in ARMS:
        _require(
            _finite_or_nan_close(means[arm], seed_summary["mean_preference_shift"][arm]),
            f"mean preference shift mismatch {arm}",
        )
    _require(
        int(seed_summary["informative_evaluation_prompts"]) == informative_prompts,
        "informative terminal prompt count mismatch",
    )
    return means, informative_prompts


def validate_seed(
    seed_root: Path,
    seed_summary: dict[str, Any],
    lock: dict[str, Any],
) -> dict[str, Any]:
    _require(seed_root.exists(), f"missing seed directory {seed_root.name}")
    skip_reason = seed_summary.get("skip_reason")
    if skip_reason in FULL_SKIP_REASONS:
        _require(not bool(seed_summary.get("eligible_primary_seed")), "underpowered verifier-fit seed marked eligible")
        return {"seed": int(seed_summary["seed"]), "status": str(skip_reason), "validated_primary": False}

    effect_rows = _effect_rows(seed_root)
    geometry_file = load_json(seed_root / "pre_update_verifier_geometry.json")
    recorded_geometry = seed_summary["pre_update_geometry"]
    _require(set(geometry_file) == set(recorded_geometry), "geometry file/summary keys differ")
    for arm in ARMS:
        _require(arm in recorded_geometry, f"missing geometry arm {arm}")
    grpo_ref = recorded_geometry[ARMS[0]]["grpo_aligned"]
    recomputed_geometry = recompute_effect_geometry(
        effect_rows,
        advantage_eps=float(grpo_ref["advantage_eps"]),
        clip_advantage=float(grpo_ref["clip_advantage"]),
    )
    for arm in ARMS:
        _validate_geometry_record(recorded_geometry[arm], recomputed_geometry[arm], arm=arm)
        _validate_geometry_record(geometry_file[arm], recomputed_geometry[arm], arm=arm)

    policy_shift = _recompute_policy_shift(seed_root)
    policy_file = load_json(seed_root / "pre_update_policy_shift.json")
    for key, value in policy_shift.items():
        _require(_close(policy_file[key], value), f"policy-shift file mismatch: {key}")
        _require(_close(seed_summary["pre_update_policy_shift"][key], value), f"policy-shift summary mismatch: {key}")

    if skip_reason == CALIBRATION_SKIP_REASON:
        calibration = load_json(seed_root / "drift_calibration.json")
        _require(bool(calibration.get("selection_error")), "calibration-failed seed lacks error")
        _require(not bool(seed_summary.get("eligible_primary_seed")), "calibration-failed seed marked eligible")
        return {
            "seed": int(seed_summary["seed"]),
            "status": CALIBRATION_SKIP_REASON,
            "validated_primary": False,
        }
    _require(skip_reason is None, f"unknown seed skip reason {skip_reason}")

    effect_k3 = _validate_calibration_and_effect(seed_root, seed_summary, lock)
    mean_shift, informative_prompts = _validate_terminal_preferences(seed_root, seed_summary)

    positive_k3 = [value for value in effect_k3.values() if value > 0 and math.isfinite(value)]
    ratio = max(positive_k3) / min(positive_k3) if len(positive_k3) == len(ARMS) else float("inf")
    _require(_close(seed_summary["effect_k3_ratio"], ratio), "effect k3 ratio mismatch")

    covariances = [float(recomputed_geometry[arm]["cov_y_v"]) for arm in ARMS]
    shifts = [float(mean_shift[arm]) for arm in ARMS]
    k3_values = [float(effect_k3[arm]) for arm in ARMS]
    geometry_rho = spearman(covariances, shifts)
    k3_rho = spearman(k3_values, shifts)
    _require(
        _finite_or_nan_close(seed_summary["geometry_spearman"], geometry_rho),
        "geometry Spearman mismatch",
    )
    _require(
        _finite_or_nan_close(seed_summary["effect_k3_spearman"], k3_rho),
        "effect-k3 Spearman mismatch",
    )

    policy_ok = (
        math.isfinite(float(policy_shift["sampled_current_vs_pre_shift_k3"]))
        and float(policy_shift["sampled_current_vs_pre_shift_k3"]) > 0
    )
    eligible = (
        informative_prompts >= int(lock["terminal_evaluation"]["minimum_informative_prompts"])
        and ratio <= 1.5
        and policy_ok
    )
    _require(bool(seed_summary["eligible_primary_seed"]) == eligible, "seed eligibility mismatch")
    for arm in ARMS:
        _require((seed_root / f"{arm}_terminal_greedy_unscored.json").exists(), f"missing terminal greedy evidence {arm}")
    _require((seed_root / "baseline_greedy_unscored.json").exists(), "missing baseline greedy evidence")
    return {
        "seed": int(seed_summary["seed"]),
        "status": "complete_seed",
        "validated_primary": True,
        "eligible_primary_seed": eligible,
        "geometry_spearman": geometry_rho,
        "effect_k3_spearman": k3_rho,
        "effect_candidates": len(effect_rows),
        "informative_terminal_prompts": informative_prompts,
    }


def validate_evidence(
    root: Path,
    protocol: Path,
    *,
    expected_research_sha: str | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    lock = load_json(protocol)
    manifest = validate_manifest(root)
    _require(manifest["protocol_sha256"] == sha256(protocol), "protocol hash mismatch")

    status = str(manifest.get("status"))
    if status == "failed":
        failure = load_json(root / "failure.json")
        _require(failure.get("status") == "failed", "failure record status mismatch")
        _require(bool(failure.get("traceback")), "failure traceback missing")
        return {
            "valid": True,
            "execution_status": "failed",
            "scientific_result": "not_evaluable",
            "failure_retained": True,
        }
    _require(status == "completed", f"unknown execution status {status}")

    env = load_json(root / "environment.json")
    _require(env["systems_source_sha"] == lock["systems_source_sha"], "systems source SHA mismatch")
    if expected_research_sha is not None:
        _require(env["research_source_sha"] == expected_research_sha, "research source SHA mismatch")
    _require(bool(env.get("device")), "GPU device missing")
    _require(int(env.get("device_count", 0)) >= 1, "GPU device count missing")
    deps = env.get("dependencies", {})
    for name in ("torch", "transformers", "datasets", "accelerate", "huggingface_hub", "numpy"):
        _require(deps.get(name), f"dependency version missing: {name}")
    _require((root / "nvidia-smi.txt").stat().st_size > 0, "nvidia-smi evidence empty")
    _require((root / "pip-freeze.txt").stat().st_size > 0, "pip-freeze evidence empty")
    _require((root / "task_manifest.json").exists(), "task manifest missing")

    summary = load_json(root / "summary.json")
    seed_rows = summary.get("seed_results", [])
    declared = [int(x) for x in lock["seeds"]]
    _require([int(row["seed"]) for row in seed_rows] == declared, "summary seed order/set mismatch")
    details = []
    for row in seed_rows:
        seed_summary = load_json(root / f'seed-{int(row["seed"])}' / "seed_summary.json")
        _require(seed_summary.get("seed") == row.get("seed"), "seed summary identity mismatch")
        details.append(validate_seed(root / f'seed-{int(row["seed"])}', seed_summary, lock))

    eligible = [row for row in seed_rows if row.get("eligible_primary_seed")]
    _require(int(summary["eligible_seeds"]) == len(eligible), "eligible seed count mismatch")
    required = int(summary["required_eligible_seeds"])
    _require(required == 2, "locked learned-verifier primary requires two eligible seeds")
    sufficient = len(eligible) >= required
    _require(bool(summary["primary_evidence_sufficient"]) == sufficient, "primary sufficiency mismatch")

    if not sufficient:
        _require(summary["mean_geometry_spearman"] is None, "underpowered geometry mean must be null")
        _require(summary["mean_effect_k3_spearman"] is None, "underpowered k3 mean must be null")
        _require(summary["primary_direction_passed"] is None, "underpowered result cannot pass/fail")
        scientific = "underpowered"
    else:
        geometry_values = [float(row["geometry_spearman"]) for row in eligible]
        k3_values = [float(row["effect_k3_spearman"]) for row in eligible]
        mean_geometry = float(np.nanmean(geometry_values))
        mean_k3 = float(np.nanmean(k3_values))
        _require(_finite_or_nan_close(summary["mean_geometry_spearman"], mean_geometry), "primary geometry mean mismatch")
        _require(_finite_or_nan_close(summary["mean_effect_k3_spearman"], mean_k3), "primary k3 mean mismatch")
        passed = bool(
            np.isfinite(mean_geometry)
            and np.isfinite(mean_k3)
            and mean_geometry > 0
            and mean_geometry > mean_k3
        )
        _require(bool(summary["primary_direction_passed"]) == passed, "primary direction flag mismatch")
        scientific = "direction_passed" if passed else "direction_failed"

    return {
        "valid": True,
        "execution_status": "completed",
        "scientific_result": scientific,
        "eligible_seeds": len(eligible),
        "seed_details": details,
        "research_source_sha": env["research_source_sha"],
        "systems_source_sha": env["systems_source_sha"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-research-sha")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_evidence(
        args.root,
        args.protocol,
        expected_research_sha=args.expected_research_sha,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
