"""Locked real-model learned-verifier extension for Recursive Verification Lag.

This follows configs/qwen_learned_verifier_bridge_v1.json. It uses the same
causal LM sequentially as a policy and as a version-frozen representation
backbone for a separately trained neural reward head. It does not modify the
protocol after outcomes are observed and retains failed/underpowered seeds.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np

from scripts.run_qwen_alignment_bridge import (
    _current_token_logps,
    _set_optimizer_lr,
    candidate_preference_shift,
    post_update_drift_evidence,
    sequence_logprob,
    save_json,
    save_jsonl,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("oracle", "fresh", "stale", "shuffled")


def verifier_geometry_metrics(
    trusted_rewards: list[float] | np.ndarray,
    verifier_scores: list[float] | np.ndarray,
    *,
    ece_bins: int = 10,
) -> dict[str, float]:
    y = np.asarray(trusted_rewards, float)
    v = np.asarray(verifier_scores, float)
    if (
        y.ndim != 1
        or v.shape != y.shape
        or len(y) < 2
        or not np.isfinite(y).all()
        or not np.isfinite(v).all()
        or np.any(y < 0)
        or np.any(y > 1)
        or np.any(v < 0)
        or np.any(v > 1)
        or ece_bins <= 0
    ):
        raise ValueError("aligned finite y,v in [0,1] and ece_bins>0 required")
    e = v - y
    cov_y_v = float(np.mean((y - y.mean()) * (v - v.mean())))
    cov_y_e = float(np.mean((y - y.mean()) * (e - e.mean())))
    brier = float(np.mean((v - y) ** 2))
    edges = np.linspace(0.0, 1.0, ece_bins + 1)
    ece = 0.0
    for index in range(ece_bins):
        if index == ece_bins - 1:
            mask = (v >= edges[index]) & (v <= edges[index + 1])
        else:
            mask = (v >= edges[index]) & (v < edges[index + 1])
        if mask.any():
            ece += float(mask.mean()) * abs(float(v[mask].mean() - y[mask].mean()))
    positives = np.flatnonzero(y > 0.5)
    negatives = np.flatnonzero(y <= 0.5)
    if len(positives) and len(negatives):
        wins = 0.0
        total = 0
        for i in positives:
            for j in negatives:
                total += 1
                if v[i] > v[j]:
                    wins += 1.0
                elif v[i] == v[j]:
                    wins += 0.5
        ranking = wins / total
    else:
        ranking = float("nan")
    return {
        "cov_y_v": cov_y_v,
        "mean_error": float(e.mean()),
        "error_rmse": float(np.sqrt(np.mean(e * e))),
        "cov_y_e": cov_y_e,
        "brier": brier,
        "ece_10": float(ece),
        "ranking_accuracy": float(ranking),
        "score_mean": float(v.mean()),
        "score_std": float(v.std()),
        "trusted_mean": float(y.mean()),
    }


def grpo_aligned_geometry_metrics(
    groups: list[dict[str, Any]],
    scores: dict[str, list[float]],
    *,
    advantage_eps: float = 1e-6,
    clip_advantage: float = 5.0,
) -> dict[str, Any]:
    """Decompose pooled verifier geometry into components visible to GRPO.

    HFCausalLMGRPOTrainer centers and standardizes rewards within each prompt before
    applying the clipped surrogate. Prompt-level score offsets therefore contribute
    to pooled Cov(y,v) but not to the actual GRPO advantage. This diagnostic is
    secondary only and never participates in LR selection or arm selection.
    """
    if advantage_eps <= 0 or clip_advantage <= 0 or not groups:
        raise ValueError("positive advantage_eps/clip_advantage and nonempty groups required")
    prompt_rows: list[dict[str, Any]] = []
    all_y: list[float] = []
    all_v: list[float] = []
    total = 0
    within_weighted = 0.0
    advantage_cov_weighted = 0.0
    for group in groups:
        task_id = str(group["task_id"])
        y = np.asarray(group["trusted_rewards"], float)
        v = np.asarray(scores[task_id], float)
        if (
            y.ndim != 1
            or v.shape != y.shape
            or y.size == 0
            or not np.isfinite(y).all()
            or not np.isfinite(v).all()
            or np.any(y < 0)
            or np.any(y > 1)
            or np.any(v < 0)
            or np.any(v > 1)
        ):
            raise ValueError(f"invalid aligned prompt geometry for {task_id}")
        e = v - y
        y_centered = y - y.mean()
        v_centered = v - v.mean()
        e_centered = e - e.mean()
        variance_v = float(np.mean(v_centered * v_centered))
        scale = math.sqrt(variance_v + advantage_eps)
        advantage = np.clip(v_centered / scale, -clip_advantage, clip_advantage)
        advantage_centered = advantage - advantage.mean()
        cov_y_v = float(np.mean(y_centered * v_centered))
        cov_y_e = float(np.mean(y_centered * e_centered))
        cov_y_advantage = float(np.mean(y_centered * advantage_centered))
        n = int(y.size)
        total += n
        within_weighted += n * cov_y_v
        advantage_cov_weighted += n * cov_y_advantage
        all_y.extend(y.tolist())
        all_v.extend(v.tolist())
        prompt_rows.append(
            {
                "task_id": task_id,
                "occurrences": n,
                "trusted_mean": float(y.mean()),
                "score_mean": float(v.mean()),
                "score_variance": variance_v,
                "cov_y_v": cov_y_v,
                "cov_y_e": cov_y_e,
                "cov_y_grpo_advantage": cov_y_advantage,
                "informative_trusted_labels": bool(np.any(y > 0.5) and np.any(y <= 0.5)),
                "nonconstant_verifier_scores": bool(variance_v > 0.0),
            }
        )

    y_all = np.asarray(all_y, float)
    v_all = np.asarray(all_v, float)
    pooled_cov = float(np.mean((y_all - y_all.mean()) * (v_all - v_all.mean())))
    within_occurrence_weighted = float(within_weighted / total)
    between = 0.0
    for row in prompt_rows:
        weight = float(row["occurrences"]) / total
        between += weight * (
            (float(row["trusted_mean"]) - float(y_all.mean()))
            * (float(row["score_mean"]) - float(v_all.mean()))
        )
    reconstruction_error = float(
        pooled_cov - (within_occurrence_weighted + between)
    )
    return {
        "pooled_cov_y_v": pooled_cov,
        "within_prompt_cov_y_v_occurrence_weighted": within_occurrence_weighted,
        "between_prompt_cov_y_v": float(between),
        "pooled_cov_reconstruction_error": reconstruction_error,
        "mean_prompt_cov_y_v": float(np.mean([row["cov_y_v"] for row in prompt_rows])),
        "mean_prompt_cov_y_e": float(np.mean([row["cov_y_e"] for row in prompt_rows])),
        "mean_prompt_cov_y_grpo_advantage": float(
            np.mean([row["cov_y_grpo_advantage"] for row in prompt_rows])
        ),
        "occurrence_weighted_cov_y_grpo_advantage": float(
            advantage_cov_weighted / total
        ),
        "trusted_informative_prompts": int(
            sum(bool(row["informative_trusted_labels"]) for row in prompt_rows)
        ),
        "nonconstant_score_prompts": int(
            sum(bool(row["nonconstant_verifier_scores"]) for row in prompt_rows)
        ),
        "advantage_eps": float(advantage_eps),
        "clip_advantage": float(clip_advantage),
        "per_prompt": prompt_rows,
    }


def pairwise_reversal_rate(
    first: list[float] | np.ndarray,
    second: list[float] | np.ndarray,
) -> dict[str, float]:
    a = np.asarray(first, float)
    b = np.asarray(second, float)
    if a.ndim != 1 or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("aligned finite score vectors required")
    eligible = reversals = 0
    ties = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da, db = a[i] - a[j], b[i] - b[j]
            if abs(da) <= 1e-12 or abs(db) <= 1e-12:
                ties += 1
                continue
            eligible += 1
            reversals += int((da > 0) != (db > 0))
    return {
        "eligible_pairs": eligible,
        "reversals": reversals,
        "reversal_rate": float(reversals / eligible) if eligible else float("nan"),
        "tie_excluded_pairs": ties,
    }


def _rankdata(values: list[float]) -> np.ndarray:
    x = np.asarray(values, float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("finite 1D values required")
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        raise ValueError("aligned vectors with at least 2 elements required")
    a = _rankdata(values_a)
    b = _rankdata(values_b)
    if np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def select_matched_drift_lrs_multi(
    grid: dict[str, list[dict[str, float]]],
    *,
    target_fraction: float = 0.8,
) -> dict[str, Any]:
    if set(grid) != set(ARMS):
        raise ValueError("exact oracle/fresh/stale/shuffled arms required")
    maxima: dict[str, float] = {}
    valid: dict[str, list[dict[str, float]]] = {}
    for arm in ARMS:
        usable = [
            row
            for row in grid[arm]
            if float(row["lr"]) > 0
            and np.isfinite(float(row["post_update_k3"]))
            and float(row["post_update_k3"]) > 0
        ]
        if not usable:
            raise ValueError(f"no positive finite calibration drift for {arm}")
        valid[arm] = usable
        maxima[arm] = max(float(row["post_update_k3"]) for row in usable)
    target = target_fraction * min(maxima.values())
    selected = {}
    for arm in ARMS:
        selected[arm] = min(
            valid[arm],
            key=lambda row: (
                abs(math.log(float(row["post_update_k3"]) / target)),
                float(row["lr"]),
            ),
        )
    return {"target_k3": target, "arm_max_k3": maxima, "selected": selected}


def deterministic_prompt_shuffle(
    scores_by_task: dict[str, list[float]],
    *,
    seed: int,
) -> dict[str, list[float]]:
    result = {}
    for index, task_id in enumerate(sorted(scores_by_task)):
        values = np.asarray(scores_by_task[task_id], float)
        rng = np.random.default_rng(seed + index * 1009)
        permutation = rng.permutation(len(values))
        result[task_id] = values[permutation].tolist()
    return result


def _flatten_labels(groups: list[dict[str, Any]]) -> np.ndarray:
    values = [float(y) for group in groups for y in group["trusted_rewards"]]
    return np.asarray(values, float)


def _fit_eligible(groups: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    y = _flatten_labels(groups)
    positives = int(np.sum(y > 0.5))
    negatives = int(np.sum(y <= 0.5))
    eligible = (
        positives >= int(config["minimum_positive_labels"])
        and negatives >= int(config["minimum_negative_labels"])
    )
    return {
        "eligible": eligible,
        "positives": positives,
        "negatives": negatives,
        "examples": len(y),
    }


def _extract_group_features(model: Any, groups: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    import torch

    was_training = model.training
    model.eval()
    output: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for group in groups:
            rows = []
            for generation in group["generations"]:
                prompt = [int(x) for x in generation.metadata["prompt_token_ids"]]
                response = [int(x) for x in generation.metadata["response_token_ids"]]
                if not prompt or not response:
                    raise ValueError("verifier representation requires nonempty tokens")
                sequence = torch.tensor(
                    [prompt + response],
                    dtype=torch.long,
                    device=next(model.parameters()).device,
                )
                hidden = model(input_ids=sequence, output_hidden_states=True).hidden_states[-1][0]
                start = len(prompt)
                feature = hidden[start : start + len(response)].float().mean(dim=0)
                rows.append(feature.detach().cpu().numpy())
            output[group["task_id"]] = np.stack(rows, axis=0)
    if was_training:
        model.train()
    return output


def _stack_features(groups: list[dict[str, Any]], features: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([features[group["task_id"]] for group in groups], axis=0)


def _make_head(torch: Any, hidden_size: int, width: int):
    return torch.nn.Sequential(
        torch.nn.LayerNorm(hidden_size),
        torch.nn.Linear(hidden_size, width),
        torch.nn.GELU(),
        torch.nn.Linear(width, 1),
    )


def train_verifier_head(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    config: dict[str, Any],
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    import torch

    if (
        features.ndim != 2
        or labels.ndim != 1
        or features.shape[0] != labels.shape[0]
        or not np.isfinite(features).all()
        or not np.isfinite(labels).all()
    ):
        raise ValueError("finite aligned verifier features/labels required")
    positives = int(np.sum(labels > 0.5))
    negatives = int(np.sum(labels <= 0.5))
    if positives == 0 or negatives == 0:
        raise ValueError("verifier head training requires both classes")
    torch.manual_seed(seed)
    head = _make_head(torch, features.shape[1], int(config["hidden_width"])).cpu()
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    losses = []
    head.train()
    for _ in range(int(config["full_batch_steps"])):
        optimizer.zero_grad(set_to_none=True)
        logits = head(x)
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite verifier training loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    head.eval()
    return head, {
        "steps": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "positives": positives,
        "negatives": negatives,
        "pos_weight": negatives / positives,
        "parameters": sum(p.numel() for p in head.parameters()),
        "hidden_size": features.shape[1],
    }


def score_head(head: Any, features: dict[str, np.ndarray]) -> dict[str, list[float]]:
    import torch

    result = {}
    head.eval()
    with torch.inference_mode():
        for task_id, matrix in features.items():
            x = torch.tensor(matrix, dtype=torch.float32)
            result[task_id] = torch.sigmoid(head(x).squeeze(1)).numpy().astype(float).tolist()
    return result


def _score_rows(
    groups: list[dict[str, Any]],
    scores: dict[str, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    rows = []
    for group in groups:
        task_id = group["task_id"]
        for index, y in enumerate(group["trusted_rewards"]):
            rows.append(
                {
                    "task_id": task_id,
                    "candidate_index": index,
                    "candidate_id": f"{task_id}:candidate-{index}",
                    "trusted_reward": float(y),
                    **{f"{arm}_score": float(scores[arm][task_id][index]) for arm in ARMS},
                }
            )
    return rows


def _geometry_from_scores(
    groups: list[dict[str, Any]],
    scores: dict[str, dict[str, list[float]]],
    *,
    advantage_eps: float = 1e-6,
    clip_advantage: float = 5.0,
) -> dict[str, Any]:
    y = _flatten_labels(groups)
    result = {}
    for arm in ARMS:
        v = np.asarray(
            [value for group in groups for value in scores[arm][group["task_id"]]],
            float,
        )
        result[arm] = verifier_geometry_metrics(y, v)
        result[arm]["grpo_aligned"] = grpo_aligned_geometry_metrics(
            groups,
            scores[arm],
            advantage_eps=advantage_eps,
            clip_advantage=clip_advantage,
        )
    fresh = np.asarray(
        [value for group in groups for value in scores["fresh"][group["task_id"]]],
        float,
    )
    stale = np.asarray(
        [value for group in groups for value in scores["stale"][group["task_id"]]],
        float,
    )
    result["fresh_vs_stale_ranking"] = pairwise_reversal_rate(fresh, stale)
    return result


def _verified_from_scores(
    groups: list[dict[str, Any]],
    arm: str,
    scores: dict[str, list[float]],
    VerifiedGeneration: Any,
) -> list[Any]:
    verified = []
    for group in groups:
        for index, (generation, y) in enumerate(
            zip(group["generations"], group["trusted_rewards"])
        ):
            verified.append(
                VerifiedGeneration(
                    generation=generation,
                    reward=float(scores[group["task_id"]][index]),
                    verifier_latency_s=0.0,
                    verifier_version={"oracle": 0, "fresh": 2, "stale": 1, "shuffled": 3}[arm],
                    metadata={
                        "arm": arm,
                        "candidate_index": index,
                        "trusted_reward_evaluation_only": float(y),
                    },
                )
            )
    return verified


def _policy_shift_on_current_candidates(
    model: Any,
    groups: list[dict[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    all_kl = []
    all_k3 = []
    all_abs = []
    for group in groups:
        for index, generation in enumerate(group["generations"]):
            current = np.asarray(generation.metadata["response_token_logprobs"], float)
            baseline = _current_token_logps(model, generation)
            if current.shape != baseline.shape:
                raise ValueError("policy-shift token shape mismatch")
            log_current_over_baseline = current - baseline
            reverse = baseline - current
            clipped_reverse = np.clip(reverse, -20.0, 20.0)
            k3 = np.exp(clipped_reverse) - 1.0 - clipped_reverse
            rows.append(
                {
                    "task_id": group["task_id"],
                    "candidate_index": index,
                    "candidate_id": f'{group["task_id"]}:candidate-{index}',
                    "current_behavior_token_logprobs": current.tolist(),
                    "pre_shift_token_logprobs": baseline.tolist(),
                    "log_ratio_current_over_pre_shift": log_current_over_baseline.tolist(),
                    "token_k3_current_vs_pre_shift": k3.tolist(),
                }
            )
            all_kl.extend(log_current_over_baseline.tolist())
            all_k3.extend(k3.tolist())
            all_abs.extend(np.abs(log_current_over_baseline).tolist())
    return (
        {
            "sampled_current_vs_pre_shift_kl": float(np.mean(all_kl)),
            "sampled_current_vs_pre_shift_k3": float(np.mean(all_k3)),
            "max_abs_log_ratio": float(np.max(all_abs)),
            "mean_abs_log_ratio": float(np.mean(all_abs)),
            "tokens": len(all_kl),
        },
        rows,
    )


def _save_eval_logprobs(
    seed_root: Path,
    arm: str,
    model: Any,
    groups: list[dict[str, Any]],
) -> tuple[str, dict[str, list[float]]]:
    filename = f"{arm}_evaluation_candidate_token_logprobs.jsonl"
    rows = []
    means: dict[str, list[float]] = {}
    for group in groups:
        task_values = []
        for index, generation in enumerate(group["generations"]):
            baseline = np.asarray(generation.metadata["response_token_logprobs"], float)
            current = _current_token_logps(model, generation)
            task_values.append(sequence_logprob(current))
            rows.append(
                {
                    "task_id": group["task_id"],
                    "candidate_index": index,
                    "candidate_id": f'{group["task_id"]}:candidate-{index}',
                    "arm": arm,
                    "response": generation.response,
                    "response_token_ids": generation.metadata["response_token_ids"],
                    "baseline_token_logprobs": baseline.tolist(),
                    "post_update_token_logprobs": current.tolist(),
                    "baseline_mean_token_logprob": float(np.mean(baseline)),
                    "post_update_mean_token_logprob": float(np.mean(current)),
                    "baseline_sequence_logprob": sequence_logprob(baseline),
                    "post_update_sequence_logprob": sequence_logprob(current),
                }
            )
        means[group["task_id"]] = task_values
    save_jsonl(seed_root, filename, rows)
    return filename, means


async def run(lock: dict[str, Any], systems: Path, output: Path) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from src.rvl_systems.hf_backend import HFLocalBackend
    from src.rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
    from src.rvl_systems.rlvr_benchmark import (
        build_math_prompt,
        gsm8k_reference_answer,
        response_reward,
    )
    from src.rvl_systems.types import VerifiedGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU/MPS substitution prohibited")
    output.mkdir(parents=True, exist_ok=False)
    (output / "nvidia-smi.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    (output / "pip-freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    deps = {}
    for name in ("torch", "transformers", "datasets", "accelerate", "huggingface_hub", "numpy"):
        try:
            deps[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            deps[name] = None
    save_json(
        output,
        "environment.json",
        {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "dependencies": deps,
            "systems_source_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=systems, text=True
            ).strip(),
            "research_source_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
    )

    model_path = snapshot_download(lock["model"], revision=lock["model_revision"])
    data = load_dataset(lock["dataset"], "main", revision=lock["dataset_revision"])
    partition = lock["task_partition"]

    def tasks(split: str, indices: list[int], prefix: str):
        source = data[split]
        return [
            {
                "task_id": f"{prefix}-{index}",
                "prompt": build_math_prompt(str(source[int(index)]["question"])),
                "answer": gsm8k_reference_answer(str(source[int(index)]["answer"])),
            }
            for index in indices
        ]

    task_sets = {
        "verifier_fit": tasks("train", partition["verifier_fit_train_rows"], "verifier-fit"),
        "verifier_holdout": tasks(
            "train", partition["verifier_holdout_train_rows"], "verifier-holdout"
        ),
        "shift": tasks(
            "train", partition["stale_inducing_shift_train_rows"], "stale-shift"
        ),
        "calibration": tasks(
            "train", partition["lr_calibration_train_rows"], "calibration"
        ),
        "effect": tasks("train", partition["effect_train_rows"], "effect"),
        "evaluation": tasks("test", partition["evaluation_test_rows"], "evaluation"),
    }
    train_id_sets = [
        set(partition[key])
        for key in (
            "verifier_fit_train_rows",
            "verifier_holdout_train_rows",
            "stale_inducing_shift_train_rows",
            "lr_calibration_train_rows",
            "effect_train_rows",
        )
    ]
    for i in range(len(train_id_sets)):
        for j in range(i + 1, len(train_id_sets)):
            if train_id_sets[i] & train_id_sets[j]:
                raise AssertionError("learned-verifier train partitions overlap")
    save_json(
        output,
        "task_manifest.json",
        {
            key: [
                {"task_id": row["task_id"], "prompt": row["prompt"]}
                if key == "evaluation"
                else row
                for row in rows
            ]
            for key, rows in task_sets.items()
        },
    )

    head_cfg = lock["learned_verifier"]["head"]
    lr_grid = [float(x) for x in lock["drift_calibration"]["learning_rate_grid"]]
    all_seed_results = []

    for seed_value in lock["seeds"]:
        seed = int(seed_value)
        seed_root = output / f"seed-{seed}"
        seed_root.mkdir(parents=True, exist_ok=False)
        random_seed = seed * 1_000_003
        backend = HFLocalBackend(
            model_path,
            max_new_tokens=int(lock["max_new_tokens"]),
            precision=lock["precision"],
        )
        backend.ensure_loaded()
        trainer = HFCausalLMGRPOTrainer(
            backend.model,
            config=HFTTrainerConfig(learning_rate=min(lr_grid)),
        )
        pre_shift_state = trainer.snapshot_training_state()

        async def generate_groups(task_rows, *, seed_offset: int, expose_labels: bool):
            groups = []
            for i, task in enumerate(task_rows):
                generations = await backend.generate(
                    task["task_id"],
                    task["prompt"],
                    n=int(lock["samples_per_prompt"]),
                    temperature=float(lock["temperature"]),
                    seed=random_seed + seed_offset + i * 10_007,
                )
                labels = (
                    [response_reward(g.response, task["answer"]) for g in generations]
                    if expose_labels
                    else None
                )
                groups.append(
                    {
                        "task_id": task["task_id"],
                        "generations": generations,
                        "trusted_rewards": labels,
                    }
                )
            return groups

        baseline_fit = await generate_groups(
            task_sets["verifier_fit"], seed_offset=10_000, expose_labels=True
        )
        stale_fit_status = _fit_eligible(baseline_fit, head_cfg)
        save_jsonl(
            seed_root,
            "stale_verifier_fit_rollouts.jsonl",
            [
                {
                    "task_id": g["task_id"],
                    "trusted_rewards": g["trusted_rewards"],
                    "generations": [asdict(x) for x in g["generations"]],
                }
                for g in baseline_fit
            ],
        )
        if not stale_fit_status["eligible"]:
            record = {
                "seed": seed,
                "eligible_primary_seed": False,
                "skip_reason": "stale_verifier_fit_underpowered",
                "stale_fit": stale_fit_status,
            }
            save_json(seed_root, "seed_summary.json", record)
            all_seed_results.append(record)
            del trainer, backend
            gc.collect()
            torch.cuda.empty_cache()
            continue

        stale_fit_features = _extract_group_features(trainer.model, baseline_fit)
        stale_head, stale_training = train_verifier_head(
            _stack_features(baseline_fit, stale_fit_features),
            _flatten_labels(baseline_fit),
            config=head_cfg,
            seed=random_seed + int(head_cfg["initialization_seed_offset"]),
        )
        torch.save(stale_head.state_dict(), seed_root / "stale_verifier_head.pt")
        save_json(seed_root, "stale_verifier_training.json", stale_training)
        del stale_fit_features
        gc.collect()

        shift = await generate_groups(
            task_sets["shift"], seed_offset=20_000, expose_labels=True
        )
        save_jsonl(
            seed_root,
            "stale_inducing_shift_rollouts.jsonl",
            [
                {
                    "task_id": g["task_id"],
                    "trusted_rewards": g["trusted_rewards"],
                    "generations": [asdict(x) for x in g["generations"]],
                }
                for g in shift
            ],
        )
        shift_verified = _verified_from_scores(
            shift,
            "oracle",
            {g["task_id"]: [float(x) for x in g["trusted_rewards"]] for g in shift},
            VerifiedGeneration,
        )
        _set_optimizer_lr(
            trainer, float(lock["stale_inducing_policy_shift"]["learning_rate"])
        )
        shift_metrics = trainer.train_step(shift_verified)
        torch.cuda.synchronize()
        shift_drift, shift_drift_rows = post_update_drift_evidence(
            trainer.model, shift_verified
        )
        save_jsonl(seed_root, "stale_inducing_shift_token_drift.jsonl", shift_drift_rows)
        current_state = trainer.snapshot_training_state()

        current_fit = await generate_groups(
            task_sets["verifier_fit"], seed_offset=30_000, expose_labels=True
        )
        current_holdout = await generate_groups(
            task_sets["verifier_holdout"], seed_offset=40_000, expose_labels=True
        )
        calibration = await generate_groups(
            task_sets["calibration"], seed_offset=50_000, expose_labels=True
        )
        effect = await generate_groups(
            task_sets["effect"], seed_offset=60_000, expose_labels=True
        )
        evaluation_bank = await generate_groups(
            task_sets["evaluation"], seed_offset=70_000, expose_labels=False
        )
        baseline_greedy = []
        for i, task in enumerate(task_sets["evaluation"]):
            baseline_greedy.append(
                (
                    await backend.generate(
                        task["task_id"],
                        task["prompt"],
                        n=1,
                        temperature=0.0,
                        seed=random_seed + 80_000 + i,
                    )
                )[0]
            )
        for name, groups in (
            ("fresh_verifier_fit_rollouts.jsonl", current_fit),
            ("current_verifier_holdout_rollouts.jsonl", current_holdout),
            ("lr_calibration_rollouts.jsonl", calibration),
            ("effect_rollouts.jsonl", effect),
        ):
            save_jsonl(
                seed_root,
                name,
                [
                    {
                        "task_id": g["task_id"],
                        "trusted_rewards": g["trusted_rewards"],
                        "generations": [asdict(x) for x in g["generations"]],
                    }
                    for g in groups
                ],
            )
        save_jsonl(
            seed_root,
            "evaluation_candidates_unscored.jsonl",
            [
                {
                    "task_id": g["task_id"],
                    "generations": [asdict(x) for x in g["generations"]],
                }
                for g in evaluation_bank
            ],
        )
        save_json(
            seed_root,
            "baseline_greedy_unscored.json",
            [asdict(g) for g in baseline_greedy],
        )

        fresh_fit_status = _fit_eligible(current_fit, head_cfg)
        if not fresh_fit_status["eligible"]:
            record = {
                "seed": seed,
                "eligible_primary_seed": False,
                "skip_reason": "fresh_verifier_fit_underpowered",
                "stale_fit": stale_fit_status,
                "fresh_fit": fresh_fit_status,
                "shift_train_metrics": shift_metrics,
                "shift_drift": shift_drift,
            }
            save_json(seed_root, "seed_summary.json", record)
            all_seed_results.append(record)
            del trainer, backend, stale_head
            gc.collect()
            torch.cuda.empty_cache()
            continue

        current_fit_features = _extract_group_features(trainer.model, current_fit)
        current_holdout_features = _extract_group_features(trainer.model, current_holdout)
        current_calibration_features = _extract_group_features(trainer.model, calibration)
        current_effect_features = _extract_group_features(trainer.model, effect)
        fresh_head, fresh_training = train_verifier_head(
            _stack_features(current_fit, current_fit_features),
            _flatten_labels(current_fit),
            config=head_cfg,
            seed=random_seed + int(head_cfg["initialization_seed_offset"]) + 1,
        )
        torch.save(fresh_head.state_dict(), seed_root / "fresh_verifier_head.pt")
        save_json(seed_root, "fresh_verifier_training.json", fresh_training)

        fresh_scores = {
            "holdout": score_head(fresh_head, current_holdout_features),
            "calibration": score_head(fresh_head, current_calibration_features),
            "effect": score_head(fresh_head, current_effect_features),
        }

        trainer.restore_training_state(pre_shift_state)
        stale_holdout_features = _extract_group_features(trainer.model, current_holdout)
        stale_calibration_features = _extract_group_features(trainer.model, calibration)
        stale_effect_features = _extract_group_features(trainer.model, effect)
        stale_scores = {
            "holdout": score_head(stale_head, stale_holdout_features),
            "calibration": score_head(stale_head, stale_calibration_features),
            "effect": score_head(stale_head, stale_effect_features),
        }
        policy_shift, policy_shift_rows = _policy_shift_on_current_candidates(
            trainer.model, effect
        )
        save_jsonl(seed_root, "pre_update_policy_shift_tokens.jsonl", policy_shift_rows)
        trainer.restore_training_state(current_state)

        def score_bundle(groups, section: str):
            oracle = {
                g["task_id"]: [float(x) for x in g["trusted_rewards"]]
                for g in groups
            }
            fresh = fresh_scores[section]
            stale = stale_scores[section]
            shuffled = deterministic_prompt_shuffle(
                fresh,
                seed=random_seed
                + int(lock["learned_verifier"]["shuffled_seed_offset"])
                + {"holdout": 1, "calibration": 2, "effect": 3}[section],
            )
            return {
                "oracle": oracle,
                "fresh": fresh,
                "stale": stale,
                "shuffled": shuffled,
            }

        holdout_scores = score_bundle(current_holdout, "holdout")
        calibration_scores = score_bundle(calibration, "calibration")
        effect_scores = score_bundle(effect, "effect")
        geometry_kwargs = {
            "advantage_eps": float(trainer.config.advantage_eps),
            "clip_advantage": float(trainer.config.clip_advantage),
        }
        holdout_geometry = _geometry_from_scores(
            current_holdout, holdout_scores, **geometry_kwargs
        )
        effect_geometry = _geometry_from_scores(
            effect, effect_scores, **geometry_kwargs
        )
        save_json(seed_root, "verifier_holdout_metrics.json", holdout_geometry)
        save_json(seed_root, "pre_update_verifier_geometry.json", effect_geometry)
        save_json(seed_root, "pre_update_policy_shift.json", policy_shift)
        save_jsonl(
            seed_root,
            "verifier_holdout_scores.jsonl",
            _score_rows(current_holdout, holdout_scores),
        )
        save_jsonl(
            seed_root,
            "effect_verifier_scores.jsonl",
            _score_rows(effect, effect_scores),
        )
        save_jsonl(
            seed_root,
            "calibration_verifier_scores.jsonl",
            _score_rows(calibration, calibration_scores),
        )

        calibration_verified = {
            arm: _verified_from_scores(
                calibration, arm, calibration_scores[arm], VerifiedGeneration
            )
            for arm in ARMS
        }
        effect_verified = {
            arm: _verified_from_scores(effect, arm, effect_scores[arm], VerifiedGeneration)
            for arm in ARMS
        }

        calibration_grid = {arm: [] for arm in ARMS}
        for arm in ARMS:
            for lr in lr_grid:
                trainer.restore_training_state(current_state)
                _set_optimizer_lr(trainer, lr)
                tick = time.perf_counter()
                metrics = trainer.train_step(calibration_verified[arm])
                torch.cuda.synchronize()
                drift, drift_rows = post_update_drift_evidence(
                    trainer.model, calibration_verified[arm]
                )
                drift_file = f"calibration_token_drift/{arm}-lr-{lr:.0e}.jsonl"
                save_jsonl(seed_root, drift_file, drift_rows)
                calibration_grid[arm].append(
                    {
                        "lr": lr,
                        "wall_s": time.perf_counter() - tick,
                        "token_drift_file": drift_file,
                        **metrics,
                        **drift,
                    }
                )
        try:
            selection = select_matched_drift_lrs_multi(
                calibration_grid, target_fraction=0.8
            )
        except ValueError as exc:
            save_json(
                seed_root,
                "drift_calibration.json",
                {"grid": calibration_grid, "selection_error": str(exc)},
            )
            record = {
                "seed": seed,
                "eligible_primary_seed": False,
                "skip_reason": "matched_drift_calibration_failed",
                "stale_fit": stale_fit_status,
                "fresh_fit": fresh_fit_status,
                "shift_train_metrics": shift_metrics,
                "shift_drift": shift_drift,
                "pre_update_policy_shift": policy_shift,
                "pre_update_geometry": effect_geometry,
                "calibration_error": str(exc),
            }
            save_json(seed_root, "seed_summary.json", record)
            all_seed_results.append(record)
            del trainer, backend, stale_head, fresh_head
            gc.collect()
            torch.cuda.empty_cache()
            continue
        save_json(
            seed_root,
            "drift_calibration.json",
            {"grid": calibration_grid, **selection},
        )

        arm_records = {}
        eval_post_means = {}
        terminal_greedy = {}
        for arm_index, arm in enumerate(ARMS):
            trainer.restore_training_state(current_state)
            lr = float(selection["selected"][arm]["lr"])
            _set_optimizer_lr(trainer, lr)
            metrics = trainer.train_step(effect_verified[arm])
            torch.cuda.synchronize()
            drift, drift_rows = post_update_drift_evidence(
                trainer.model, effect_verified[arm]
            )
            drift_file = f"{arm}_effect_token_drift.jsonl"
            save_jsonl(seed_root, drift_file, drift_rows)
            eval_file, means = _save_eval_logprobs(
                seed_root, arm, trainer.model, evaluation_bank
            )
            eval_post_means[arm] = means
            greedy = []
            for i, task in enumerate(task_sets["evaluation"]):
                greedy.append(
                    (
                        await backend.generate(
                            task["task_id"],
                            task["prompt"],
                            n=1,
                            temperature=0.0,
                            seed=random_seed + 90_000 + arm_index * 1000 + i,
                        )
                    )[0]
                )
            terminal_greedy[arm] = greedy
            save_json(
                seed_root,
                f"{arm}_terminal_greedy_unscored.json",
                [asdict(g) for g in greedy],
            )
            arm_records[arm] = {
                "lr": lr,
                "train_metrics": metrics,
                "effect_drift": drift,
                "effect_token_drift_file": drift_file,
                "evaluation_candidate_logprob_file": eval_file,
            }

        # Trusted terminal labels are first read here, after all arm updates and
        # all candidate/greedy output collection have been fixed.
        answer_by_task = {x["task_id"]: x["answer"] for x in task_sets["evaluation"]}
        bank_by_task = {g["task_id"]: g for g in evaluation_bank}
        evaluation_rows = []
        candidate_labels = []
        for task in task_sets["evaluation"]:
            task_id = task["task_id"]
            group = bank_by_task[task_id]
            y = [
                response_reward(g.response, answer_by_task[task_id])
                for g in group["generations"]
            ]
            for index, reward in enumerate(y):
                candidate_labels.append(
                    {
                        "task_id": task_id,
                        "candidate_index": index,
                        "candidate_id": f"{task_id}:candidate-{index}",
                        "trusted_reward": float(reward),
                    }
                )
            baseline = [
                sequence_logprob(g.metadata["response_token_logprobs"])
                for g in group["generations"]
            ]
            for arm in ARMS:
                metric = candidate_preference_shift(
                    baseline, eval_post_means[arm][task_id], y
                )
                evaluation_rows.append(
                    {
                        "task_id": task_id,
                        "arm": arm,
                        "trusted_rewards": y,
                        **metric,
                    }
                )
        save_jsonl(seed_root, "evaluation_candidate_trusted_labels.jsonl", candidate_labels)
        save_jsonl(seed_root, "evaluation_preference.jsonl", evaluation_rows)

        informative = {
            arm: [
                float(row["preference_shift"])
                for row in evaluation_rows
                if row["arm"] == arm and row["informative"]
            ]
            for arm in ARMS
        }
        informative_prompts = min(len(informative[arm]) for arm in ARMS)
        mean_shift = {
            arm: float(np.mean(informative[arm]))
            if informative[arm]
            else float("nan")
            for arm in ARMS
        }
        effect_k3 = {
            arm: float(arm_records[arm]["effect_drift"]["post_update_k3"])
            for arm in ARMS
        }
        positive_k3 = [x for x in effect_k3.values() if x > 0 and np.isfinite(x)]
        k3_ratio = (
            max(positive_k3) / min(positive_k3)
            if len(positive_k3) == len(ARMS)
            else float("inf")
        )
        covariances = [float(effect_geometry[arm]["cov_y_v"]) for arm in ARMS]
        shifts = [mean_shift[arm] for arm in ARMS]
        k3_values = [effect_k3[arm] for arm in ARMS]
        geometry_rho = spearman(covariances, shifts)
        k3_rho = spearman(k3_values, shifts)

        baseline_correct = [
            response_reward(g.response, answer_by_task[g.prompt_id])
            for g in baseline_greedy
        ]
        terminal_accuracy = {
            arm: float(
                np.mean(
                    [
                        response_reward(g.response, answer_by_task[g.prompt_id])
                        for g in terminal_greedy[arm]
                    ]
                )
            )
            for arm in ARMS
        }
        policy_shift_ok = (
            np.isfinite(policy_shift["sampled_current_vs_pre_shift_k3"])
            and policy_shift["sampled_current_vs_pre_shift_k3"] > 0
        )
        eligible = (
            informative_prompts
            >= int(lock["terminal_evaluation"]["minimum_informative_prompts"])
            and k3_ratio <= 1.5
            and policy_shift_ok
        )
        record = {
            "seed": seed,
            "stale_fit": stale_fit_status,
            "fresh_fit": fresh_fit_status,
            "shift_train_metrics": shift_metrics,
            "shift_drift": shift_drift,
            "pre_update_policy_shift": policy_shift,
            "pre_update_geometry": effect_geometry,
            "selected_lrs": {
                arm: float(selection["selected"][arm]["lr"]) for arm in ARMS
            },
            "arms": arm_records,
            "effect_k3_ratio": k3_ratio,
            "informative_evaluation_prompts": informative_prompts,
            "mean_preference_shift": mean_shift,
            "geometry_spearman": geometry_rho,
            "effect_k3_spearman": k3_rho,
            "eligible_primary_seed": eligible,
            "greedy": {
                "baseline_accuracy": float(np.mean(baseline_correct)),
                **{f"{arm}_accuracy": value for arm, value in terminal_accuracy.items()},
            },
            "evaluation_access_before_arm_outputs_fixed": 0,
        }
        save_json(seed_root, "seed_summary.json", record)
        all_seed_results.append(record)

        del (
            trainer,
            backend,
            stale_head,
            fresh_head,
            current_fit_features,
            current_holdout_features,
            current_calibration_features,
            current_effect_features,
            stale_holdout_features,
            stale_calibration_features,
            stale_effect_features,
        )
        gc.collect()
        torch.cuda.empty_cache()

    eligible = [row for row in all_seed_results if row.get("eligible_primary_seed")]
    sufficient = len(eligible) >= 2
    mean_geometry_rho = (
        float(np.nanmean([row["geometry_spearman"] for row in eligible]))
        if sufficient
        else None
    )
    mean_k3_rho = (
        float(np.nanmean([row["effect_k3_spearman"] for row in eligible]))
        if sufficient
        else None
    )
    prediction = (
        bool(
            mean_geometry_rho is not None
            and mean_k3_rho is not None
            and np.isfinite(mean_geometry_rho)
            and np.isfinite(mean_k3_rho)
            and mean_geometry_rho > 0
            and mean_geometry_rho > mean_k3_rho
        )
        if sufficient
        else None
    )
    conclusion = {
        "eligible_seeds": len(eligible),
        "required_eligible_seeds": 2,
        "primary_evidence_sufficient": sufficient,
        "mean_geometry_spearman": mean_geometry_rho,
        "mean_effect_k3_spearman": mean_k3_rho,
        "primary_direction_passed": prediction,
        "seed_results": all_seed_results,
        "claim_scope": lock["scope"],
    }
    save_json(output, "summary.json", conclusion)
    return conclusion


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/qwen_learned_verifier_bridge_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/qwen_learned_verifier_bridge_v1",
    )
    args = parser.parse_args()
    lock = json.loads(args.protocol.read_text())
    if not lock["status"].startswith(
        "prospective learned-verifier neural bridge locked before GPU outcomes"
    ):
        raise ValueError("locked learned-verifier protocol required")
    if args.output.exists():
        raise FileExistsError(args.output)

    systems = ROOT.parent / "rvl-pinned-systems"
    if not systems.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/mitukx/Recursive-Verification-Lag.git",
                str(systems),
            ],
            check=True,
        )
    subprocess.run(
        ["git", "checkout", "--detach", lock["systems_source_sha"]],
        cwd=systems,
        check=True,
    )
    sys.path.insert(0, str(systems))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        asyncio.run(run(lock, systems, args.output))
        status = {"status": "completed"}
    except BaseException:
        args.output.mkdir(parents=True, exist_ok=True)
        status = {"status": "failed", "traceback": traceback.format_exc()}
        save_json(args.output, "failure.json", status)
        raise
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        files = {
            str(path.relative_to(args.output)): sha256(path)
            for path in args.output.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        save_json(
            args.output,
            "manifest.json",
            {
                **status,
                "protocol_sha256": sha256(args.protocol),
                "runner_sha256": sha256(Path(__file__)),
                "files": files,
            },
        )


if __name__ == "__main__":
    main()
