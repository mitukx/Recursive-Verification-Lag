"""Single-GPU bridge from controlled verifier geometry to a real neural policy.

The experiment contract is locked in configs/qwen_alignment_bridge_v1.json.
Evaluation prompts may be generated before updates, but their correctness labels
do not enter learning-rate selection or gradients. The runner keeps failures and
partial evidence rather than replacing a negative/underpowered result.
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
import random
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(root: Path, name: str, value: Any) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def save_jsonl(root: Path, name: str, rows: list[dict[str, Any]]) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def alignment_proxy(
    trusted_rewards: list[float],
    *,
    sigma: float,
    rho: float,
    seed: int,
) -> dict[str, Any]:
    """Construct fixed-norm error at declared alignment on one rollout group."""
    y = np.asarray(trusted_rewards, float)
    if (
        y.ndim != 1
        or len(y) < 3
        or not np.isfinite(y).all()
        or not np.isfinite(sigma)
        or sigma <= 0
        or not np.isfinite(rho)
        or abs(rho) > 1
    ):
        raise ValueError("finite rewards, sigma>0, |rho|<=1 and >=3 samples required")
    centered = y - y.mean()
    sd = float(np.sqrt(np.mean(centered * centered)))
    if sd <= 1e-12:
        return {
            "informative": False,
            "proxy": [0.0] * len(y),
            "trusted_standardized": [0.0] * len(y),
            "nuisance": [0.0] * len(y),
            "error_norm": 0.0,
            "trusted_proxy_covariance": 0.0,
        }
    z = centered / sd
    rng = np.random.default_rng(seed)
    u = None
    for _ in range(100):
        candidate = rng.normal(size=len(y))
        candidate -= candidate.mean()
        candidate -= float(np.mean(candidate * z)) * z
        norm = float(np.sqrt(np.mean(candidate * candidate)))
        if norm > 1e-12:
            u = candidate / norm
            break
    if u is None:
        raise ArithmeticError("failed to construct nuisance direction")
    if abs(float(u.mean())) > 1e-10 or abs(float(np.mean(u * z))) > 1e-10:
        raise AssertionError("nuisance orthogonality failed")
    error = sigma * (rho * z + math.sqrt(max(0.0, 1.0 - rho * rho)) * u)
    proxy = z + error
    error_norm = float(np.sqrt(np.mean(error * error)))
    covariance = float(np.mean(z * proxy))
    if not np.isclose(error_norm, sigma, atol=1e-10, rtol=1e-10):
        raise AssertionError("verifier error norm changed")
    return {
        "informative": True,
        "proxy": proxy.tolist(),
        "trusted_standardized": z.tolist(),
        "nuisance": u.tolist(),
        "error_norm": error_norm,
        "trusted_proxy_covariance": covariance,
    }


def select_matched_drift_lrs(
    grid: dict[str, list[dict[str, float]]],
    *,
    target_fraction: float = 0.8,
) -> dict[str, Any]:
    """Choose arm LRs from calibration-only k3 curves with a common target."""
    if not 0 < target_fraction <= 1:
        raise ValueError("target_fraction must be in (0,1]")
    if set(grid) != {"harmful", "benign"}:
        raise ValueError("exact harmful/benign calibration arms required")
    maxima = {}
    valid = {}
    for arm, rows in grid.items():
        usable = [
            row for row in rows
            if row["lr"] > 0 and np.isfinite(row["post_update_k3"]) and row["post_update_k3"] > 0
        ]
        if not usable:
            raise ValueError(f"no positive finite calibration drift for {arm}")
        valid[arm] = usable
        maxima[arm] = max(row["post_update_k3"] for row in usable)
    target = target_fraction * min(maxima.values())
    selected = {}
    for arm, rows in valid.items():
        ranked = sorted(
            rows,
            key=lambda row: (
                abs(math.log(row["post_update_k3"] / target)),
                row["lr"],
            ),
        )
        selected[arm] = ranked[0]
    return {
        "target_k3": target,
        "arm_max_k3": maxima,
        "selected": selected,
    }


def candidate_preference_shift(
    baseline_avg_logps: list[float],
    post_avg_logps: list[float],
    trusted_rewards: list[float],
) -> dict[str, Any]:
    """Correct-vs-incorrect sequence-logprob margin and its update-induced shift."""
    base = np.asarray(baseline_avg_logps, float)
    post = np.asarray(post_avg_logps, float)
    y = np.asarray(trusted_rewards, float)
    if (
        base.ndim != 1
        or post.shape != base.shape
        or y.shape != base.shape
        or not np.isfinite(base).all()
        or not np.isfinite(post).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("aligned finite candidate arrays required")
    positive = y > 0.5
    negative = ~positive
    if not positive.any() or not negative.any():
        return {
            "informative": False,
            "baseline_margin": None,
            "post_margin": None,
            "preference_shift": None,
        }
    baseline_margin = float(base[positive].mean() - base[negative].mean())
    post_margin = float(post[positive].mean() - post[negative].mean())
    return {
        "informative": True,
        "baseline_margin": baseline_margin,
        "post_margin": post_margin,
        "preference_shift": post_margin - baseline_margin,
    }


def effect_drift_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or a < 0 or b < 0:
        raise ValueError("finite nonnegative drifts required")
    small, large = min(a, b), max(a, b)
    if small == 0:
        return 1.0 if large == 0 else float("inf")
    return float(large / small)


def _set_optimizer_lr(trainer: Any, lr: float) -> None:
    for group in trainer.optimizer.param_groups:
        group["lr"] = float(lr)


def _current_token_logps(model: Any, generation: Any) -> np.ndarray:
    import torch

    meta = generation.metadata
    prompt = [int(x) for x in meta["prompt_token_ids"]]
    response = [int(x) for x in meta["response_token_ids"]]
    if not prompt or not response:
        raise ValueError("generation token metadata is empty")
    sequence = torch.tensor(
        [prompt + response],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        logits = model(input_ids=sequence).logits[0]
        start = len(prompt) - 1
        response_logits = logits[start : start + len(response)]
        targets = torch.tensor(response, dtype=torch.long, device=sequence.device)
        current = (
            torch.log_softmax(response_logits.float(), dim=-1)
            .gather(1, targets.unsqueeze(1))
            .squeeze(1)
        )
    if was_training:
        model.train()
    values = current.detach().cpu().numpy().astype(float)
    if not np.isfinite(values).all():
        raise FloatingPointError("non-finite current log probabilities")
    return values


def token_drift_stats(
    old_token_logps: list[float] | np.ndarray,
    new_token_logps: list[float] | np.ndarray,
) -> dict[str, Any]:
    """Return raw token log-ratios plus the locked clipped-k3 drift statistic."""
    old = np.asarray(old_token_logps, float)
    new = np.asarray(new_token_logps, float)
    if (
        old.ndim != 1
        or new.shape != old.shape
        or old.size == 0
        or not np.isfinite(old).all()
        or not np.isfinite(new).all()
    ):
        raise ValueError("aligned nonempty finite token log-probabilities required")
    raw = new - old
    clipped = np.clip(raw, -20.0, 20.0)
    k3 = np.exp(clipped) - 1.0 - clipped
    return {
        "old_token_logprobs": old.tolist(),
        "new_token_logprobs": new.tolist(),
        "raw_log_ratio": raw.tolist(),
        "clipped_log_ratio": clipped.tolist(),
        "token_k3": k3.tolist(),
        "mean_k3": float(np.mean(k3)),
        "max_abs_log_ratio": float(np.max(np.abs(clipped))),
        "mean_abs_log_ratio": float(np.mean(np.abs(clipped))),
        "tokens": int(old.size),
    }


def post_update_drift_evidence(
    model: Any,
    verified: list[Any],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Capture candidate-level old/new token logprobs and aggregate locked drift."""
    rows: list[dict[str, Any]] = []
    all_k3: list[float] = []
    all_abs: list[float] = []
    for candidate_index, item in enumerate(verified):
        generation = item.generation
        current = _current_token_logps(model, generation)
        old = np.asarray(generation.metadata["response_token_logprobs"], float)
        stats = token_drift_stats(old, current)
        all_k3.extend(stats["token_k3"])
        all_abs.extend(abs(x) for x in stats["clipped_log_ratio"])
        rows.append(
            {
                "candidate_index": candidate_index,
                "prompt_id": generation.prompt_id,
                "response": generation.response,
                "response_token_ids": generation.metadata["response_token_ids"],
                "proxy_reward": float(item.reward),
                "trusted_reward": float(
                    item.metadata["trusted_reward_evaluation_only"]
                ),
                "arm": item.metadata["arm"],
                **stats,
            }
        )
    if not rows:
        raise ValueError("post-update drift requires nonempty behavior samples")
    return (
        {
            "post_update_k3": float(np.mean(all_k3)),
            "post_update_max_abs_log_ratio": float(np.max(all_abs)),
            "post_update_mean_abs_log_ratio": float(np.mean(all_abs)),
            "post_update_tokens": len(all_k3),
        },
        rows,
    )


def post_update_drift(model: Any, verified: list[Any]) -> dict[str, float]:
    """Compatibility summary for callers that do not need retained raw evidence."""
    summary, _ = post_update_drift_evidence(model, verified)
    return summary


def _grouped_verified(
    groups: list[dict[str, Any]],
    *,
    arm: str,
    rho: float,
    sigma: float,
    seed: int,
    VerifiedGeneration: Any,
) -> tuple[list[Any], list[dict[str, Any]]]:
    verified, diagnostics = [], []
    for group_index, group in enumerate(groups):
        trusted = group["trusted_rewards"]
        geometry = alignment_proxy(
            trusted,
            sigma=sigma,
            rho=rho,
            seed=seed * 100_003 + group_index * 997 + 31,
        )
        diagnostics.append(
            {
                "task_id": group["task_id"],
                "arm": arm,
                "rho": rho,
                "sigma": sigma,
                "trusted_rewards": trusted,
                **geometry,
            }
        )
        if not geometry["informative"]:
            continue
        for generation, proxy, true_reward in zip(
            group["generations"], geometry["proxy"], trusted
        ):
            verified.append(
                VerifiedGeneration(
                    generation=generation,
                    reward=float(proxy),
                    verifier_latency_s=0.0,
                    verifier_version=0,
                    metadata={
                        "arm": arm,
                        "rho": rho,
                        "sigma": sigma,
                        "trusted_reward_evaluation_only": float(true_reward),
                    },
                )
            )
    return verified, diagnostics


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
    if lock["temperature"] != 1.0:
        raise ValueError("temperature=1 required for behavior-logprob/trainer alignment")

    output.mkdir(parents=True, exist_ok=False)
    (output / "nvidia-smi.txt").write_text(
        subprocess.check_output(["nvidia-smi"], text=True)
    )
    (output / "pip-freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    dependency_names = [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "huggingface_hub",
        "numpy",
    ]
    dependency_versions = {}
    for name in dependency_names:
        try:
            dependency_versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            dependency_versions[name] = None
    save_json(
        output,
        "environment.json",
        {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "dependencies": dependency_versions,
            "systems_source_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=systems, text=True
            ).strip(),
            "research_source_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
    )

    model_path = snapshot_download(lock["model"], revision=lock["model_revision"])
    data = load_dataset(
        lock["dataset"], "main", revision=lock["dataset_revision"]
    )
    partition = lock["task_partition"]

    def tasks(split: str, indices: list[int], prefix: str):
        source = data[split]
        out = []
        for index in indices:
            row = source[int(index)]
            out.append(
                {
                    "task_id": f"{prefix}-{index}",
                    "prompt": build_math_prompt(str(row["question"])),
                    "answer": gsm8k_reference_answer(str(row["answer"])),
                }
            )
        return out

    calibration_tasks = tasks(
        "train", partition["calibration_train_rows"], "calibration"
    )
    effect_tasks = tasks("train", partition["effect_train_rows"], "effect")
    evaluation_tasks = tasks(
        "test", partition["evaluation_test_rows"], "evaluation"
    )
    if (
        {x["task_id"] for x in calibration_tasks}
        & {x["task_id"] for x in effect_tasks}
    ):
        raise AssertionError("calibration/effect task overlap")

    save_json(
        output,
        "task_manifest.json",
        {
            "calibration": calibration_tasks,
            "effect": effect_tasks,
            "evaluation": [
                {"task_id": x["task_id"], "prompt": x["prompt"]}
                for x in evaluation_tasks
            ],
            "evaluation_answers_withheld_from_update_selection": True,
            "model_revision": lock["model_revision"],
            "dataset_revision": lock["dataset_revision"],
        },
    )

    arm_rhos = {
        str(name): float(value)
        for name, value in lock["alignment_intervention"]["arms"].items()
    }
    sigma = float(lock["alignment_intervention"]["sigma"])
    lr_grid = [float(x) for x in lock["drift_calibration"]["learning_rate_grid"]]
    all_seed_results = []

    for seed in lock["seeds"]:
        seed = int(seed)
        seed_root = output / f"seed-{seed}"
        seed_root.mkdir(parents=True, exist_ok=False)
        backend = HFLocalBackend(
            model_path,
            max_new_tokens=int(lock["max_new_tokens"]),
            precision=lock["precision"],
        )
        backend.ensure_loaded()

        async def generate_groups(task_rows, *, seed_offset: int, expose_labels: bool):
            groups = []
            for i, task in enumerate(task_rows):
                generations = await backend.generate(
                    task["task_id"],
                    task["prompt"],
                    n=int(lock["samples_per_prompt"]),
                    temperature=float(lock["temperature"]),
                    seed=seed * 1_000_003 + seed_offset + i * 10_007,
                )
                trusted = (
                    [response_reward(g.response, task["answer"]) for g in generations]
                    if expose_labels
                    else None
                )
                groups.append(
                    {
                        "task_id": task["task_id"],
                        "generations": generations,
                        "trusted_rewards": trusted,
                    }
                )
            return groups

        # Only training-task labels are exposed before LR/update decisions.
        calibration = await generate_groups(
            calibration_tasks, seed_offset=10_000, expose_labels=True
        )
        effect = await generate_groups(
            effect_tasks, seed_offset=20_000, expose_labels=True
        )

        # Evaluation candidates and greedy outputs are generated without reading
        # correctness. Their answer keys are not consumed until both arm updates
        # and all LR choices have been fixed.
        evaluation_bank = await generate_groups(
            evaluation_tasks, seed_offset=30_000, expose_labels=False
        )
        baseline_greedy = []
        for i, task in enumerate(evaluation_tasks):
            generation = (
                await backend.generate(
                    task["task_id"],
                    task["prompt"],
                    n=1,
                    temperature=0.0,
                    seed=seed * 1_000_003 + 40_000 + i,
                )
            )[0]
            baseline_greedy.append(generation)

        save_jsonl(
            seed_root,
            "calibration_rollouts.jsonl",
            [
                {
                    "task_id": group["task_id"],
                    "trusted_rewards": group["trusted_rewards"],
                    "generations": [asdict(g) for g in group["generations"]],
                }
                for group in calibration
            ],
        )
        save_jsonl(
            seed_root,
            "effect_rollouts.jsonl",
            [
                {
                    "task_id": group["task_id"],
                    "trusted_rewards": group["trusted_rewards"],
                    "generations": [asdict(g) for g in group["generations"]],
                }
                for group in effect
            ],
        )
        save_jsonl(
            seed_root,
            "evaluation_candidates_unscored.jsonl",
            [
                {
                    "task_id": group["task_id"],
                    "generations": [asdict(g) for g in group["generations"]],
                }
                for group in evaluation_bank
            ],
        )
        save_json(
            seed_root,
            "baseline_greedy_unscored.json",
            [asdict(g) for g in baseline_greedy],
        )

        trainer = HFCausalLMGRPOTrainer(
            backend.model,
            config=HFTTrainerConfig(learning_rate=min(lr_grid)),
        )
        baseline_state = trainer.snapshot_training_state()
        calibration_verified = {}
        calibration_geometry = {}
        effect_verified = {}
        effect_geometry = {}
        for arm, rho in arm_rhos.items():
            calibration_verified[arm], calibration_geometry[arm] = _grouped_verified(
                calibration,
                arm=arm,
                rho=rho,
                sigma=sigma,
                seed=seed + 101,
                VerifiedGeneration=VerifiedGeneration,
            )
            effect_verified[arm], effect_geometry[arm] = _grouped_verified(
                effect,
                arm=arm,
                rho=rho,
                sigma=sigma,
                seed=seed + 202,
                VerifiedGeneration=VerifiedGeneration,
            )
            if not calibration_verified[arm]:
                raise RuntimeError(f"seed {seed}: no informative calibration groups")
            if not effect_verified[arm]:
                raise RuntimeError(f"seed {seed}: no informative effect groups")

        save_json(
            seed_root,
            "training_geometry.json",
            {
                "calibration": calibration_geometry,
                "effect": effect_geometry,
            },
        )

        calibration_grid = {arm: [] for arm in arm_rhos}
        for arm in ("harmful", "benign"):
            for lr in lr_grid:
                trainer.restore_training_state(baseline_state)
                _set_optimizer_lr(trainer, lr)
                tick = time.perf_counter()
                train_metrics = trainer.train_step(calibration_verified[arm])
                torch.cuda.synchronize()
                drift, drift_rows = post_update_drift_evidence(
                    trainer.model, calibration_verified[arm]
                )
                drift_file = (
                    f"calibration_token_drift/{arm}-lr-{lr:.0e}.jsonl"
                )
                save_jsonl(seed_root, drift_file, drift_rows)
                calibration_grid[arm].append(
                    {
                        "lr": lr,
                        "wall_s": time.perf_counter() - tick,
                        "token_drift_file": drift_file,
                        **train_metrics,
                        **drift,
                    }
                )
        selection = select_matched_drift_lrs(calibration_grid, target_fraction=0.8)
        save_json(
            seed_root,
            "drift_calibration.json",
            {"grid": calibration_grid, **selection},
        )

        # All update decisions are now fixed. Do not mutate selection below from
        # any evaluation outcome.
        arm_records = {}
        evaluation_post_logps = {}
        terminal_greedy = {}
        for arm in ("harmful", "benign"):
            trainer.restore_training_state(baseline_state)
            lr = float(selection["selected"][arm]["lr"])
            _set_optimizer_lr(trainer, lr)
            train_metrics = trainer.train_step(effect_verified[arm])
            effect_drift, effect_drift_rows = post_update_drift_evidence(
                trainer.model, effect_verified[arm]
            )
            effect_drift_file = f"{arm}_effect_token_drift.jsonl"
            save_jsonl(seed_root, effect_drift_file, effect_drift_rows)

            post_by_task = {}
            evaluation_logprob_rows = []
            for group in evaluation_bank:
                task_values = []
                for candidate_index, generation in enumerate(group["generations"]):
                    current = _current_token_logps(trainer.model, generation)
                    baseline = np.asarray(
                        generation.metadata["response_token_logprobs"], float
                    )
                    task_values.append(float(np.mean(current)))
                    evaluation_logprob_rows.append(
                        {
                            "task_id": group["task_id"],
                            "candidate_index": candidate_index,
                            "candidate_id": (
                                f'{group["task_id"]}:candidate-{candidate_index}'
                            ),
                            "arm": arm,
                            "response": generation.response,
                            "response_token_ids": generation.metadata[
                                "response_token_ids"
                            ],
                            "baseline_token_logprobs": baseline.tolist(),
                            "post_update_token_logprobs": current.tolist(),
                            "baseline_sequence_logprob": float(np.sum(baseline)),
                            "post_update_sequence_logprob": float(np.sum(current)),
                            "baseline_mean_token_logprob": float(np.mean(baseline)),
                            "post_update_mean_token_logprob": float(np.mean(current)),
                        }
                    )
                post_by_task[group["task_id"]] = task_values
            evaluation_post_logps[arm] = post_by_task
            evaluation_logprob_file = (
                f"{arm}_evaluation_candidate_token_logprobs.jsonl"
            )
            save_jsonl(seed_root, evaluation_logprob_file, evaluation_logprob_rows)

            greedy = []
            for i, task in enumerate(evaluation_tasks):
                generation = (
                    await backend.generate(
                        task["task_id"],
                        task["prompt"],
                        n=1,
                        temperature=0.0,
                        seed=seed * 1_000_003 + 50_000 + i,
                    )
                )[0]
                greedy.append(generation)
            terminal_greedy[arm] = greedy
            arm_records[arm] = {
                "lr": lr,
                "train_metrics": train_metrics,
                "effect_drift": effect_drift,
                "effect_token_drift_file": effect_drift_file,
                "evaluation_candidate_logprob_file": evaluation_logprob_file,
            }
            save_json(
                seed_root,
                f"{arm}_terminal_greedy_unscored.json",
                [asdict(g) for g in greedy],
            )

        drift_ratio = effect_drift_ratio(
            arm_records["harmful"]["effect_drift"]["post_update_k3"],
            arm_records["benign"]["effect_drift"]["post_update_k3"],
        )

        # Evaluation answer access starts only here, after both LR selections,
        # both effect updates, and all model-output collection.
        evaluation_rows = []
        evaluation_candidate_labels = []
        answer_by_task = {x["task_id"]: x["answer"] for x in evaluation_tasks}
        bank_by_task = {x["task_id"]: x for x in evaluation_bank}
        for task in evaluation_tasks:
            task_id = task["task_id"]
            group = bank_by_task[task_id]
            rewards = [
                response_reward(g.response, answer_by_task[task_id])
                for g in group["generations"]
            ]
            evaluation_candidate_labels.extend(
                {
                    "task_id": task_id,
                    "candidate_index": candidate_index,
                    "candidate_id": f"{task_id}:candidate-{candidate_index}",
                    "trusted_reward": float(reward),
                }
                for candidate_index, reward in enumerate(rewards)
            )
            baseline_avg = [
                float(np.mean(g.metadata["response_token_logprobs"]))
                for g in group["generations"]
            ]
            for arm in ("harmful", "benign"):
                metric = candidate_preference_shift(
                    baseline_avg,
                    evaluation_post_logps[arm][task_id],
                    rewards,
                )
                evaluation_rows.append(
                    {
                        "task_id": task_id,
                        "arm": arm,
                        "trusted_rewards": rewards,
                        **metric,
                    }
                )
        save_jsonl(seed_root, "evaluation_preference.jsonl", evaluation_rows)
        save_jsonl(
            seed_root,
            "evaluation_candidate_trusted_labels.jsonl",
            evaluation_candidate_labels,
        )

        informative = {
            arm: [
                row["preference_shift"]
                for row in evaluation_rows
                if row["arm"] == arm and row["informative"]
            ]
            for arm in ("harmful", "benign")
        }
        informative_tasks = min(len(informative["harmful"]), len(informative["benign"]))
        mean_shift = {
            arm: (
                float(np.mean(informative[arm]))
                if informative[arm]
                else float("nan")
            )
            for arm in ("harmful", "benign")
        }

        baseline_correct = [
            response_reward(g.response, answer_by_task[g.prompt_id])
            for g in baseline_greedy
        ]
        terminal_correct = {
            arm: [
                response_reward(g.response, answer_by_task[g.prompt_id])
                for g in terminal_greedy[arm]
            ]
            for arm in ("harmful", "benign")
        }
        greedy_metrics = {
            "baseline_accuracy": float(np.mean(baseline_correct)),
            **{
                f"{arm}_accuracy": float(np.mean(values))
                for arm, values in terminal_correct.items()
            },
        }
        eligible = informative_tasks >= 3 and drift_ratio <= 1.5
        seed_record = {
            "seed": seed,
            "selected_lrs": {
                arm: float(selection["selected"][arm]["lr"])
                for arm in ("harmful", "benign")
            },
            "target_calibration_k3": selection["target_k3"],
            "arms": arm_records,
            "effect_k3_ratio": drift_ratio,
            "informative_evaluation_prompts": informative_tasks,
            "mean_preference_shift": mean_shift,
            "benign_minus_harmful_preference_shift": (
                mean_shift["benign"] - mean_shift["harmful"]
                if informative_tasks
                else float("nan")
            ),
            "eligible_primary_seed": eligible,
            "greedy": greedy_metrics,
            "evaluation_access_before_lr_selection": 0,
        }
        save_json(seed_root, "seed_summary.json", seed_record)
        all_seed_results.append(seed_record)

        del trainer, backend
        gc.collect()
        torch.cuda.empty_cache()

    eligible_rows = [row for row in all_seed_results if row["eligible_primary_seed"]]
    primary_sufficient = len(eligible_rows) >= 2
    primary_mean = (
        float(
            np.mean(
                [
                    row["benign_minus_harmful_preference_shift"]
                    for row in eligible_rows
                ]
            )
        )
        if primary_sufficient
        else None
    )
    conclusion = {
        "eligible_seeds": len(eligible_rows),
        "required_eligible_seeds": 2,
        "primary_evidence_sufficient": primary_sufficient,
        "primary_mean_benign_minus_harmful_preference_shift": primary_mean,
        "primary_direction_passed": (
            bool(primary_mean > 0) if primary_sufficient else None
        ),
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
        default=ROOT / "configs/qwen_alignment_bridge_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/qwen_alignment_bridge_v1",
    )
    args = parser.parse_args()
    lock = json.loads(args.protocol.read_text())
    if not lock["status"].startswith(
        "pre-execution real-model geometry bridge locked before GPU outcomes"
    ):
        raise ValueError("locked pre-execution bridge protocol required")
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

    args.output.mkdir(parents=True, exist_ok=False)
    # run() owns output creation; remove the sentinel after reserving parent.
    args.output.rmdir()
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
