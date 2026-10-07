"""Verify raw Colab pilot evidence and report paired, descriptive outcomes.

No training or hyperparameter selection happens here. A completed engineering
pilot never becomes a frontier-scale, causal or hiring-readiness claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import fmean

import numpy as np

from src.rvl_systems.rlvr_benchmark import response_reward


LOCK = Path(__file__).resolve().parents[1] / "configs/colab_rlvr_pilot_v1.json"
RUNNER = LOCK.parent.parent / "scripts/run_colab_rlvr_pilot.py"
RUNNER_SHA256 = "6ad6174a5193be320ed16f8bdf1b1a5cfc42b36b0a0060090b1ee5616de679b5"
RESEARCH_SHA = "30b3a438ac7bd77a0cc216ff250873a9b8818925"


def load(path):
    return json.loads(path.read_text())


def verify_manifest(root, lock_path=LOCK):
    manifest = load(root / "manifest.json")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("nonempty evidence manifest required")
    for name, digest in files.items():
        relative = Path(name)
        path = root / relative
        if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())):
            raise ValueError("unsafe evidence path")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"evidence checksum mismatch: {name}")
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and p != root / "manifest.json"}
    if actual != set(files):
        raise ValueError("all raw files must be indexed by the evidence manifest")
    # The runner hashes the source lock, then saves protocol.json with sorted
    # keys. Its separately indexed raw-file hash covers that reformatted copy.
    # Verify both source bytes and semantic identity; do not equate their hashes.
    source_lock = lock_path.read_bytes()
    if (hashlib.sha256(source_lock).hexdigest() != manifest.get("protocol_sha256") or
            load(root / "protocol.json") != json.loads(source_lock)):
        raise ValueError("source protocol checksum/content mismatch")
    if manifest.get("runner_sha256") != RUNNER_SHA256:
        raise ValueError("runner differs from the retained locked implementation")
    return manifest


def predictions(rows, expected):
    if len(rows) != len(expected):
        raise ValueError("prediction cardinality mismatch")
    by_id = {}
    for row in rows:
        key = row["task_id"]
        if key not in expected or key in by_id or row["expected"] != expected[key]:
            raise ValueError("prediction task identity/answer mismatch")
        correct = response_reward(row["response"], expected[key])
        if row["correct"] != correct:
            raise ValueError("stored correctness disagrees with retained prediction")
        by_id[key] = correct
    return by_id


def descriptive_interval(values):
    values = np.asarray(values, dtype=np.float64)
    draws = np.random.default_rng(20261008).integers(0, len(values), size=(10000, len(values)))
    means = values[draws].mean(axis=1)
    return {"mean": float(values.mean()), "task_bootstrap_95_interval":
            [float(x) for x in np.quantile(means, [0.025, 0.975])],
            "uncertainty_scope": "descriptive task bootstrap conditional on retained seeds; not seed uncertainty or a confirmatory test"}


def summarize(root, lock_path=LOCK):
    lock = load(lock_path)
    output = {"schema_version": 1, "status": "incomplete", "frontier_or_hiring_readiness_claim": False,
              "claim_boundary": lock["scope"], "attempt_path": str(root), "runs": []}
    if not (root / "manifest.json").is_file():
        output["missing"] = ["manifest.json"]
        return output
    manifest = verify_manifest(root, lock_path)
    protocol = load(root / "protocol.json")
    if protocol != lock:
        raise ValueError("execution protocol differs from the pre-execution lock")
    output["protocol_sha256"] = manifest["protocol_sha256"]
    output["runner_sha256"] = manifest["runner_sha256"]
    required = {"environment.json", "nvidia-smi.txt", "task_manifest.json", "baseline.json",
                "zero_update_repeat.json", "zero_update_control.json", "summary.json", "completion.json"}
    for seed in lock["seeds"]:
        for arm in lock["arms"]:
            required.update(f"{seed}_{arm}_{suffix}" for suffix in
                            ("rollouts.jsonl", "history.json", "terminal.json"))
    missing = sorted(required - set(manifest["files"]))
    if missing:
        output["missing"] = missing
        output["retained_failure"] = load(root / "failure.json") if (root / "failure.json").is_file() else None
        return output
    completion = load(root / "completion.json")
    if completion.get("status") != "completed" or (root / "failure.json").exists():
        raise ValueError("completed pilot cannot contain an execution failure")
    pairs = {(seed, arm) for seed in lock["seeds"] for arm in lock["arms"]}
    if completion.get("all_seed_arm_runs") != len(pairs):
        raise ValueError("completion seed/arm count mismatch")
    environment = load(root / "environment.json")
    if (environment.get("systems_source_sha") != lock["systems_source_sha"] or
            environment.get("research_source_sha") != RESEARCH_SHA or not environment.get("cuda")):
        raise ValueError("pinned CUDA systems provenance required")
    tasks = load(root / "task_manifest.json")
    if tasks.get("model_revision") != lock["model_revision"] or tasks.get("dataset_revision") != lock["dataset_revision"]:
        raise ValueError("model/dataset revision mismatch")
    if tasks.get("selection") != lock["split_selection"]:
        raise ValueError("locked split selection required")
    if len(tasks["train"]) != lock["train_tasks"] or len(tasks["evaluation"]) != lock["evaluation_tasks"]:
        raise ValueError("locked task counts required")
    expected = {row[0]: row[2] for row in tasks["evaluation"]}
    train_ids = {row[0] for row in tasks["train"]}
    train_answers = {row[0]: row[2] for row in tasks["train"]}
    if len(expected) != lock["evaluation_tasks"] or len(train_ids) != lock["train_tasks"] or train_ids & set(expected):
        raise ValueError("distinct train/evaluation task identities required")
    baseline_rows = load(root / "baseline.json")
    baseline = predictions(baseline_rows, expected)
    repeated_rows = load(root / "zero_update_repeat.json")
    repeat = predictions(repeated_rows, expected)
    zero = load(root / "zero_update_control.json")
    baseline_responses = {row["task_id"]: row["response"] for row in baseline_rows}
    if (baseline != repeat or zero.get("identical") is not True or zero.get("tasks") != len(expected) or
            {row["task_id"]: row["response"] for row in repeated_rows} != baseline_responses):
        raise ValueError("zero-update reproducibility failed")
    summaries = load(root / "summary.json")
    if len(summaries) != len(pairs) or {(row["seed"], row["arm"]) for row in summaries} != pairs:
        raise ValueError("exact declared seed/arm summary required")
    ordered_ids = sorted(expected)
    terminal_by_pair = {}
    for row in summaries:
        seed, arm = row["seed"], row["arm"]
        terminal = predictions(load(root / f"{seed}_{arm}_terminal.json"), expected)
        history = load(root / f"{seed}_{arm}_history.json")
        if len(history) != lock["steps"] or [h["step"] for h in history] != list(range(lock["steps"])):
            raise ValueError("every locked update must be retained")
        if any(h["seed"] != seed or h["arm"] != arm for h in history):
            raise ValueError("update attribution mismatch")
        for h in history:
            for key in ("loss", "grad_norm", "train_wall_s", "parameter_probe_max_abs_change", "max_abs_log_ratio", "clip_fraction"):
                if not math.isfinite(float(h[key])):
                    raise ValueError("finite update diagnostics required")
        rollouts = [json.loads(line) for line in (root / f"{seed}_{arm}_rollouts.jsonl").read_text().splitlines() if line.strip()]
        if len(rollouts) != lock["steps"] * lock["samples_per_prompt"]:
            raise ValueError("every sampled training action must be retained")
        if any(v["generation"]["prompt_id"] not in train_ids or v["metadata"]["arm"] != arm for v in rollouts):
            raise ValueError("training rollout identity mismatch or evaluation leakage")
        for step, h in enumerate(history):
            group = rollouts[step * lock["samples_per_prompt"]:(step + 1) * lock["samples_per_prompt"]]
            if len({v["generation"]["prompt_id"] for v in group}) != lock["prompts_per_step"]:
                raise ValueError("locked within-prompt group required")
            true_rewards = [response_reward(v["generation"]["response"], train_answers[v["generation"]["prompt_id"]]) for v in group]
            supplied = [v["reward"] for v in group]
            prescribed = list(true_rewards)
            if arm == "within_prompt_shuffled_reward":
                random.Random(seed + 900000 + step).shuffle(prescribed)
            if ([v["metadata"]["true_reward_evaluation_only"] for v in group] != true_rewards or
                    supplied != prescribed):
                raise ValueError("rollout rewards disagree with retained actions")
            if (h["nonconstant_reward_group"] != (len(set(true_rewards)) > 1) or
                    not math.isclose(float(h["true_reward_mean"]), fmean(true_rewards), abs_tol=1e-12) or
                    h["response_tokens"] != sum(v["generation"]["token_count"] for v in group)):
                raise ValueError("update diagnostics disagree with retained rollouts")
        deltas = [terminal[key] - baseline[key] for key in ordered_ids]
        before, after, delta = fmean(baseline.values()), fmean(terminal.values()), fmean(deltas)
        for key, actual in (("baseline_accuracy", before), ("terminal_accuracy", after), ("accuracy_delta", delta)):
            if not math.isclose(float(row[key]), actual, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"summary inconsistent with raw predictions: {key}")
        nonconstant = sum(bool(h["nonconstant_reward_group"]) for h in history)
        if row["nonconstant_groups"] != nonconstant:
            raise ValueError("reward diversity summary mismatch")
        if (row["peak_allocated_gpu_bytes"] <= 0 or not math.isfinite(float(row["wall_s"])) or
                not math.isclose(float(row["max_parameter_probe_change"]),
                                 max(float(h["parameter_probe_max_abs_change"]) for h in history), abs_tol=1e-12)):
            raise ValueError("GPU/parameter summary diagnostics invalid")
        terminal_by_pair[seed, arm] = terminal
        output["runs"].append({"seed": seed, "arm": arm, "before_accuracy": before,
                               "after_accuracy": after, "accuracy_delta": delta,
                               "nonconstant_reward_groups": nonconstant,
                               "nonzero_gradient_updates": sum(float(h["grad_norm"]) > 0 for h in history),
                               "maximum_parameter_probe_change": max(float(h["parameter_probe_max_abs_change"]) for h in history),
                               "max_fresh_rollout_abs_log_ratio": max(float(h["max_abs_log_ratio"]) for h in history),
                               "mean_fresh_rollout_clip_fraction": fmean(float(h["clip_fraction"]) for h in history),
                               "train_wall_s": sum(float(h["train_wall_s"]) for h in history),
                               "peak_allocated_gpu_bytes": row["peak_allocated_gpu_bytes"]})
    output["arm_outcomes"] = {}
    for arm in lock["arms"]:
        per_task_delta = [fmean(terminal_by_pair[seed, arm][key] - baseline[key] for seed in lock["seeds"])
                          for key in ordered_ids]
        output["arm_outcomes"][arm] = descriptive_interval(per_task_delta)
    contrasts = [fmean(terminal_by_pair[seed, "trusted_reward"][key] -
                       terminal_by_pair[seed, "within_prompt_shuffled_reward"][key]
                       for seed in lock["seeds"]) for key in ordered_ids]
    output.update({"status": "completed_small_gpu_pilot", "environment": environment,
                   "evaluation_tasks": len(expected), "training_seeds": lock["seeds"],
                   "baseline_accuracy": fmean(baseline.values()), "zero_update_reproducible": True,
                   "trusted_minus_shuffled": descriptive_interval(contrasts)})
    maximum_ratio_gap = max(r["max_fresh_rollout_abs_log_ratio"] for r in output["runs"])
    output["fresh_rollout_numerical_diagnostic"] = {
        "max_abs_log_ratio": maximum_ratio_gap,
        "exceeds_0_01": maximum_ratio_gap > 0.01,
        "assessment": "exploratory diagnostic added after first T4 updates; not a preregistered primary outcome",
        "interpretation": "investigate sampling/learner policy mismatch before attributing capability changes to correct on-policy GRPO" if maximum_ratio_gap > .01 else "no large mismatch observed by this diagnostic"}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.evidence.resolve()):
        parser.error("scorecard must remain outside frozen raw evidence")
    result = summarize(args.evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
