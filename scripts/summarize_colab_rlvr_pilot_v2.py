"""Verify corrected pilot artifacts and report negative, null or positive outcomes.

Offline artifact consistency is not remote GPU attestation. Evaluation never
selects parameters, conditions or updates in this protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import fmean

import numpy as np

from scripts.summarize_colab_rlvr_pilot import descriptive_interval, load, predictions
from scripts.verify_colab_evidence_bundle import verify_files
from src.rvl_systems.rlvr_benchmark import response_reward

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs/colab_rlvr_pilot_v2.json"


def near(actual, expected, name, tol=1e-12):
    if not math.isfinite(float(actual)) or not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=tol):
        raise ValueError(f"inconsistent evidence: {name}")


def summarize(raw):
    lock = load(LOCK)
    output = {"schema_version": 2, "status": "incomplete", "frontier_capability_claim": False,
              "scope": lock["scope"], "verification": "artifact consistency; no remote GPU attestation", "runs": []}
    if not (raw / "manifest.json").is_file():
        output["missing"] = ["manifest.json"]
        return output
    manifest = load(raw / "manifest.json")
    verify_files(raw, manifest["files"], "manifest.json")
    source = (raw / "source_protocol.json").read_bytes()
    if (source != LOCK.read_bytes() or load(raw / "protocol.json") != lock or
            hashlib.sha256(source).hexdigest() != manifest["protocol_sha256"]):
        raise ValueError("locked source protocol mismatch")
    if hashlib.sha256((raw / "source_runner.py").read_bytes()).hexdigest() != manifest["runner_sha256"]:
        raise ValueError("retained source runner checksum mismatch")
    output["protocol_sha256"] = manifest["protocol_sha256"]
    output["runner_sha256"] = manifest["runner_sha256"]
    output["retained_failure"] = load(raw / "failure.json") if (raw / "failure.json").exists() else None
    if output["retained_failure"] is not None or not (raw / "completion.json").exists():
        output["missing"] = ["successful completion"]
        return output
    completion = load(raw / "completion.json")
    pairs = [(seed, arm) for seed in lock["seeds"] for arm in lock["arms"]]
    if (completion["status"] != "completed" or completion["conditions"] != len(pairs) or
            completion["optimizer_steps"] != len(pairs)*lock["steps"]):
        raise ValueError("completion counts mismatch")
    environment = load(raw / "environment.json")
    if (environment["systems_source_sha"] != lock["systems_source_sha"] or not environment["cuda"] or
            environment["dependencies"] != lock["dependencies"] or environment["paid_compute_started"] is not False):
        raise ValueError("locked CUDA/dependency/source provenance required")
    tasks = load(raw / "task_manifest.json")
    if (tasks["selection"] != lock["split_selection"] or tasks["model_revision"] != lock["model_revision"] or
            tasks["dataset_revision"] != lock["dataset_revision"]):
        raise ValueError("task selection/revision mismatch")
    if ([row[0] for row in tasks["train"]] != [f"train-{i}" for i in lock["train_indices"]] or
            [row[0] for row in tasks["evaluation"]] != [f"test-{i}" for i in lock["evaluation_indices"]]):
        raise ValueError("exact preselected train/evaluation identities required")
    expected = {row[0]: row[2] for row in tasks["evaluation"]}
    train = {row[0]: row for row in tasks["train"]}
    baseline_rows = load(raw / "baseline.json")
    baseline = predictions(baseline_rows, expected)
    repeated_rows = load(raw / "zero_update_repeat.json")
    repeated = predictions(repeated_rows, expected)
    zero = load(raw / "zero_update_control.json")
    if (zero["identical"] is not True or zero["tasks"] != len(expected) or baseline != repeated or
            {r["task_id"]: r["response"] for r in baseline_rows} != {r["task_id"]: r["response"] for r in repeated_rows}):
        raise ValueError("zero-update reproducibility mismatch")
    summaries = load(raw / "summary.json")
    if len(summaries) != len(pairs) or [(r["seed"], r["arm"]) for r in summaries] != pairs:
        raise ValueError("all ordered seed/arm conditions required")
    initial_hash = load(raw / "initial_model.json")["parameter_sha256"]
    if not isinstance(initial_hash, str) or len(initial_hash) != 64:
        raise ValueError("whole-parameter hash required")
    terminal_by_pair = {}
    fresh_maxima, stage_totals, token_total, nonzero_total = [], {}, 0, 0
    expected_events = []
    for summary in summaries:
        seed, arm = summary["seed"], summary["arm"]
        prefix = f"{seed}_{arm}"
        history = load(raw / f"{prefix}_history.json")
        rollouts = [json.loads(line) for line in (raw / f"{prefix}_rollouts.jsonl").read_text().splitlines()]
        fresh = [json.loads(line) for line in (raw / f"{prefix}_fresh_scores.jsonl").read_text().splitlines()]
        if len(history) != lock["steps"] or len(fresh) != lock["steps"] or len(rollouts) != lock["steps"]*lock["samples_per_prompt"]:
            raise ValueError("complete training actions, scores and updates required")
        permutation = list(tasks["train"]); random.Random(seed).shuffle(permutation)
        for step, (h, scored) in enumerate(zip(history, fresh)):
            if (h["step"] != step or h["seed"] != seed or h["arm"] != arm or scored["step"] != step or
                    len(scored["scores"]) != lock["samples_per_prompt"]):
                raise ValueError("update/score attribution mismatch")
            group = rollouts[step*lock["samples_per_prompt"]:(step+1)*lock["samples_per_prompt"]]
            task, prompt, answer = permutation[step % len(permutation)]
            true = []
            for sample, scores in zip(group, scored["scores"]):
                g, meta = sample["generation"], sample["generation"]["metadata"]
                if (g["prompt_id"] != task or g["prompt"] != prompt or sample["metadata"]["arm"] != arm or
                        meta["sampling_temperature"] != lock["train_temperature"] or
                        meta["logprob_distribution"] != "neutral_temperature_scaled_policy"):
                    raise ValueError("training action policy/task mismatch or evaluation leakage")
                old, current = meta["response_token_logprobs"], scores["learner_token_logprobs"]
                count = len(meta["response_token_ids"])
                if not 0 < count <= lock["max_new_tokens"] or count != len(old) or count != len(current) or g["token_count"] != count:
                    raise ValueError("complete token-exact fresh scores required")
                if any(not math.isfinite(float(v)) for v in old+current):
                    raise ValueError("nonfinite fresh score")
                near(g["logprob"], sum(old), "behavior sum")
                gap = np.asarray(current, dtype=np.float32)-np.asarray(old, dtype=np.float32)
                maximum = float(np.max(np.abs(gap)))
                ratios = np.exp(gap)
                clipping = float(np.mean((ratios < .8) | (ratios > 1.2)))
                near(scores["max_abs_log_ratio"], maximum, "fresh maximum")
                near(scores["clip_fraction"], clipping, "fresh clipping", tol=1e-7)
                if maximum > lock["fresh_policy_gate"]["max_abs_log_ratio"] or clipping > lock["fresh_policy_gate"]["max_clip_fraction"]:
                    raise ValueError("optimizer followed a failed fresh-policy gate")
                fresh_maxima.append(maximum)
                reward = response_reward(g["response"], answer)
                if sample["metadata"]["true_reward_evaluation_only"] != reward:
                    raise ValueError("retained true reward mismatch")
                true.append(reward)
            prescribed = list(true)
            if arm == "within_prompt_shuffled_reward":
                random.Random(seed+900000+step).shuffle(prescribed)
            if [s["reward"] for s in group] != prescribed:
                raise ValueError("trusted/shuffled training reward mismatch")
            near(h["true_reward_mean"], fmean(true), "true reward mean")
            if (h["nonconstant_reward_group"] != (len(set(true)) > 1) or
                    h["response_tokens"] != sum(s["generation"]["token_count"] for s in group)):
                raise ValueError("update reward diversity/token counts mismatch")
            near(h["max_abs_log_ratio"], max(s["max_abs_log_ratio"] for s in scored["scores"]), "learner/probe parity", tol=2e-6)
            near(h["clip_fraction"], fmean(s["clip_fraction"] for s in scored["scores"]), "learner clipping", tol=1e-7)
            for key in ("loss", "grad_norm", "generation_wall_s", "fresh_score_wall_s", "reward_wall_s", "learn_wall_s"):
                if not math.isfinite(float(h[key])):
                    raise ValueError("finite update/stage diagnostics required")
            for key in ("generation_wall_s", "fresh_score_wall_s", "reward_wall_s", "learn_wall_s"):
                if h[key] < 0:
                    raise ValueError("nonnegative stage wall times required")
                stage_totals[key] = stage_totals.get(key, 0)+h[key]
            token_total += h["response_tokens"]; nonzero_total += h["grad_norm"] > 0
            expected_events += [{"seed": seed, "arm": arm, "step": step, "event": event,
                                "optimization_steps_completed": step+int(event == "optimizer_completed")}
                               for event in ("fresh_gate_passed_before_optimizer", "optimizer_completed")]
        terminal = predictions(load(raw / f"{prefix}_terminal.json"), expected)
        before, after = fmean(baseline.values()), fmean(terminal.values())
        for key, actual in (("baseline_accuracy", before), ("terminal_accuracy", after), ("accuracy_delta", after-before)):
            near(summary[key], actual, key)
        if (summary["initial_parameter_sha256"] != initial_hash or len(summary["terminal_parameter_sha256"]) != 64 or
                summary["nonconstant_groups"] != sum(h["nonconstant_reward_group"] for h in history)):
            raise ValueError("parameter start/reward group provenance mismatch")
        for key in ("parameter_hash_wall_s", "terminal_evaluation_wall_s", "condition_wall_s", "peak_allocated_gpu_bytes"):
            if not math.isfinite(float(summary[key])) or summary[key] < 0:
                raise ValueError("invalid condition resource measurement")
        for key in ("parameter_hash_wall_s", "terminal_evaluation_wall_s"):
            stage_totals[key] = stage_totals.get(key, 0)+summary[key]
        output["runs"].append({**summary, "nonzero_gradient_updates": sum(h["grad_norm"] > 0 for h in history),
                               "whole_parameter_hash_changed": initial_hash != summary["terminal_parameter_sha256"]})
        terminal_by_pair[seed, arm] = terminal
    events = [json.loads(line) for line in (raw / "events.jsonl").read_text().splitlines()]
    if events != expected_events:
        raise ValueError("every optimizer must follow its own fresh gate")
    ids = sorted(expected)
    outcomes = {arm: descriptive_interval([fmean(terminal_by_pair[seed, arm][task]-baseline[task]
                         for seed in lock["seeds"]) for task in ids]) for arm in lock["arms"]}
    contrast = descriptive_interval([fmean(terminal_by_pair[seed, lock["arms"][0]][task]-
                                          terminal_by_pair[seed, lock["arms"][1]][task] for seed in lock["seeds"])
                                    for task in ids])
    output.update({"status": "completed_corrected_small_gpu_pilot", "environment": environment,
                   "baseline_accuracy": fmean(baseline.values()), "evaluation_tasks": len(expected),
                   "zero_update_reproducible": True, "arm_outcomes": outcomes, "trusted_minus_shuffled": contrast,
                   "training_rollouts": len(pairs)*lock["steps"]*lock["samples_per_prompt"],
                   "training_response_tokens": token_total, "optimizer_steps": completion["optimizer_steps"],
                   "nonzero_gradient_updates": nonzero_total, "fresh_policy_max_abs_log_ratio": max(fresh_maxima),
                   "all_pre_update_gates_passed": True, "stage_totals": stage_totals,
                   "stage_scope": "captured wall times; excludes loading/setup, no kernel throughput or speedup claim"})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.evidence.resolve()):
        parser.error("derived scorecard must remain outside frozen raw evidence")
    result = summarize(args.evidence)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("status", "frontier_capability_claim")}, indent=2))


if __name__ == "__main__":
    main()
