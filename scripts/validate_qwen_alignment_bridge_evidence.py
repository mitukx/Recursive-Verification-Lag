"""Fail-closed integrity validator for Issue #38 Qwen bridge evidence.

This validates evidence completeness and internal numerical consistency only. It does
not decide whether the scientific prediction passed. A negative, underpowered, or
failed execution can still be valid retained evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ARMS = ("harmful", "benign")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _close(a: float, b: float, *, atol: float = 1e-10, rtol: float = 1e-8) -> bool:
    return bool(np.isclose(float(a), float(b), atol=atol, rtol=rtol))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_token_drift_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    _require(bool(rows), "token drift evidence is empty")
    all_k3: list[float] = []
    all_abs: list[float] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = (str(row["prompt_id"]), int(row["candidate_index"]))
        _require(key not in seen, f"duplicate token-drift identity {key}")
        seen.add(key)
        old = np.asarray(row["old_token_logprobs"], dtype=float)
        new = np.asarray(row["new_token_logprobs"], dtype=float)
        _require(old.ndim == 1 and old.size > 0, f"empty old logprobs for {key}")
        _require(new.shape == old.shape, f"old/new shape mismatch for {key}")
        _require(np.isfinite(old).all() and np.isfinite(new).all(), f"nonfinite logprobs for {key}")
        raw = new - old
        clipped = np.clip(raw, -20.0, 20.0)
        k3 = np.exp(clipped) - 1.0 - clipped
        np.testing.assert_allclose(row["raw_log_ratio"], raw, atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(row["clipped_log_ratio"], clipped, atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(row["token_k3"], k3, atol=1e-12, rtol=1e-10)
        _require(int(row["tokens"]) == old.size, f"token count mismatch for {key}")
        _require(_close(row["mean_k3"], float(k3.mean())), f"mean k3 mismatch for {key}")
        _require(
            _close(row["max_abs_log_ratio"], float(np.abs(clipped).max())),
            f"max log-ratio mismatch for {key}",
        )
        _require(
            _close(row["mean_abs_log_ratio"], float(np.abs(clipped).mean())),
            f"mean log-ratio mismatch for {key}",
        )
        all_k3.extend(k3.tolist())
        all_abs.extend(np.abs(clipped).tolist())
    return {
        "post_update_k3": float(np.mean(all_k3)),
        "post_update_max_abs_log_ratio": float(np.max(all_abs)),
        "post_update_mean_abs_log_ratio": float(np.mean(all_abs)),
        "post_update_tokens": len(all_k3),
    }


def validate_manifest(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    files = manifest.get("files")
    _require(isinstance(files, dict) and files, "manifest files map missing")
    actual = {
        str(path.relative_to(root)): sha256(path)
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    _require(set(files) == set(actual), "manifest file set differs from artifact")
    for rel, digest in files.items():
        _require(actual[rel] == digest, f"SHA-256 mismatch: {rel}")
    return manifest


def _candidate_bank(root: Path) -> dict[str, dict[str, Any]]:
    bank: dict[str, dict[str, Any]] = {}
    for group in load_jsonl(root / "evaluation_candidates_unscored.jsonl"):
        _require("trusted_rewards" not in group, "trusted labels leaked into unscored evaluation bank")
        for index, generation in enumerate(group["generations"]):
            candidate_id = f'{group["task_id"]}:candidate-{index}'
            _require(candidate_id not in bank, f"duplicate candidate {candidate_id}")
            bank[candidate_id] = generation
    return bank


def _validate_eval_logprobs(
    path: Path,
    arm: str,
    bank: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = str(row["candidate_id"])
        _require(cid not in by_id, f"duplicate evaluation candidate {cid} for {arm}")
        _require(row["arm"] == arm, f"wrong arm in {path.name}")
        _require(cid in bank, f"unknown evaluation candidate {cid}")
        generation = bank[cid]
        baseline = np.asarray(row["baseline_token_logprobs"], float)
        behavior = np.asarray(generation["metadata"]["response_token_logprobs"], float)
        post = np.asarray(row["post_update_token_logprobs"], float)
        np.testing.assert_allclose(baseline, behavior, atol=0, rtol=0)
        _require(post.shape == baseline.shape and post.size > 0, f"post logprob shape mismatch {cid}")
        _require(
            _close(row["baseline_sequence_logprob"], float(baseline.sum())),
            f"baseline sequence logprob mismatch {cid}",
        )
        _require(
            _close(row["post_update_sequence_logprob"], float(post.sum())),
            f"post sequence logprob mismatch {cid}",
        )
        _require(
            _close(row["baseline_mean_token_logprob"], float(baseline.mean())),
            f"baseline mean logprob mismatch {cid}",
        )
        _require(
            _close(row["post_update_mean_token_logprob"], float(post.mean())),
            f"post mean logprob mismatch {cid}",
        )
        by_id[cid] = row
    _require(set(by_id) == set(bank), f"{arm} evaluation candidate coverage mismatch")
    return by_id


def validate_seed(seed_root: Path, seed_summary: dict[str, Any]) -> dict[str, Any]:
    _require((seed_root / "training_geometry.json").exists(), "missing training geometry")
    calibration = load_json(seed_root / "drift_calibration.json")
    grid = calibration["grid"]
    _require(set(grid) == set(ARMS), "calibration arms mismatch")
    for arm in ARMS:
        lrs = [float(row["lr"]) for row in grid[arm]]
        selected = float(seed_summary["selected_lrs"][arm])
        _require(selected in lrs, f"selected LR not in declared calibration grid for {arm}")
        _require(
            _close(selected, calibration["selected"][arm]["lr"], atol=0, rtol=0),
            f"selected LR mismatch for {arm}",
        )
        for row in grid[arm]:
            drift_path = seed_root / row["token_drift_file"]
            _require(drift_path.exists(), f"missing calibration token drift: {drift_path}")
            recomputed = validate_token_drift_rows(load_jsonl(drift_path))
            for key, value in recomputed.items():
                _require(_close(row[key], value), f"calibration {arm} {key} mismatch")

    for arm in ARMS:
        arm_record = seed_summary["arms"][arm]
        drift_file = seed_root / arm_record["effect_token_drift_file"]
        _require(drift_file.exists(), f"missing effect token drift for {arm}")
        recomputed = validate_token_drift_rows(load_jsonl(drift_file))
        for key, value in recomputed.items():
            _require(
                _close(arm_record["effect_drift"][key], value),
                f"effect {arm} {key} mismatch",
            )
        _require(
            "grad_norm" in arm_record["train_metrics"]
            and math.isfinite(float(arm_record["train_metrics"]["grad_norm"])),
            f"missing/nonfinite effect gradient norm for {arm}",
        )

    bank = _candidate_bank(seed_root)
    labels = load_jsonl(seed_root / "evaluation_candidate_trusted_labels.jsonl")
    label_by_id: dict[str, float] = {}
    for row in labels:
        cid = str(row["candidate_id"])
        _require(cid not in label_by_id, f"duplicate trusted label {cid}")
        reward = float(row["trusted_reward"])
        _require(reward in (0.0, 1.0), f"nonbinary GSM8K reward {cid}")
        label_by_id[cid] = reward
    _require(set(label_by_id) == set(bank), "trusted-label coverage does not match candidate bank")

    eval_rows = {}
    for arm in ARMS:
        path = seed_root / seed_summary["arms"][arm]["evaluation_candidate_logprob_file"]
        _require(path.exists(), f"missing evaluation logprob file for {arm}")
        eval_rows[arm] = _validate_eval_logprobs(path, arm, bank)
    _require(set(eval_rows["harmful"]) == set(eval_rows["benign"]), "arm candidate identities differ")

    grouped: dict[str, list[str]] = {}
    for cid in bank:
        task_id = cid.rsplit(":candidate-", 1)[0]
        grouped.setdefault(task_id, []).append(cid)

    recorded_preference_rows = load_jsonl(seed_root / "evaluation_preference.jsonl")
    recorded_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in recorded_preference_rows:
        key = (str(row["task_id"]), str(row["arm"]))
        _require(key not in recorded_by_key, f"duplicate evaluation preference row {key}")
        _require(key[1] in ARMS, f"unknown evaluation preference arm {key[1]}")
        recorded_by_key[key] = row
    expected_keys = {(task_id, arm) for task_id in grouped for arm in ARMS}
    _require(set(recorded_by_key) == expected_keys, "evaluation preference row coverage mismatch")

    shifts: dict[str, list[float]] = {arm: [] for arm in ARMS}
    informative = 0
    for task_id, ids in grouped.items():
        ids = sorted(ids, key=lambda cid: int(cid.rsplit(":candidate-", 1)[1]))
        y = np.asarray([label_by_id[cid] for cid in ids], float)
        informative_task = bool(np.any(y > 0.5) and np.any(y <= 0.5))
        informative += int(informative_task)
        positive = y > 0.5
        for arm in ARMS:
            recorded = recorded_by_key[(task_id, arm)]
            np.testing.assert_allclose(
                np.asarray(recorded["trusted_rewards"], float),
                y,
                atol=0,
                rtol=0,
            )
            _require(
                bool(recorded["informative"]) == informative_task,
                f"evaluation informative flag mismatch for {task_id}/{arm}",
            )
            if not informative_task:
                for field in ("baseline_margin", "post_margin", "preference_shift"):
                    _require(
                        recorded.get(field) is None,
                        f"uninformative evaluation row has {field} for {task_id}/{arm}",
                    )
                continue
            base = np.asarray(
                [eval_rows[arm][cid]["baseline_sequence_logprob"] for cid in ids],
                float,
            )
            post = np.asarray(
                [eval_rows[arm][cid]["post_update_sequence_logprob"] for cid in ids],
                float,
            )
            base_margin = float(base[positive].mean() - base[~positive].mean())
            post_margin = float(post[positive].mean() - post[~positive].mean())
            shift = post_margin - base_margin
            _require(
                _close(recorded["baseline_margin"], base_margin),
                f"evaluation baseline margin mismatch for {task_id}/{arm}",
            )
            _require(
                _close(recorded["post_margin"], post_margin),
                f"evaluation post margin mismatch for {task_id}/{arm}",
            )
            _require(
                _close(recorded["preference_shift"], shift),
                f"evaluation preference shift mismatch for {task_id}/{arm}",
            )
            shifts[arm].append(shift)

    _require(
        int(seed_summary["informative_evaluation_prompts"]) == informative,
        "informative evaluation prompt count mismatch",
    )
    for arm in ARMS:
        mean = float(np.mean(shifts[arm])) if shifts[arm] else float("nan")
        recorded = float(seed_summary["mean_preference_shift"][arm])
        _require(
            (math.isnan(mean) and math.isnan(recorded)) or _close(mean, recorded),
            f"mean preference shift mismatch for {arm}",
        )
    if informative:
        delta = float(np.mean(shifts["benign"]) - np.mean(shifts["harmful"]))
        _require(
            _close(delta, seed_summary["benign_minus_harmful_preference_shift"]),
            "benign-minus-harmful preference shift mismatch",
        )
    return {
        "seed": int(seed_summary["seed"]),
        "candidates": len(bank),
        "informative_prompts": informative,
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

    summary = load_json(root / "summary.json")
    seed_rows = summary.get("seed_results", [])
    declared = [int(x) for x in lock["seeds"]]
    _require([int(row["seed"]) for row in seed_rows] == declared, "summary seed order/set mismatch")
    details = []
    for row in seed_rows:
        details.append(validate_seed(root / f'seed-{int(row["seed"])}', row))

    eligible = [row for row in seed_rows if row["eligible_primary_seed"]]
    _require(int(summary["eligible_seeds"]) == len(eligible), "eligible seed count mismatch")
    sufficient = len(eligible) >= int(summary["required_eligible_seeds"])
    _require(bool(summary["primary_evidence_sufficient"]) == sufficient, "primary sufficiency mismatch")
    if not sufficient:
        scientific = "underpowered"
        _require(summary["primary_direction_passed"] is None, "underpowered result cannot pass/fail direction")
    else:
        mean = float(np.mean([row["benign_minus_harmful_preference_shift"] for row in eligible]))
        _require(
            _close(mean, summary["primary_mean_benign_minus_harmful_preference_shift"]),
            "primary mean mismatch",
        )
        passed = bool(mean > 0)
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
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
