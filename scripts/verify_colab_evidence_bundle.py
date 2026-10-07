"""Recompute the checked-in T4 evidence report without rerunning a model.

Hashes provide artifact consistency, not remote hardware attestation. Legacy
errors are recomputed from retained arrays; corrected learner diagnostics are
validated as captured measurements, not independently regenerated GPU scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.summarize_colab_rlvr_pilot import load, summarize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/colab_rlvr_pilot_20261008/20261007T162508345719Z"


def verify_files(root, files, excluded):
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.relative_to(root).as_posix() != excluded}
    if actual != set(files):
        raise ValueError("evidence file index mismatch")
    for name, digest in files.items():
        path = root / name
        if (Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())):
            raise ValueError("unsafe evidence path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"evidence checksum mismatch: {name}")


def close(actual, expected, name):
    if not math.isfinite(float(actual)) or not math.isclose(float(actual), float(expected), abs_tol=1e-12):
        raise ValueError(f"inconsistent captured diagnostic: {name}")


def report(evidence=DEFAULT):
    raw = evidence / "raw"
    bundle = load(raw / "bundle-manifest.json")
    verify_files(raw, bundle, "bundle-manifest.json")
    pilot = summarize(raw / "pilot-results")
    pilot.pop("attempt_path")
    diagnostic = raw / "behavior-diagnostic-02"
    manifest = load(diagnostic / "manifest.json")
    verify_files(diagnostic, manifest["files"], "manifest.json")
    for name, digest in manifest["source_files"].items():
        path = evidence / "source_snapshots" / name
        if (Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink()
                or not path.resolve().is_relative_to((evidence / "source_snapshots").resolve())):
            raise ValueError("unsafe source snapshot path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"source snapshot checksum mismatch: {name}")
    config = load(diagnostic / "protocol.json")
    if config != load(evidence / "source_snapshots/configs/hf_behavior_parity_diagnostic_v1.json"):
        raise ValueError("diagnostic protocol differs from source snapshot")
    replay = load(diagnostic / "legacy_replay.json")
    legacy = [json.loads(line) for line in (raw / "pilot-results/17_trusted_reward_rollouts.jsonl").read_text().splitlines()][:4]
    if len(replay) != 4:
        raise ValueError("complete legacy group required")
    for row, original in zip(replay, legacy):
        metadata = original["generation"]["metadata"]
        for key in ("prompt_token_ids", "response_token_ids"):
            if row[key] != metadata[key]:
                raise ValueError("legacy replay token identity mismatch")
        arrays = [row[k] for k in ("legacy_recorded_logprobs", "raw_policy_logprobs", "replayed_repetition_penalty_logprobs")]
        if not arrays[0] or any(len(a) != len(row["response_token_ids"]) for a in arrays):
            raise ValueError("legacy replay array length mismatch")
        for a in arrays:
            if any(not math.isfinite(float(v)) for v in a):
                raise ValueError("nonfinite replay score")
        if any(not math.isclose(a, b, abs_tol=1e-6) for a, b in zip(arrays[0], metadata["response_token_logprobs"])):
            raise ValueError("legacy behavior scores changed")
        close(row["raw_policy_max_abs_log_ratio"], max(abs(a-b) for a,b in zip(arrays[1], arrays[0])), "legacy raw gap")
        close(row["penalty_replay_max_abs_error"], max(abs(a-b) for a,b in zip(arrays[2], arrays[0])), "legacy penalty error")
    groups = load(diagnostic / "corrected_samples.json")
    if [g["temperature"] for g in groups] != config["corrected_temperatures"]:
        raise ValueError("temperature coverage mismatch")
    for group in groups:
        if len(group["samples"]) != config["samples_per_temperature"]:
            raise ValueError("corrected sample coverage mismatch")
        for row in group["samples"]:
            meta = row["generation"]["metadata"]
            if (meta["sampling_temperature"] != group["temperature"] or
                    meta["logprob_distribution"] != "neutral_temperature_scaled_policy" or
                    not 0 < len(meta["response_token_ids"]) <= config["max_new_tokens"] or
                    len(meta["response_token_ids"]) != len(meta["response_token_logprobs"])):
                raise ValueError("corrected action/score contract mismatch")
            for key in ("max_abs_log_ratio", "clip_fraction", "behavior_kl_estimate"):
                if not math.isfinite(float(row[key])):
                    raise ValueError("nonfinite corrected diagnostic")
    result = load(diagnostic / "summary.json")
    close(result["legacy_raw_policy_max_abs_log_ratio"], max(r["raw_policy_max_abs_log_ratio"] for r in replay), "legacy aggregate")
    close(result["legacy_penalty_replay_max_abs_error"], max(r["penalty_replay_max_abs_error"] for r in replay), "penalty aggregate")
    close(result["corrected_max_abs_log_ratio"], max(r["max_abs_log_ratio"] for g in groups for r in g["samples"]), "corrected aggregate")
    passed = result["legacy_penalty_replay_max_abs_error"] <= config["legacy_repetition_replay_tolerance"]
    parity = result["corrected_max_abs_log_ratio"] <= config["same_weights_max_abs_log_ratio_tolerance"]
    if result["penalty_reconstruction_passed"] != passed or result["corrected_parity_passed"] != parity:
        raise ValueError("declared diagnostic gate mismatch")
    rc = load(raw / "diagnostic-exit-02.json")["returncode"]
    if rc != (0 if passed and parity else 1) or (diagnostic / "failure.json").exists() != bool(rc):
        raise ValueError("diagnostic failure/exit disagreement")
    if result["optimization_steps"] != 0 or result["capability_gain_claim"] is not False:
        raise ValueError("diagnostic cannot claim training or capability gains")
    if load(raw / "exit.json")["returncode"] != 0 or load(raw / "diagnostic-exit.json")["returncode"] != 1:
        raise ValueError("pilot/first diagnostic execution status mismatch")
    return {"schema_version": 1, "verification": "artifact consistency; no remote hardware attestation or GPU rerun",
            "pilot": pilot, "diagnostic": result, "diagnostic_process_status": "failed" if rc else "passed",
            "corrected_score_scope": "captured learner diagnostics; aggregates verified without independently recomputing model logits",
            "retained_attempts": ["pilot-results", "behavior-diagnostic", "behavior-diagnostic-02"],
            "raw_indexed_files": len(bundle)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    value = report(args.evidence)
    path = args.evidence / "scorecard.json"
    if args.write:
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    elif value != load(path):
        raise ValueError("checked-in scorecard differs from recomputed evidence")
    print(json.dumps({"artifact_verification": "passed", "pilot_status": value["pilot"]["status"],
                      "diagnostic_process_status": value["diagnostic_process_status"],
                      "corrected_parity_passed": value["diagnostic"]["corrected_parity_passed"]}, indent=2))


if __name__ == "__main__":
    main()
