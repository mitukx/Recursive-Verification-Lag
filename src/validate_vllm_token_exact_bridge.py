"""Validate the real serving-to-learner token-exact bridge evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate(
    replay: Path,
    metadata: Path,
    train_report: Path,
    *,
    model_repo: str,
    model_revision: str,
) -> dict:
    meta = json.loads(metadata.read_text())
    report = json.loads(train_report.read_text())
    replay_hash = sha256_file(replay)
    require(meta.get("status") == "remote_token_exact_verified_replay", "unexpected replay status")
    require(meta.get("model") == model_repo, "served model identity mismatch")
    require(meta.get("model_revision") == model_revision, "served model revision mismatch")
    require(meta.get("replay_sha256") == replay_hash, "replay metadata hash mismatch")
    require(report.get("replay_sha256") == replay_hash, "learner consumed different replay")
    require(report.get("status") == "completed", "learner report not completed")
    parity = dict(report.get("preupdate_parity") or {})
    require(parity.get("passed") is True, "serving/learner parity gate did not pass")
    max_ratio = float(parity.get("max_abs_log_ratio", float("nan")))
    threshold = float(parity.get("threshold", float("nan")))
    require(math.isfinite(max_ratio) and math.isfinite(threshold), "non-finite parity evidence")
    require(max_ratio <= threshold, "pre-update serving/learner log-ratio exceeds threshold")
    require(int(meta.get("samples", 0)) == int(report.get("samples", -1)), "sample count mismatch")
    require(int(meta.get("groups", 0)) == int(report.get("groups", -1)), "group count mismatch")
    require(int(meta.get("generated_tokens", 0)) > 0, "no generated tokens")
    require(int(report.get("sampled_parameter_values", 0)) > 0, "parameter probe missing")
    return {
        "valid": True,
        "model_repo": model_repo,
        "model_revision": model_revision,
        "replay_sha256": replay_hash,
        "samples": int(report["samples"]),
        "groups": int(report["groups"]),
        "informative_reward_groups": int(report["informative_reward_groups"]),
        "preupdate_max_abs_log_ratio": max_ratio,
        "preupdate_threshold": threshold,
        "optimizer_had_nonzero_gradient": bool(report["optimizer_had_nonzero_gradient"]),
        "parameter_probe_l1_change": float(report["parameter_probe_l1_change"]),
        "claim_boundary": (
            "Validation covers immutable token/logprob provenance and serving-to-learner "
            "numerical compatibility. It does not assert downstream quality improvement."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--train-report", type=Path, required=True)
    p.add_argument("--model-repo", required=True)
    p.add_argument("--model-revision", required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = validate(
        args.replay,
        args.metadata,
        args.train_report,
        model_repo=args.model_repo,
        model_revision=args.model_revision,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
