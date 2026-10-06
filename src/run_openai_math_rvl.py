from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.rvl_systems.openai_math import (
    LeanTrustedVerifier,
    OpenAIMathManifest,
    hash_chain,
    sha256_file,
    summarize_rows,
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} is not an object")
            rows.append(row)
    return rows


def validate_record(row: dict, task_ids: set[str], index: int) -> None:
    if row.get("task_id") not in task_ids:
        raise ValueError(f"record {index}: task not in selected split")
    if not isinstance(row.get("response"), str) or not row["response"].strip():
        raise ValueError(f"record {index}: non-empty response required")
    score = row.get("proxy_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
        raise ValueError(f"record {index}: proxy_score must be in [0,1]")
    for key in ("policy_version", "verifier_version"):
        value = row.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"record {index}: {key} must be nonnegative int")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("configs/openai_math_rvl_v1.json"))
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "heldout"], default="development")
    parser.add_argument("--proxy-threshold", type=float, default=0.5)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--sandbox-prefix-json", default="[]")
    parser.add_argument("--allow-unsandboxed", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    prefix = json.loads(args.sandbox_prefix_json)
    if not isinstance(prefix, list) or not all(isinstance(x, str) and x for x in prefix):
        raise SystemExit("--sandbox-prefix-json must be a JSON argv list")

    full_manifest = OpenAIMathManifest.load(args.manifest)
    manifest = full_manifest.select(args.split)
    verifier = LeanTrustedVerifier(
        args.checkout,
        manifest,
        timeout_s=args.timeout_s,
        sandbox_prefix=tuple(prefix),
        allow_unsandboxed=args.allow_unsandboxed,
    )
    records = read_jsonl(args.input)
    tasks = manifest.task_map()
    for i, row in enumerate(records, 1):
        validate_record(row, set(tasks), i)

    out = []
    for sequence, row in enumerate(records):
        task = tasks[row["task_id"]]
        result = verifier.verify_candidate(task.task_id, row["response"])
        policy_version = int(row.get("policy_version", 0))
        verifier_version = int(row.get("verifier_version", 0))
        out.append({
            "sequence": sequence,
            "task_id": task.task_id,
            "family": task.family,
            "mechanism": task.mechanism,
            "split": task.split,
            "proxy_score": float(row["proxy_score"]),
            "trusted_pass": bool(result.passed),
            "policy_version": policy_version,
            "verifier_version": verifier_version,
            "stale_age": max(0, policy_version - verifier_version),
            "lean_returncode": result.returncode,
            "lean_latency_s": result.latency_s,
            "candidate_sha256": result.candidate_sha256,
            "source_sha256": result.source_sha256,
            "source_blob_sha": result.source_blob_sha,
            "metadata": row.get("metadata", {}),
            "stdout_tail": result.stdout_tail,
            "stderr_tail": result.stderr_tail,
        })

    chained, head = hash_chain(out)
    summary = summarize_rows(out, args.proxy_threshold)
    summary["protocol"] = {
        "manifest_fingerprint": full_manifest.fingerprint,
        "input_sha256": sha256_file(args.input),
        "split": args.split,
        "upstream_repo": full_manifest.upstream_repo,
        "upstream_sha": full_manifest.upstream_sha,
        "lean_toolchain": full_manifest.lean_toolchain,
        "audit_chain_head": head,
        "sandboxed": bool(prefix),
    }

    args.output.mkdir(parents=True)
    with (args.output / "rows.jsonl").open("w", encoding="utf-8") as stream:
        for row in chained:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
