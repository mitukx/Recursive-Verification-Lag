from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return {"path": str(p), "sha256": _sha256(p), "bytes": p.stat().st_size}


def summarize(
    *,
    qwen: str | Path | None = None,
    fsdp: str | Path | None = None,
    fsdp_resume: str | Path | None = None,
    vllm: str | Path | None = None,
    failover: str | Path | None = None,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    evidence: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    git_shas: list[str] = []

    if qwen is not None:
        payload = _load(qwen)
        metrics = payload.get("metrics", {})
        transactional = bool(metrics.get("transactional_promotion", 0))
        git_shas.append(str(payload.get("git_sha", "unknown")))
        promotion_records = int(metrics.get("promotion_records", 0))
        evidence["qwen_rlvr"] = {
            "model": payload.get("model"),
            "before_accuracy": metrics.get("before_accuracy"),
            "after_accuracy": metrics.get("after_accuracy"),
            "accuracy_delta": metrics.get("accuracy_delta"),
            "transactional_promotion": transactional,
            "promotion_records": promotion_records,
            "promotion_head_sha256": metrics.get("promotion_head_sha256"),
        }
        before = metrics.get("before_accuracy")
        after = metrics.get("after_accuracy")
        head = metrics.get("promotion_head_sha256")
        checks["qwen_transactional_rlvr"] = (
            transactional
            and promotion_records > 0
            and isinstance(head, str) and len(head) == 64
            and isinstance(before, (int, float)) and math.isfinite(float(before)) and 0 <= float(before) <= 1
            and isinstance(after, (int, float)) and math.isfinite(float(after)) and 0 <= float(after) <= 1
        )
        sources["qwen_rlvr"] = _source(qwen)

    if fsdp is not None:
        payload = _load(fsdp)
        git_shas.append(str(payload.get("git_sha", "unknown")))
        evidence["fsdp_scaling"] = {
            "single_world_size": payload.get("single_world_size"),
            "multi_world_size": payload.get("multi_world_size"),
            "single_tokens_per_s": payload.get("single_tokens_per_s"),
            "multi_tokens_per_s": payload.get("multi_tokens_per_s"),
            "speedup": payload.get("speedup"),
            "scaling_efficiency": payload.get("scaling_efficiency"),
            "single_gpu_peak_memory_bytes": payload.get("single_gpu_peak_memory_bytes"),
            "multi_gpu_peak_memory_bytes": payload.get("multi_gpu_peak_memory_bytes"),
        }
        checks["fsdp_scaling_measured"] = (
            int(payload.get("multi_world_size", 0)) > int(payload.get("single_world_size", 0))
            and math.isfinite(float(payload.get("speedup", 0.0)))
            and math.isfinite(float(payload.get("scaling_efficiency", 0.0)))
            and float(payload.get("speedup", 0.0)) > 0.0
            and float(payload.get("scaling_efficiency", 0.0)) > 0.0
        )
        sources["fsdp_scaling"] = _source(fsdp)

    if fsdp_resume is not None:
        payload = _load(fsdp_resume)
        git_shas.append(str(payload.get("git_sha", "unknown")))
        resumed = payload.get("resumed_from_checkpoint") is True
        evidence["fsdp_resume"] = {
            "world_size": payload.get("world_size"),
            "resumed_from_checkpoint": resumed,
            "tokens_per_s": payload.get("tokens_per_s"),
        }
        checks["fsdp_resume_verified"] = resumed
        sources["fsdp_resume"] = _source(fsdp_resume)

    if vllm is not None:
        payload = _load(vllm)
        if not isinstance(payload, list):
            raise ValueError("vLLM summary must be a list of benchmark reports")
        phases = []
        for row in payload:
            git_shas.append(str(row.get("git_sha", "unknown")))
            metrics = row.get("metrics", {})
            config = row.get("config", {})
            phases.append({
                "concurrency": int(config.get("concurrency", 0)),
                "requests_per_s": float(metrics.get("requests_per_s", 0.0)),
                "tokens_per_s": float(metrics.get("tokens_per_s", 0.0)),
                "ttft_ms_p95": float(metrics.get("ttft_ms_p95", 0.0)),
                "tbt_ms_p95": float(metrics.get("tbt_ms_p95", 0.0)),
                "latency_ms_p95": float(metrics.get("latency_ms_p95", 0.0)),
            })
        evidence["vllm_serving"] = {
            "phases": phases,
            "max_tokens_per_s": max((x["tokens_per_s"] for x in phases), default=0.0),
            "max_requests_per_s": max((x["requests_per_s"] for x in phases), default=0.0),
        }
        unique_concurrency = {x["concurrency"] for x in phases if x["concurrency"] > 0}
        checks["vllm_concurrency_sweep"] = (
            len(unique_concurrency) >= 3
            and all(math.isfinite(x["tokens_per_s"]) and x["tokens_per_s"] > 0 for x in phases)
        )
        sources["vllm_serving"] = _source(vllm)

    if failover is not None:
        payload = _load(failover)
        git_shas.append(str(payload.get("git_sha", "unknown")))
        metrics = payload.get("metrics", {})
        completion = float(metrics.get("completion_rate", 0.0))
        successes = float(metrics.get("scheduler.failover_successes", 0.0))
        failures = float(metrics.get("scheduler.failures", 0.0))
        evidence["vllm_failover"] = {
            "completion_rate": completion,
            "failover_successes": successes,
            "scheduler_failures": failures,
            "requests": metrics.get("requests"),
            "wall_s": metrics.get("wall_s"),
        }
        checks["vllm_failover_verified"] = (
            completion == 1.0 and successes >= 1.0 and failures >= 1.0
        )
        sources["vllm_failover"] = _source(failover)

    known_git_shas = [sha for sha in git_shas if sha and sha != "unknown"]
    checks["consistent_git_sha"] = (
        len(known_git_shas) == len(git_shas)
        and bool(known_git_shas)
        and len(set(known_git_shas)) == 1
    )

    required = {
        "qwen_transactional_rlvr",
        "fsdp_scaling_measured",
        "fsdp_resume_verified",
        "vllm_concurrency_sweep",
        "vllm_failover_verified",
        "consistent_git_sha",
    }
    missing = sorted(required - checks.keys())
    failed = sorted(key for key, ok in checks.items() if not ok)
    complete = not missing and not failed
    return {
        "schema": 1,
        "complete": complete,
        "checks": checks,
        "missing_checks": missing,
        "failed_checks": failed,
        "evidence": evidence,
        "sources": sources,
        "git_sha": known_git_shas[0] if checks["consistent_git_sha"] else None,
        "claim_boundary": (
            "Complete means the required raw evidence files satisfy mechanical "
            "acceptance checks; it is not a claim of frontier-scale performance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate hiring-grade GPU evidence into one auditable report."
    )
    parser.add_argument("--qwen")
    parser.add_argument("--fsdp")
    parser.add_argument("--fsdp-resume")
    parser.add_argument("--vllm")
    parser.add_argument("--failover")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    report = summarize(
        qwen=args.qwen,
        fsdp=args.fsdp,
        fsdp_resume=args.fsdp_resume,
        vllm=args.vllm,
        failover=args.failover,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_all and not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
