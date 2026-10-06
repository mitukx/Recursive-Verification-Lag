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
    low_precision: str | Path | None = None,
    profile: str | Path | None = None,
    weight_sync: str | Path | None = None,
    sync_async: str | Path | None = None,
    gpu_inventory: str | Path | None = None,
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

    if low_precision is not None:
        payload = _load(low_precision)
        git_shas.append(str(payload.get("git_sha", "unknown")))
        runs = payload.get("runs", {})
        comparisons = payload.get("comparisons", {})
        compact_runs = {}
        finite = True
        for precision in ("fp32", "bf16", "fp16"):
            row = runs.get(precision, {})
            compact_runs[precision] = {
                "success": bool(row.get("success")),
                "tokens_per_s": row.get("tokens_per_s"),
                "gpu_peak_memory_bytes": row.get("gpu_peak_memory_bytes"),
                "loss": (row.get("metrics") or {}).get("loss"),
                "grad_norm": (row.get("metrics") or {}).get("grad_norm"),
                "behavior_kl_estimate": (row.get("metrics") or {}).get("behavior_kl_estimate"),
            }
            finite = finite and bool(row.get("success"))
            for key in ("tokens_per_s", "gpu_peak_memory_bytes"):
                value = row.get(key)
                finite = finite and isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0
        evidence["low_precision_grpo"] = {
            "model": payload.get("model"),
            "device_name": payload.get("device_name"),
            "runs": compact_runs,
            "comparisons": comparisons,
        }
        checks["low_precision_grpo"] = (
            payload.get("all_precisions_finite") is True and finite
        )
        sources["low_precision_grpo"] = _source(low_precision)

    if profile is not None:
        payload = _load(profile)
        git_shas.append(str(payload.get("git_sha", "unknown")))
        stages = payload.get("named_stages") or []
        top_device = payload.get("top_self_device_time") or []
        evidence["grpo_profile"] = {
            "model":payload.get("model"),
            "precision":payload.get("precision"),
            "device_name":payload.get("device_name"),
            "tokens_per_s":payload.get("tokens_per_s"),
            "gpu_peak_memory_bytes":payload.get("gpu_peak_memory_bytes"),
            "trace_bytes":payload.get("trace_bytes"),
            "named_stages":stages,
            "top_self_device_time":top_device[:10],
        }
        checks["grpo_profile"] = (
            isinstance(payload.get("tokens_per_s"), (int, float))
            and math.isfinite(float(payload.get("tokens_per_s")))
            and float(payload.get("tokens_per_s")) > 0
            and int(payload.get("trace_bytes", 0)) > 0
            and len(stages) >= 4
        )
        sources["grpo_profile"] = _source(profile)


    if weight_sync is not None:
        payload = _load(weight_sync)
        git_shas.append(str(payload.get("git_sha", "unknown")))
        metrics = payload.get("metrics", {})
        latency = float(metrics.get("latency_ms_p50", 0.0))
        bandwidth = float(metrics.get("effective_gib_per_s_p50", 0.0))
        payload_bytes = int(metrics.get("payload_bytes", 0))
        world_size = int(metrics.get("world_size", 0))
        evidence["policy_weight_sync"] = {
            "model": payload.get("model"),
            "world_size": world_size,
            "payload_bytes": payload_bytes,
            "latency_ms_p50": latency,
            "latency_ms_p95": metrics.get("latency_ms_p95"),
            "effective_gib_per_s_p50": bandwidth,
        }
        checks["policy_weight_sync_measured"] = (
            world_size >= 2
            and payload_bytes > 0
            and math.isfinite(latency) and latency > 0
            and math.isfinite(bandwidth) and bandwidth > 0
        )
        sources["policy_weight_sync"] = _source(weight_sync)

    if sync_async is not None:
        payload = _load(sync_async)
        git_shas.append(str(payload.get("git_sha", "unknown")))
        metrics = payload.get("metrics", {})
        serial_tps = float(metrics.get("serial_tokens_per_s", 0.0))
        async_tps = float(metrics.get("async_tokens_per_s", 0.0))
        speedup = float(metrics.get("token_throughput_speedup", 0.0))
        evidence["sync_vs_async_rollout"] = {
            "model": payload.get("model"),
            "serial_tokens_per_s": serial_tps,
            "async_tokens_per_s": async_tps,
            "token_throughput_speedup": speedup,
            "request_throughput_speedup": metrics.get("request_throughput_speedup"),
        }
        checks["sync_vs_async_measured"] = (
            math.isfinite(serial_tps) and serial_tps > 0
            and math.isfinite(async_tps) and async_tps > 0
            and math.isfinite(speedup) and speedup > 0
        )
        sources["sync_vs_async_rollout"] = _source(sync_async)

    known_git_shas = [sha for sha in git_shas if sha and sha != "unknown"]
    checks["consistent_git_sha"] = (
        len(known_git_shas) == len(git_shas)
        and bool(known_git_shas)
        and len(set(known_git_shas)) == 1
    )

    hardware = None
    if gpu_inventory is not None:
        hardware = _load(gpu_inventory)
        sources["gpu_inventory"] = _source(gpu_inventory)

    required = {
        "qwen_transactional_rlvr",
        "fsdp_scaling_measured",
        "fsdp_resume_verified",
        "vllm_concurrency_sweep",
        "vllm_failover_verified",
        "low_precision_grpo",
        "grpo_profile",
        "policy_weight_sync_measured",
        "sync_vs_async_measured",
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
        "hardware": hardware,
        "git_sha": known_git_shas[0] if checks["consistent_git_sha"] else None,
        "claim_boundary": (
            "Complete means the required raw evidence files satisfy mechanical "
            "acceptance checks; it is not a claim of frontier-scale performance."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    status = "PASS" if report["complete"] else "INCOMPLETE"
    lines = [
        "# GPU Evidence Card",
        "",
        f"**Status:** {status}  ",
        f"**Git SHA:** {report.get('git_sha') or 'unverified'}",
        "",
    ]
    hardware = report.get("hardware") or {}
    gpus = hardware.get("gpus") or []
    if gpus:
        lines += [
            "## Hardware",
            "",
            " / ".join(
                f"{gpu.get('name','unknown')} ({gpu.get('memory_total_mb','?')} MiB)"
                for gpu in gpus
            ),
            "",
        ]
    evidence = report.get("evidence", {})
    qwen = evidence.get("qwen_rlvr")
    if qwen:
        lines += [
            "## RLVR",
            "",
            f"- Model: {qwen.get('model')}",
            f"- Held-out accuracy: {qwen.get('before_accuracy')} -> {qwen.get('after_accuracy')} "
            f"(delta {qwen.get('accuracy_delta')})",
            f"- Transactional promotion records: {qwen.get('promotion_records')}",
            "",
        ]
    fsdp = evidence.get("fsdp_scaling")
    if fsdp:
        lines += [
            "## Distributed training",
            "",
            f"- Throughput: {fsdp.get('single_tokens_per_s')} -> {fsdp.get('multi_tokens_per_s')} tokens/s",
            f"- 1->2 GPU speedup: {fsdp.get('speedup')}",
            f"- Scaling efficiency: {fsdp.get('scaling_efficiency')}",
            f"- Resume verified: {bool(evidence.get('fsdp_resume',{}).get('resumed_from_checkpoint'))}",
            "",
        ]
    serving = evidence.get("vllm_serving")
    if serving:
        lines += [
            "## Serving",
            "",
            f"- Peak measured tokens/s: {serving.get('max_tokens_per_s')}",
            f"- Peak measured requests/s: {serving.get('max_requests_per_s')}",
            "",
        ]
    precision = evidence.get("low_precision_grpo")
    if precision:
        lines += [
            "## Low-precision RL numerics",
            "",
            f"- Device: {precision.get('device_name')}",
            f"- FP32 tokens/s: {precision.get('runs',{}).get('fp32',{}).get('tokens_per_s')}",
            f"- BF16 tokens/s: {precision.get('runs',{}).get('bf16',{}).get('tokens_per_s')}",
            f"- FP16 tokens/s: {precision.get('runs',{}).get('fp16',{}).get('tokens_per_s')}",
            f"- BF16 relative loss error: {precision.get('comparisons',{}).get('bf16',{}).get('relative_loss_error')}",
            f"- FP16 relative loss error: {precision.get('comparisons',{}).get('fp16',{}).get('relative_loss_error')}",
            "",
        ]
    profile_evidence = evidence.get("grpo_profile")
    if profile_evidence:
        top = profile_evidence.get("top_self_device_time") or []
        hottest = top[0] if top else {}
        lines += [
            "## GRPO profile",
            "",
            f"- Profiled precision: {profile_evidence.get('precision')}",
            f"- Profiled tokens/s: {profile_evidence.get('tokens_per_s')}",
            f"- Trace bytes: {profile_evidence.get('trace_bytes')}",
            f"- Hottest self-device op: {hottest.get('name')} ({hottest.get('self_device_time_us')} us)",
            "",
        ]
    weight_sync_evidence = evidence.get("policy_weight_sync")
    if weight_sync_evidence:
        lines += [
            "## Policy weight synchronization",
            "",
            f"- Payload: {weight_sync_evidence.get('payload_bytes')} bytes",
            f"- 2-rank p50 activation latency: {weight_sync_evidence.get('latency_ms_p50')} ms",
            f"- Effective p50 bandwidth: {weight_sync_evidence.get('effective_gib_per_s_p50')} GiB/s",
            "",
        ]
    sync_async_evidence = evidence.get("sync_vs_async_rollout")
    if sync_async_evidence:
        lines += [
            "## Sync vs async rollout",
            "",
            f"- Serial tokens/s: {sync_async_evidence.get('serial_tokens_per_s')}",
            f"- Async tokens/s: {sync_async_evidence.get('async_tokens_per_s')}",
            f"- Token-throughput ratio: {sync_async_evidence.get('token_throughput_speedup')}",
            "",
        ]
    failover = evidence.get("vllm_failover")
    if failover:
        lines += [
            "## Failure recovery",
            "",
            f"- Completion rate: {failover.get('completion_rate')}",
            f"- Successful failovers: {failover.get('failover_successes')}",
            "",
        ]
    if report.get("failed_checks") or report.get("missing_checks"):
        lines += [
            "## Unmet checks",
            "",
            f"- Failed: {', '.join(report.get('failed_checks') or []) or 'none'}",
            f"- Missing: {', '.join(report.get('missing_checks') or []) or 'none'}",
            "",
        ]
    lines += [
        "## Claim boundary",
        "",
        report["claim_boundary"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate hiring-grade GPU evidence into one auditable report."
    )
    parser.add_argument("--qwen")
    parser.add_argument("--fsdp")
    parser.add_argument("--fsdp-resume")
    parser.add_argument("--vllm")
    parser.add_argument("--failover")
    parser.add_argument("--low-precision")
    parser.add_argument("--profile")
    parser.add_argument("--weight-sync")
    parser.add_argument("--sync-async")
    parser.add_argument("--gpu-inventory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    report = summarize(
        qwen=args.qwen,
        fsdp=args.fsdp,
        fsdp_resume=args.fsdp_resume,
        vllm=args.vllm,
        failover=args.failover,
        low_precision=args.low_precision,
        profile=args.profile,
        weight_sync=args.weight_sync,
        sync_async=args.sync_async,
        gpu_inventory=args.gpu_inventory,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.markdown:
        Path(args.markdown).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_all and not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
