from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

from src.rvl_systems.benchmark_report import BenchmarkReport


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    fraction = position - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def summarize_latencies(latencies_s: list[float], payload_bytes: int) -> dict[str, float | int]:
    if payload_bytes <= 0 or not latencies_s:
        raise ValueError("positive payload and non-empty latency samples required")
    if any((not math.isfinite(x) or x <= 0) for x in latencies_s):
        raise ValueError("latencies must be finite and positive")
    median = statistics.median(latencies_s)
    return {
        "payload_bytes": payload_bytes,
        "iterations": len(latencies_s),
        "latency_ms_p50": 1000.0 * median,
        "latency_ms_p95": 1000.0 * percentile(latencies_s, 0.95),
        "latency_ms_max": 1000.0 * max(latencies_s),
        "effective_gib_per_s_p50": payload_bytes / median / (1024.0 ** 3),
    }


def _dtype(torch, precision: str):
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported precision: {precision}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure learner-to-rollout policy tensor broadcast overhead with torch.distributed."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("invalid warmup/iteration count")

    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU weight-sync evidence")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world < 2:
        raise RuntimeError("weight-sync benchmark requires at least two ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dtype = _dtype(torch, args.precision)
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requested but unsupported on this GPU")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    parameters = [p.data for p in model.parameters()]
    payload_bytes = sum(p.numel() * p.element_size() for p in parameters)
    if payload_bytes <= 0:
        raise RuntimeError("model has no parameter payload")

    # Corrupt a small receiver slice before the first broadcast so the
    # correctness check verifies that data actually moved from rank 0.
    first = parameters[0].view(-1)
    if rank != 0:
        first[: min(first.numel(), 1024)].add_(1)

    def sync_once() -> float:
        dist.barrier()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for tensor in parameters:
            dist.broadcast(tensor, src=0)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        dist.barrier()
        return elapsed

    for _ in range(args.warmup):
        sync_once()

    probe = first[: min(first.numel(), 4096)].float().sum().detach()
    probes = [torch.zeros_like(probe) for _ in range(world)]
    dist.all_gather(probes, probe)
    reference = probes[0]
    if any(not torch.allclose(x, reference, rtol=0, atol=0) for x in probes[1:]):
        raise RuntimeError("post-broadcast parameter probe mismatch")

    latencies = [sync_once() for _ in range(args.iterations)]
    local = torch.tensor(latencies, dtype=torch.float64, device=device)
    gathered = [torch.zeros_like(local) for _ in range(world)]
    dist.all_gather(gathered, local)

    if rank == 0:
        all_latencies = [float(x) for row in gathered for x in row.cpu().tolist()]
        # Each rank measures the same collective. Use the worst rank per
        # iteration as end-to-end activation latency.
        collective_latencies = [
            max(float(gathered[r][i].item()) for r in range(world))
            for i in range(args.iterations)
        ]
        metrics = summarize_latencies(collective_latencies, payload_bytes)
        metrics.update({
            "world_size": world,
            "per_rank_samples": len(all_latencies),
            "aggregate_payload_bytes": payload_bytes * (world - 1),
        })
        report = BenchmarkReport(
            name="policy-weight-broadcast",
            metrics=metrics,
            config={
                "world_size": world,
                "precision": args.precision,
                "warmup": args.warmup,
                "iterations": args.iterations,
            },
            git_sha=os.environ.get("GITHUB_SHA", "unknown"),
            model=args.model,
        )
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        report.write_json(path)
        print(json.dumps(report.payload(), indent=2, sort_keys=True))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
