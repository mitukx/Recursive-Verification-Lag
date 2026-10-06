from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.rvl_systems.benchmark_report import BenchmarkReport
from src.rvl_systems.triton_grpo import (
    torch_grpo_surrogate,
    triton_available,
    triton_grpo_surrogate,
)


def benchmark_ms(torch, fn, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iterations


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark fused tokenwise GRPO Triton against PyTorch eager "
            "with forward and gradient parity checks."
        )
    )
    parser.add_argument("--sizes", default="4096,16384,65536,262144,1048576")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--output", default="artifacts/triton-grpo.json")
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required") from exc
    if not triton_available():
        raise SystemExit("CUDA + Triton are required for this benchmark")

    sizes = [int(item) for item in args.sizes.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("sizes must contain positive integers")

    torch.manual_seed(20261006)
    device = torch.device("cuda")
    metrics: dict[str, float | int | str] = {}
    worst_surrogate_error = 0.0
    worst_gradient_error = 0.0
    best_speedup = float("inf")

    for size in sizes:
        old = torch.randn(size, device=device, dtype=torch.float32) * 2.0
        current_seed = old + torch.randn(
            size,
            device=device,
            dtype=torch.float32,
        ) * 0.35
        advantages = torch.randn(
            size,
            device=device,
            dtype=torch.float32,
        )

        reference_x = current_seed.detach().clone().requires_grad_(True)
        reference, _, _, _ = torch_grpo_surrogate(
            reference_x,
            old,
            advantages,
            clip_eps=args.clip_eps,
        )
        reference_grad = torch.autograd.grad(
            reference.mean(),
            reference_x,
        )[0]

        triton_x = current_seed.detach().clone().requires_grad_(True)
        candidate, _, _, _ = triton_grpo_surrogate(
            triton_x,
            old,
            advantages,
            clip_eps=args.clip_eps,
        )
        candidate_grad = torch.autograd.grad(
            candidate.mean(),
            triton_x,
        )[0]
        torch.cuda.synchronize()

        surrogate_error = float(
            (reference.detach() - candidate.detach()).abs().max().item()
        )
        gradient_error = float(
            (reference_grad - candidate_grad).abs().max().item()
        )
        worst_surrogate_error = max(
            worst_surrogate_error,
            surrogate_error,
        )
        worst_gradient_error = max(
            worst_gradient_error,
            gradient_error,
        )

        with torch.no_grad():
            torch_ms = benchmark_ms(
                torch,
                lambda: torch_grpo_surrogate(
                    current_seed,
                    old,
                    advantages,
                    clip_eps=args.clip_eps,
                )[0],
                warmup=args.warmup,
                iterations=args.iterations,
            )
            triton_ms = benchmark_ms(
                torch,
                lambda: triton_grpo_surrogate(
                    current_seed,
                    old,
                    advantages,
                    clip_eps=args.clip_eps,
                )[0],
                warmup=args.warmup,
                iterations=args.iterations,
            )

        speedup = torch_ms / max(triton_ms, 1e-12)
        best_speedup = min(best_speedup, speedup)
        prefix = f"n{size}"
        metrics[f"{prefix}_torch_forward_ms"] = torch_ms
        metrics[f"{prefix}_triton_forward_ms"] = triton_ms
        metrics[f"{prefix}_forward_speedup"] = speedup
        metrics[f"{prefix}_surrogate_max_abs_error"] = surrogate_error
        metrics[f"{prefix}_gradient_max_abs_error"] = gradient_error

    metrics["worst_surrogate_max_abs_error"] = worst_surrogate_error
    metrics["worst_gradient_max_abs_error"] = worst_gradient_error
    metrics["minimum_forward_speedup"] = best_speedup
    metrics["gpu_name"] = torch.cuda.get_device_name(0)

    if worst_surrogate_error > 2e-5:
        raise RuntimeError(
            f"Triton surrogate parity failed: {worst_surrogate_error}"
        )
    if worst_gradient_error > 2e-5:
        raise RuntimeError(
            f"Triton gradient parity failed: {worst_gradient_error}"
        )

    report = BenchmarkReport(
        name="triton-grpo",
        metrics=metrics,
        config={
            "sizes": args.sizes,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "clip_eps": args.clip_eps,
            "dtype": "fp32",
        },
        git_sha=os.environ.get("GITHUB_SHA", "unknown"),
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    report.write_json(target)
    print(json.dumps(report.payload(), indent=2))


if __name__ == "__main__":
    main()
