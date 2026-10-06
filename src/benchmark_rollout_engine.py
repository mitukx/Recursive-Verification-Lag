from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from src.rvl_systems.backends import ToyTabularBackend
from src.rvl_systems.benchmark_report import BenchmarkReport
from src.rvl_systems.rollout import AsyncRolloutEngine, RolloutRequest
from src.rvl_systems.telemetry import Telemetry


async def run_benchmark(requests: int, samples: int, latency_ms: float, concurrency: int) -> dict:
    backend = ToyTabularBackend(actions=("0", "1", "2", "3"), base_latency_s=latency_ms / 1000.0)
    reqs = [
        RolloutRequest(f"p{i}", f"prompt {i}", samples=samples, seed=1000 + i)
        for i in range(requests)
    ]

    serial_start = time.perf_counter()
    serial_count = 0
    for req in reqs:
        serial_count += len(await backend.generate(
            req.prompt_id,
            req.prompt,
            n=req.samples,
            temperature=req.temperature,
            seed=req.seed,
        ))
    serial_s = time.perf_counter() - serial_start

    telemetry = Telemetry()
    engine = AsyncRolloutEngine(backend, max_concurrency=concurrency, telemetry=telemetry)
    async_start = time.perf_counter()
    generated = await engine.run(reqs)
    async_s = time.perf_counter() - async_start

    return {
        "samples": len(generated),
        "serial_samples": serial_count,
        "serial_wall_s": serial_s,
        "async_wall_s": async_s,
        "speedup": serial_s / max(async_s, 1e-12),
        "async_samples_per_s": len(generated) / max(async_s, 1e-12),
        **telemetry.snapshot(),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--latency-ms", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output")
    args = parser.parse_args()
    metrics = await run_benchmark(
        args.requests, args.samples, args.latency_ms, args.concurrency
    )
    report = BenchmarkReport(
        name="async-rollout",
        metrics=metrics,
        config={
            "requests": args.requests,
            "samples_per_request": args.samples,
            "latency_ms_per_sample": args.latency_ms,
            "concurrency": args.concurrency,
        },
        git_sha=os.environ.get("GITHUB_SHA", "unknown"),
    )
    payload = report.payload()
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        report.write_json(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
