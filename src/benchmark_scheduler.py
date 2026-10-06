from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from src.rvl_systems.backends import ToyTabularBackend
from src.rvl_systems.benchmark_report import BenchmarkReport
from src.rvl_systems.rollout import RolloutRequest
from src.rvl_systems.scheduler import LeastLoadedScheduler, WorkerSlot
from src.rvl_systems.telemetry import Telemetry


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--latency-ms", type=float, default=5.0)
    parser.add_argument("--queue-limit", type=int, default=16)
    parser.add_argument("--output")
    args = parser.parse_args()

    telemetry = Telemetry()
    workers = [
        WorkerSlot(
            f"w{i}",
            ToyTabularBackend(base_latency_s=args.latency_ms / 1000.0),
            max_inflight=1,
        )
        for i in range(args.workers)
    ]
    scheduler = LeastLoadedScheduler(
        workers,
        queue_limit=args.queue_limit,
        telemetry=telemetry,
    )
    reqs = [RolloutRequest(str(i), "benchmark", samples=1, seed=i) for i in range(args.requests)]
    start = time.perf_counter()
    out = await scheduler.run(reqs)
    elapsed = time.perf_counter() - start
    metrics = {
        "elapsed_s": elapsed,
        "samples_per_s": len(out) / max(elapsed, 1e-12),
        "samples": len(out),
        **telemetry.snapshot(),
    }
    report = BenchmarkReport(
        name="scheduler",
        metrics=metrics,
        config={
            "workers": args.workers,
            "requests": args.requests,
            "latency_ms": args.latency_ms,
            "queue_limit": args.queue_limit,
        },
        git_sha=os.environ.get("GITHUB_SHA", "unknown"),
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        report.write_json(args.output)
    print(json.dumps(report.payload(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
