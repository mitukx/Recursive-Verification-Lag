from __future__ import annotations

import argparse
import asyncio
import json
import time

from src.rvl_systems.backends import ToyTabularBackend
from src.rvl_systems.rollout import RolloutRequest
from src.rvl_systems.scheduler import LeastLoadedScheduler, WorkerSlot
from src.rvl_systems.telemetry import Telemetry


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--latency-ms", type=float, default=5.0)
    parser.add_argument("--queue-limit", type=int, default=16)
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
    print(json.dumps({
        "workers": args.workers,
        "requests": args.requests,
        "elapsed_s": elapsed,
        "samples_per_s": len(out) / max(elapsed, 1e-12),
        "telemetry": telemetry.snapshot(),
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
