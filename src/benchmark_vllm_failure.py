from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import time

from src.rvl_systems.backends import VLLMHTTPBackend
from src.rvl_systems.benchmark_report import BenchmarkReport
from src.rvl_systems.rollout import RolloutRequest
from src.rvl_systems.scheduler import LeastLoadedScheduler, WorkerSlot
from src.rvl_systems.telemetry import Telemetry
from src.rvl_systems.worker_health import WorkerHealth


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kill one local vLLM process during load and verify cross-endpoint failover."
    )
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--kill-pid", type=int, required=True)
    parser.add_argument("--kill-after-s", type=float, default=2.0)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--concurrency-per-worker", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", default="artifacts/vllm-failover.json")
    args = parser.parse_args()

    if len(args.endpoint) < 2:
        raise ValueError("failure benchmark requires at least two endpoints")
    telemetry = Telemetry()
    health = WorkerHealth(failure_threshold=1)
    workers = [
        WorkerSlot(
            f"endpoint-{index}",
            VLLMHTTPBackend(
                endpoint=endpoint,
                model=args.model,
                timeout_s=30.0,
                max_tokens=args.max_tokens,
            ),
            max_inflight=args.concurrency_per_worker,
        )
        for index, endpoint in enumerate(args.endpoint)
    ]
    scheduler = LeastLoadedScheduler(
        workers,
        queue_limit=max(args.requests, 8),
        request_timeout_s=40.0,
        max_attempts_per_request=len(workers),
        telemetry=telemetry,
        health=health,
    )

    async def inject_failure() -> float:
        await asyncio.sleep(args.kill_after_s)
        started = time.perf_counter()
        os.kill(args.kill_pid, signal.SIGTERM)
        telemetry.increment("failure_injection.process_sigterm", 1)
        return started

    requests = [
        RolloutRequest(
            f"gpu-failure-{i}",
            "Write exactly 128 tokens explaining distributed inference reliability.",
            samples=1,
            temperature=0.0,
            seed=20261006 + i,
            estimated_tokens=args.max_tokens,
            workload_id="gpu-failure",
        )
        for i in range(args.requests)
    ]

    injection = asyncio.create_task(inject_failure())
    start = time.perf_counter()
    generations = await scheduler.run(requests)
    injected_at = await injection
    wall_s = time.perf_counter() - start

    snap = telemetry.snapshot()
    completed = len(generations)
    if completed != args.requests:
        raise RuntimeError(
            f"lost requests under worker failure: {completed}/{args.requests}"
        )
    if snap.get("scheduler.failures", 0.0) < 1:
        raise RuntimeError("failure injection did not produce a scheduler failure")
    if snap.get("scheduler.failover_successes", 0.0) < 1:
        raise RuntimeError("no request was recovered by failover")

    metrics = {
        "requests": args.requests,
        "completed_requests": completed,
        "completion_rate": completed / args.requests,
        "wall_s": wall_s,
        "requests_per_s": completed / max(wall_s, 1e-12),
        "failure_injected_at_s": injected_at - start,
        **snap,
    }
    report = BenchmarkReport(
        name="vllm-process-failure",
        metrics=metrics,
        config={
            "endpoints": len(args.endpoint),
            "concurrency_per_worker": args.concurrency_per_worker,
            "max_tokens": args.max_tokens,
            "kill_after_s": args.kill_after_s,
        },
        git_sha=os.environ.get("GITHUB_SHA", "unknown"),
        model=args.model,
    )
    report.write_json(args.output)
    print(json.dumps(report.payload(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
