from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from src.rvl_systems.backends import ToyTabularBackend
from src.rvl_systems.benchmark_report import BenchmarkReport
from src.rvl_systems.event_log import ControlPlaneEventLog
from src.rvl_systems.rollout import RolloutRequest
from src.rvl_systems.scheduler import LeastLoadedScheduler, WorkerSlot
from src.rvl_systems.telemetry import Telemetry
from src.rvl_systems.worker_health import WorkerHealth


class FailFirstBackend:
    """Deterministically fail the first N calls, then behave normally."""

    def __init__(self, failures: int, *, latency_s: float) -> None:
        if failures < 0:
            raise ValueError("failures must be non-negative")
        self.remaining_failures = failures
        self.calls = 0
        self.delegate = ToyTabularBackend(base_latency_s=latency_s)

    async def generate(self, *args, **kwargs):
        self.calls += 1
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise ConnectionError("injected rollout worker failure")
        return await self.delegate.generate(*args, **kwargs)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic scheduler chaos benchmark with failover and recovery."
    )
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--recovery-requests", type=int, default=12)
    parser.add_argument("--latency-ms", type=float, default=2.0)
    parser.add_argument("--queue-limit", type=int, default=16)
    parser.add_argument("--output")
    parser.add_argument("--event-log")
    args = parser.parse_args()

    if args.requests <= 0 or args.recovery_requests <= 0:
        raise ValueError("request counts must be positive")

    telemetry = Telemetry()
    event_log = ControlPlaneEventLog()
    health = WorkerHealth(failure_threshold=1)
    flaky = FailFirstBackend(1, latency_s=args.latency_ms / 1000.0)
    workers = [
        WorkerSlot("w0-flaky", flaky, max_inflight=1),
        WorkerSlot(
            "w1-healthy",
            ToyTabularBackend(base_latency_s=args.latency_ms / 1000.0),
            max_inflight=1,
        ),
        WorkerSlot(
            "w2-healthy",
            ToyTabularBackend(base_latency_s=args.latency_ms / 1000.0),
            max_inflight=1,
        ),
    ]
    scheduler = LeastLoadedScheduler(
        workers,
        queue_limit=args.queue_limit,
        max_attempts_per_request=len(workers),
        telemetry=telemetry,
        health=health,
        event_log=event_log,
    )

    first = [
        RolloutRequest(f"fault-{i}", "benchmark", samples=1, seed=i)
        for i in range(args.requests)
    ]
    start = time.perf_counter()
    phase1 = await scheduler.run(first)
    phase1_elapsed = time.perf_counter() - start

    if len(phase1) != args.requests:
        raise RuntimeError(
            f"failover lost requests: got {len(phase1)} / {args.requests} samples"
        )
    if health.is_available("w0-flaky"):
        raise RuntimeError("fault injection did not quarantine the failing worker")

    before_recovery_dispatches = telemetry.snapshot().get(
        "scheduler.dispatch.w0-flaky", 0.0
    )
    health.recover("w0-flaky")
    flaky.remaining_failures = 0

    second = [
        RolloutRequest(f"recovery-{i}", "benchmark", samples=1, seed=10_000 + i)
        for i in range(args.recovery_requests)
    ]
    recovery_start = time.perf_counter()
    phase2 = await scheduler.run(second)
    recovery_elapsed = time.perf_counter() - recovery_start

    after_recovery_dispatches = telemetry.snapshot().get(
        "scheduler.dispatch.w0-flaky", 0.0
    )
    recovered_dispatches = after_recovery_dispatches - before_recovery_dispatches
    if len(phase2) != args.recovery_requests:
        raise RuntimeError(
            f"recovery phase lost requests: got {len(phase2)} / "
            f"{args.recovery_requests} samples"
        )
    if recovered_dispatches <= 0:
        raise RuntimeError("recovered worker did not re-enter scheduler rotation")

    snap = telemetry.snapshot()
    metrics = {
        "phase1_elapsed_s": phase1_elapsed,
        "phase1_samples_per_s": len(phase1) / max(phase1_elapsed, 1e-12),
        "recovery_elapsed_s": recovery_elapsed,
        "recovery_samples_per_s": len(phase2) / max(recovery_elapsed, 1e-12),
        "phase1_samples": len(phase1),
        "recovery_samples": len(phase2),
        "injected_failures": 1,
        "flaky_backend_calls": flaky.calls,
        "recovered_worker_dispatches": recovered_dispatches,
        **snap,
    }
    report = BenchmarkReport(
        name="scheduler-chaos",
        metrics=metrics,
        config={
            "workers": len(workers),
            "requests": args.requests,
            "recovery_requests": args.recovery_requests,
            "latency_ms": args.latency_ms,
            "queue_limit": args.queue_limit,
            "failure_threshold": health.failure_threshold,
            "max_attempts_per_request": scheduler.max_attempts_per_request,
        },
        git_sha=os.environ.get("GITHUB_SHA", "unknown"),
    )
    event_log.validate()
    if args.event_log:
        event_log.write_jsonl(args.event_log)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        report.write_json(args.output)
    print(json.dumps(report.payload(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
