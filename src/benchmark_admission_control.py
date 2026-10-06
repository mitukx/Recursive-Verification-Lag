from __future__ import annotations

import argparse
import asyncio
import json
import os

from src.rvl_systems.admission import (
    FairWorkloadAdmission,
    OverloadedError,
)
from src.rvl_systems.benchmark_report import BenchmarkReport
from src.rvl_systems.telemetry import Telemetry


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic mixed-workload admission benchmark. "
            "Validates fairness, bounded work, and overload shedding."
        )
    )
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--max-queued", type=int, default=8)
    parser.add_argument("--requests-per-workload", type=int, default=8)
    parser.add_argument("--hold-ms", type=float, default=2.0)
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.capacity <= 0:
        raise ValueError("capacity must be positive")
    if args.max_queued < 0:
        raise ValueError("max-queued must be non-negative")
    if args.requests_per_workload <= 0:
        raise ValueError("requests-per-workload must be positive")

    telemetry = Telemetry()
    controller = FairWorkloadAdmission(
        args.capacity,
        max_queued_units=args.max_queued,
        telemetry=telemetry,
    )

    completion_order: list[str] = []
    shed = 0

    async def submit(
        workload_id: str,
        cost: int,
    ) -> None:
        nonlocal shed
        try:
            lease = await controller.acquire(
                workload_id,
                cost,
            )
        except OverloadedError:
            shed += 1
            return
        try:
            completion_order.append(workload_id)
            await asyncio.sleep(args.hold_ms / 1000.0)
        finally:
            await lease.release()

    tasks = []
    for index in range(args.requests_per_workload):
        tasks.append(
            asyncio.create_task(
                submit("train", 2)
            )
        )
        tasks.append(
            asyncio.create_task(
                submit("eval", 1)
            )
        )
        if index % 2 == 0:
            tasks.append(
                asyncio.create_task(
                    submit("monitor", 1)
                )
            )

    await asyncio.gather(*tasks)
    snap = telemetry.snapshot()

    per_workload = {
        name: completion_order.count(name)
        for name in ("train", "eval", "monitor")
    }
    metrics = {
        "submitted": len(tasks),
        "completed": len(completion_order),
        "shed": shed,
        "train_completed": per_workload["train"],
        "eval_completed": per_workload["eval"],
        "monitor_completed": per_workload["monitor"],
        "max_in_use_units": snap.get(
            "admission.in_use_units.max",
            0.0,
        ),
        "max_queued_units": snap.get(
            "admission.queued_units.max",
            0.0,
        ),
        **snap,
    }
    if metrics["max_in_use_units"] > args.capacity:
        raise RuntimeError("admission exceeded global work capacity")
    if metrics["max_queued_units"] > args.max_queued:
        raise RuntimeError("admission exceeded queued-work capacity")
    if len(completion_order) == 0:
        raise RuntimeError("all synthetic work was shed")

    report = BenchmarkReport(
        name="workload-admission",
        metrics=metrics,
        config={
            "capacity_units": args.capacity,
            "max_queued_units": args.max_queued,
            "requests_per_workload": args.requests_per_workload,
            "hold_ms": args.hold_ms,
            "synthetic": "true",
        },
        git_sha=os.environ.get("GITHUB_SHA", "unknown"),
    )
    if args.output:
        report.write_json(args.output)
    print(json.dumps(report.payload(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
