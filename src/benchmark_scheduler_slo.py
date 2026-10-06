from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from statistics import median

from src.rvl_systems.benchmark_report import BenchmarkReport
from src.rvl_systems.rollout import RolloutRequest
from src.rvl_systems.scheduler import LeastLoadedScheduler, WorkerSlot
from src.rvl_systems.types import Generation


class StaticDelayBackend:
    def __init__(
        self,
        delay_s: float,
        response: str,
    ) -> None:
        self.delay_s = delay_s
        self.response = response

    async def generate(
        self,
        prompt_id,
        prompt,
        *,
        n,
        temperature,
        seed,
    ):
        await asyncio.sleep(self.delay_s)
        return [
            Generation(
                prompt_id,
                prompt,
                self.response,
                0.0,
                1,
                self.delay_s,
            )
            for _ in range(n)
        ]


def percentile(
    values: list[float],
    q: float,
) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("values must be non-empty")
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return (
        ordered[lo] * (1.0 - frac)
        + ordered[hi] * frac
    )


async def one_trial(
    *,
    hedge_after_s: float | None,
    primary_delay_s: float,
    backup_delay_s: float,
    deadline_s: float,
) -> tuple[bool, float]:
    primary = WorkerSlot(
        "primary",
        StaticDelayBackend(
            primary_delay_s,
            "primary",
        ),
    )
    backup = WorkerSlot(
        "backup",
        StaticDelayBackend(
            backup_delay_s,
            "backup",
        ),
    )

    # Simulate a worker that is usually fastest but has entered a
    # tail-latency event. The stale EWMA intentionally routes the request
    # to primary first.
    primary.latency_ewma_s = 0.001
    backup.latency_ewma_s = max(
        backup_delay_s,
        0.002,
    )

    scheduler = LeastLoadedScheduler(
        [primary, backup],
        hedge_after_s=hedge_after_s,
        max_attempts_per_request=2,
    )
    request = RolloutRequest(
        "slo",
        "benchmark",
        samples=1,
        seed=1,
        deadline_s=deadline_s,
    )
    start = time.perf_counter()
    try:
        await scheduler.dispatch(request)
        success = True
    except TimeoutError:
        success = False
    return (
        success,
        time.perf_counter() - start,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic tail-latency benchmark for deadline-aware "
            "hedged rollouts. This validates the control-plane "
            "mechanism, not GPU performance."
        )
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--primary-ms",
        type=float,
        default=40.0,
    )
    parser.add_argument(
        "--backup-ms",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--hedge-after-ms",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--deadline-ms",
        type=float,
        default=20.0,
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.trials <= 0:
        raise ValueError("trials must be positive")
    if min(
        args.primary_ms,
        args.backup_ms,
        args.hedge_after_ms,
        args.deadline_ms,
    ) <= 0:
        raise ValueError(
            "latency and deadline arguments must be positive"
        )

    primary_s = args.primary_ms / 1000.0
    backup_s = args.backup_ms / 1000.0
    hedge_s = args.hedge_after_ms / 1000.0
    deadline_s = args.deadline_ms / 1000.0

    baseline = [
        await one_trial(
            hedge_after_s=None,
            primary_delay_s=primary_s,
            backup_delay_s=backup_s,
            deadline_s=deadline_s,
        )
        for _ in range(args.trials)
    ]
    hedged = [
        await one_trial(
            hedge_after_s=hedge_s,
            primary_delay_s=primary_s,
            backup_delay_s=backup_s,
            deadline_s=deadline_s,
        )
        for _ in range(args.trials)
    ]

    baseline_latency = [
        elapsed
        for _, elapsed in baseline
    ]
    hedged_latency = [
        elapsed
        for _, elapsed in hedged
    ]
    baseline_successes = sum(
        success
        for success, _ in baseline
    )
    hedged_successes = sum(
        success
        for success, _ in hedged
    )

    metrics = {
        "baseline_success_rate": (
            baseline_successes / args.trials
        ),
        "hedged_success_rate": (
            hedged_successes / args.trials
        ),
        "baseline_latency_ms_p50": (
            1000.0 * median(baseline_latency)
        ),
        "baseline_latency_ms_p95": (
            1000.0
            * percentile(
                baseline_latency,
                0.95,
            )
        ),
        "hedged_latency_ms_p50": (
            1000.0 * median(hedged_latency)
        ),
        "hedged_latency_ms_p95": (
            1000.0
            * percentile(
                hedged_latency,
                0.95,
            )
        ),
        "synthetic_primary_ms": args.primary_ms,
        "synthetic_backup_ms": args.backup_ms,
    }
    report = BenchmarkReport(
        name="scheduler-slo-tail-latency",
        metrics=metrics,
        config={
            "trials": args.trials,
            "primary_ms": args.primary_ms,
            "backup_ms": args.backup_ms,
            "hedge_after_ms": args.hedge_after_ms,
            "deadline_ms": args.deadline_ms,
            "synthetic": "true",
        },
        git_sha=os.environ.get(
            "GITHUB_SHA",
            "unknown",
        ),
    )
    if args.output:
        Path(args.output).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        report.write_json(args.output)
    print(
        json.dumps(
            report.payload(),
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
