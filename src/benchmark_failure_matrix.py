from __future__ import annotations

import argparse
import asyncio
import json
import os

from src.rvl_systems.backends import ToyTabularBackend
from src.rvl_systems.benchmark_report import BenchmarkReport
from src.rvl_systems.rollout import RolloutRequest
from src.rvl_systems.scheduler import LeastLoadedScheduler, WorkerSlot
from src.rvl_systems.telemetry import Telemetry
from src.rvl_systems.worker_health import WorkerHealth


class FaultBackend:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    async def generate(self, *args, **kwargs):
        if self.mode == "error":
            raise ConnectionError("injected connection failure")
        if self.mode == "oom":
            raise RuntimeError("injected out-of-memory failure")
        if self.mode == "timeout":
            await asyncio.sleep(0.050)
            return []
        raise ValueError(f"unknown mode: {self.mode}")


async def recovered_scenario(mode: str) -> bool:
    telemetry = Telemetry()
    scheduler = LeastLoadedScheduler(
        [
            WorkerSlot("faulty", FaultBackend(mode)),
            WorkerSlot("healthy", ToyTabularBackend(base_latency_s=0.001)),
        ],
        request_timeout_s=0.005 if mode == "timeout" else 1.0,
        max_attempts_per_request=2,
        health=WorkerHealth(failure_threshold=1),
        telemetry=telemetry,
    )
    result = await scheduler.dispatch(
        RolloutRequest(f"recover-{mode}", "x", samples=1)
    )
    snapshot = telemetry.snapshot()
    return (
        len(result) == 1
        and snapshot.get("scheduler.failover_successes", 0.0) >= 1
    )


async def expected_total_failure() -> bool:
    scheduler = LeastLoadedScheduler(
        [
            WorkerSlot("a", FaultBackend("error")),
            WorkerSlot("b", FaultBackend("oom")),
        ],
        max_attempts_per_request=2,
        health=WorkerHealth(failure_threshold=1),
    )
    try:
        await scheduler.dispatch(RolloutRequest("all-fail", "x", samples=1))
    except Exception:
        return True
    return False


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/failure-matrix.json")
    args = parser.parse_args()

    recovered = {}
    for mode in ("error", "oom", "timeout"):
        recovered[mode] = await recovered_scenario(mode)
    terminal_failure = await expected_total_failure()

    metrics = {
        "scenario_count": 4,
        "recovered_error": int(recovered["error"]),
        "recovered_oom": int(recovered["oom"]),
        "recovered_timeout": int(recovered["timeout"]),
        "expected_total_failure_observed": int(terminal_failure),
        "unexpected_failures": int(
            not all(recovered.values()) or not terminal_failure
        ),
    }
    if metrics["unexpected_failures"]:
        raise RuntimeError(
            f"failure matrix did not satisfy invariants: {metrics}"
        )

    report = BenchmarkReport(
        name="scheduler-failure-matrix",
        metrics=metrics,
        config={
            "faults": "connection,oom,timeout,total-failure",
            "synthetic": "true",
        },
        git_sha=os.environ.get("GITHUB_SHA", "unknown"),
    )
    report.write_json(args.output)
    print(json.dumps(report.payload(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
