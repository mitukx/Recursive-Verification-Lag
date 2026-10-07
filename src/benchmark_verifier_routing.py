from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from src.rvl_systems.types import Generation
from src.rvl_systems.verifier import ExactMatchVerifier
from src.rvl_systems.verifier_rpc import (
    AdaptiveVerifierFleet,
    DistributedVerifierFleet,
    TCPVerifierClient,
    VerifierWorkerServer,
)


def pct(values, q):
    xs = sorted(values)
    if not xs:
        return 0.0
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


async def run_mode(mode: str, requests: int) -> dict:
    servers = [
        VerifierWorkerServer(
            ExactMatchVerifier({"p": "42"}, latency_s=0.020),
            worker_id="slow",
            capacity=1,
            service_time_hint_s=0.020,
        ),
        VerifierWorkerServer(
            ExactMatchVerifier({"p": "42"}, latency_s=0.004),
            worker_id="fast",
            capacity=2,
            service_time_hint_s=0.004,
        ),
    ]
    addrs = [await s.start() for s in servers]
    clients = [TCPVerifierClient(h, p, timeout_s=5.0) for h, p in addrs]
    try:
        if mode == "adaptive":
            fleet = AdaptiveVerifierFleet(
                clients,
                expected_verifier_version=0,
                request_deadline_s=1.0,
                default_service_time_s=0.01,
            )
            await fleet.refresh_health()
        else:
            fleet = DistributedVerifierFleet(clients, expected_verifier_version=0)
            await fleet.refresh_health()

        async def one(i):
            g = Generation("p", "6*7?", "42", -0.1, 1, 0.0, {"request_index": i})
            started = time.perf_counter()
            out = await fleet.verify(g)
            elapsed = time.perf_counter() - started
            return {
                "request_index": i,
                "latency_s": elapsed,
                "worker_id": out.metadata.get("rpc_worker_id"),
            }

        started = time.perf_counter()
        rows = await asyncio.gather(*(one(i) for i in range(requests)))
        wall = time.perf_counter() - started
        lat = [r["latency_s"] for r in rows]
        counts = {}
        for r in rows:
            counts[r["worker_id"]] = counts.get(r["worker_id"], 0) + 1
        result = {
            "mode": mode,
            "requests": requests,
            "wall_s": wall,
            "throughput_req_s": requests / wall,
            "latency_p50_s": pct(lat, 0.50),
            "latency_p95_s": pct(lat, 0.95),
            "latency_max_s": max(lat),
            "worker_counts": counts,
            "rows": rows,
        }
        if mode == "adaptive":
            result["routing_snapshot"] = fleet.routing_snapshot()
        return result
    finally:
        for s in servers:
            await s.close()


async def main_async(requests: int, output: Path) -> None:
    baseline = await run_mode("round_robin", requests)
    adaptive = await run_mode("adaptive", requests)
    summary = {
        "status": "control_plane_cpu_rpc_benchmark",
        "requests_per_mode": requests,
        "baseline": {k: v for k, v in baseline.items() if k != "rows"},
        "adaptive": {k: v for k, v in adaptive.items() if k != "rows"},
        "comparison": {
            "p95_speedup": baseline["latency_p95_s"] / adaptive["latency_p95_s"],
            "throughput_speedup": adaptive["throughput_req_s"] / baseline["throughput_req_s"],
        },
        "claim_boundary": (
            "Loopback CPU RPC benchmark with deliberately heterogeneous verifier service times. "
            "Useful for scheduler correctness/tail-latency evidence, not a GPU or multi-host claim."
        ),
        "raw": {"baseline": baseline["rows"], "adaptive": adaptive["rows"]},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["comparison"], indent=2, sort_keys=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--requests", type=int, default=48)
    p.add_argument("--output", type=Path, default=Path("artifacts/verifier-routing.json"))
    args = p.parse_args()
    if args.requests < 4:
        raise ValueError("requests must be >=4")
    asyncio.run(main_async(args.requests, args.output))


if __name__ == "__main__":
    main()
