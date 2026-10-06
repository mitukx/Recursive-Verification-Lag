from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

from src.rvl_systems.backends import VLLMHTTPBackend
from src.rvl_systems.benchmark_report import BenchmarkReport
from src.rvl_systems.rollout import RolloutRequest
from src.rvl_systems.scheduler import LeastLoadedScheduler, WorkerSlot
from src.rvl_systems.telemetry import Telemetry


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark one or more OpenAI-compatible vLLM/SGLang endpoints."
    )
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--concurrency-per-worker", type=int, default=4)
    parser.add_argument("--queue-limit", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output", default="results/vllm_cluster_benchmark.json")
    args = parser.parse_args()

    telemetry = Telemetry()
    workers = [
        WorkerSlot(
            name=f"endpoint-{i}",
            backend=VLLMHTTPBackend(endpoint=endpoint, model=args.model),
            max_inflight=args.concurrency_per_worker,
        )
        for i, endpoint in enumerate(args.endpoint)
    ]
    scheduler = LeastLoadedScheduler(
        workers,
        queue_limit=args.queue_limit,
        telemetry=telemetry,
    )
    prompts = [
        "Return only the integer: 17 + 25",
        "Return only the integer: 9 * 7",
        "Write a Python expression equal to the sum of integers from 1 to 20.",
        "Return only the integer: 144 / 12",
    ]
    requests = [
        RolloutRequest(
            prompt_id=f"request-{i}",
            prompt=prompts[i % len(prompts)],
            samples=args.samples,
            temperature=args.temperature,
            seed=20261006 + i,
        )
        for i in range(args.requests)
    ]

    start = time.perf_counter()
    generations = await scheduler.run(requests)
    elapsed = time.perf_counter() - start
    tokens = sum(g.token_count for g in generations)
    metrics = {
        "wall_s": elapsed,
        "requests_per_s": args.requests / max(elapsed, 1e-12),
        "samples_per_s": len(generations) / max(elapsed, 1e-12),
        "tokens_per_s": tokens / max(elapsed, 1e-12),
        "generated_tokens": tokens,
        "samples": len(generations),
        **telemetry.snapshot(),
    }
    report = BenchmarkReport(
        name="vllm-cluster-rollout",
        metrics=metrics,
        config={
            "endpoints": len(args.endpoint),
            "requests": args.requests,
            "samples_per_request": args.samples,
            "concurrency_per_worker": args.concurrency_per_worker,
            "queue_limit": args.queue_limit,
            "temperature": args.temperature,
        },
        git_sha=os.environ.get("GITHUB_SHA", "unknown"),
        model=args.model,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    report.write_json(args.output)
    print(json.dumps(report.payload(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
