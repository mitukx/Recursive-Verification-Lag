from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path

from src.rvl_systems.backends import VLLMHTTPBackend
from src.rvl_systems.benchmark_report import BenchmarkReport
from src.rvl_systems.rollout import RolloutRequest
from src.rvl_systems.scheduler import LeastLoadedScheduler, WorkerSlot
from src.rvl_systems.telemetry import Telemetry


def _requests(count: int, samples: int, temperature: float) -> list[RolloutRequest]:
    prompts = [
        "Return only the integer: 17 + 25",
        "Return only the integer: 9 * 7",
        "Explain in one sentence why batching improves throughput.",
        "Write a Python expression equal to the sum of integers from 1 to 20.",
    ]
    return [
        RolloutRequest(
            prompt_id=f"sync-async-{i}",
            prompt=prompts[i % len(prompts)],
            samples=samples,
            temperature=temperature,
            seed=20261006 + i,
        )
        for i in range(count)
    ]


async def _serial(backends: list[VLLMHTTPBackend], requests: list[RolloutRequest]):
    generations = []
    start = time.perf_counter()
    for index, request in enumerate(requests):
        backend = backends[index % len(backends)]
        generations.extend(await backend.generate(
            request.prompt_id,
            request.prompt,
            n=request.samples,
            temperature=request.temperature,
            seed=request.seed,
        ))
    return generations, time.perf_counter() - start


async def _async(backends: list[VLLMHTTPBackend], requests: list[RolloutRequest], concurrency: int):
    telemetry = Telemetry()
    workers = [
        WorkerSlot(f"endpoint-{i}", backend, max_inflight=concurrency)
        for i, backend in enumerate(backends)
    ]
    scheduler = LeastLoadedScheduler(
        workers,
        queue_limit=max(len(requests), concurrency * len(workers) * 2),
        telemetry=telemetry,
    )
    start = time.perf_counter()
    generations = await scheduler.run(requests)
    return generations, time.perf_counter() - start, telemetry.snapshot()


def _throughput(generations, wall_s: float, request_count: int) -> dict[str, float | int]:
    if wall_s <= 0:
        raise ValueError("wall time must be positive")
    tokens = sum(g.token_count for g in generations)
    return {
        "wall_s": wall_s,
        "requests_per_s": request_count / wall_s,
        "samples_per_s": len(generations) / wall_s,
        "tokens_per_s": tokens / wall_s,
        "generated_tokens": tokens,
        "samples": len(generations),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired serial-vs-asynchronous rollout benchmark against the same vLLM/SGLang endpoints."
    )
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--concurrency-per-worker", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.requests <= 0 or args.samples <= 0 or args.concurrency_per_worker <= 0:
        raise ValueError("request/sample/concurrency counts must be positive")

    backends = [
        VLLMHTTPBackend(endpoint=e, model=args.model, max_tokens=args.max_tokens)
        for e in args.endpoint
    ]
    requests = _requests(args.requests, args.samples, args.temperature)

    # Warm both code paths without including first-use server effects.
    await backends[0].generate("warmup", "Return only: 1", n=1, temperature=0.0, seed=1)

    serial_generations, serial_wall = await _serial(backends, requests)
    async_generations, async_wall, telemetry = await _async(
        backends, requests, args.concurrency_per_worker
    )
    serial = _throughput(serial_generations, serial_wall, args.requests)
    asynchronous = _throughput(async_generations, async_wall, args.requests)

    speedup = asynchronous["tokens_per_s"] / max(float(serial["tokens_per_s"]), 1e-12)
    request_speedup = asynchronous["requests_per_s"] / max(float(serial["requests_per_s"]), 1e-12)
    if not math.isfinite(speedup) or speedup <= 0:
        raise RuntimeError("invalid async token-throughput speedup")

    metrics = {
        "serial_tokens_per_s": serial["tokens_per_s"],
        "async_tokens_per_s": asynchronous["tokens_per_s"],
        "token_throughput_speedup": speedup,
        "serial_requests_per_s": serial["requests_per_s"],
        "async_requests_per_s": asynchronous["requests_per_s"],
        "request_throughput_speedup": request_speedup,
        "serial_wall_s": serial_wall,
        "async_wall_s": async_wall,
        "serial_generated_tokens": serial["generated_tokens"],
        "async_generated_tokens": asynchronous["generated_tokens"],
        **{f"async_{k}": v for k, v in telemetry.items()},
    }
    report = BenchmarkReport(
        name="vllm-sync-vs-async",
        metrics=metrics,
        config={
            "endpoints": len(args.endpoint),
            "requests": args.requests,
            "samples": args.samples,
            "concurrency_per_worker": args.concurrency_per_worker,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        git_sha=os.environ.get("GITHUB_SHA", "unknown"),
        model=args.model,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report.write_json(path)
    print(json.dumps(report.payload(), indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
