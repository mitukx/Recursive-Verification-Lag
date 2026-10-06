from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

from src.rvl_systems.benchmark_report import BenchmarkReport
from src.rvl_systems.vllm_metrics import (
    counter_delta,
    fetch_vllm_metrics,
    gpu_inventory,
    max_metric,
    parse_prometheus_samples,
)


@dataclass(frozen=True)
class RequestMeasurement:
    endpoint: str
    request_id: int
    latency_s: float
    ttft_s: float
    completion_tokens: int
    prompt_tokens: int
    tbt_s: float | None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    fraction = position - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def streaming_completion(
    endpoint: str,
    model: str,
    prompt: str,
    *,
    request_id: int,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> RequestMeasurement:
    url = endpoint.rstrip("/") + "/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "temperature": temperature,
        "seed": 20261006 + request_id,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    first_token_at: float | None = None
    completion_tokens = 0
    prompt_tokens = 0

    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if not body or body == "[DONE]":
                continue
            chunk = json.loads(body)
            choices = chunk.get("choices") or []
            if choices:
                text = choices[0].get("text") or ""
                if text and first_token_at is None:
                    first_token_at = time.perf_counter()
            usage = chunk.get("usage")
            if usage:
                completion_tokens = int(usage.get("completion_tokens") or 0)
                prompt_tokens = int(usage.get("prompt_tokens") or 0)

    end = time.perf_counter()
    if first_token_at is None:
        first_token_at = end
    latency_s = end - start
    ttft_s = first_token_at - start
    tbt_s = None
    if completion_tokens > 1:
        tbt_s = max(0.0, (latency_s - ttft_s) / (completion_tokens - 1))
    return RequestMeasurement(
        endpoint=endpoint,
        request_id=request_id,
        latency_s=latency_s,
        ttft_s=ttft_s,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        tbt_s=tbt_s,
    )


async def run_phase(
    endpoints: list[str],
    model: str,
    *,
    requests: int,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> tuple[list[RequestMeasurement], float]:
    semaphore = asyncio.Semaphore(concurrency)
    prompts = [
        "Explain why asynchronous batching improves language-model serving.",
        "Write a short Python function that computes Fibonacci numbers.",
        "Give three reasons tail latency matters in distributed inference.",
        "Explain the difference between prefill and decode in LLM serving.",
    ]

    async def one(index: int) -> RequestMeasurement:
        async with semaphore:
            endpoint = endpoints[index % len(endpoints)]
            return await asyncio.to_thread(
                streaming_completion,
                endpoint,
                model,
                prompts[index % len(prompts)],
                request_id=index,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
            )

    start = time.perf_counter()
    measurements = await asyncio.gather(*(one(i) for i in range(requests)))
    return measurements, time.perf_counter() - start


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Streaming vLLM/SGLang benchmark with TTFT/TBT and /metrics capture."
    )
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--output-dir", default="artifacts/vllm")
    args = parser.parse_args()

    levels = [int(item) for item in args.concurrency.split(",") if item.strip()]
    if not levels or any(level <= 0 for level in levels):
        raise ValueError("concurrency must contain positive integers")
    if args.requests <= 0 or args.max_tokens <= 0:
        raise ValueError("requests and max-tokens must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = gpu_inventory()
    (output_dir / "gpu-inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    phase_summaries = []
    for concurrency in levels:
        before_text = fetch_vllm_metrics(args.endpoint[0])
        before = parse_prometheus_samples(before_text)
        measurements, wall_s = await run_phase(
            args.endpoint,
            args.model,
            requests=args.requests,
            concurrency=concurrency,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
        )
        after_text = fetch_vllm_metrics(args.endpoint[0])
        after = parse_prometheus_samples(after_text)

        latencies = [m.latency_s for m in measurements]
        ttfts = [m.ttft_s for m in measurements]
        tbts = [m.tbt_s for m in measurements if m.tbt_s is not None]
        completion_tokens = sum(m.completion_tokens for m in measurements)
        prompt_tokens = sum(m.prompt_tokens for m in measurements)
        server_generated = counter_delta(
            before,
            after,
            "vllm:generation_tokens_total",
        )
        server_prompt = counter_delta(
            before,
            after,
            "vllm:prompt_tokens_total",
        )
        kv_cache = max_metric(after, "vllm:kv_cache_usage_perc")

        metrics: dict[str, float | int | str] = {
            "wall_s": wall_s,
            "requests": len(measurements),
            "requests_per_s": len(measurements) / max(wall_s, 1e-12),
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "tokens_per_s": completion_tokens / max(wall_s, 1e-12),
            "latency_ms_p50": 1000.0 * percentile(latencies, 0.50),
            "latency_ms_p95": 1000.0 * percentile(latencies, 0.95),
            "latency_ms_p99": 1000.0 * percentile(latencies, 0.99),
            "ttft_ms_p50": 1000.0 * percentile(ttfts, 0.50),
            "ttft_ms_p95": 1000.0 * percentile(ttfts, 0.95),
            "ttft_ms_p99": 1000.0 * percentile(ttfts, 0.99),
            "tbt_ms_mean": 1000.0 * fmean(tbts) if tbts else 0.0,
            "tbt_ms_p95": 1000.0 * percentile(tbts, 0.95) if tbts else 0.0,
            "server_generation_tokens_delta": server_generated,
            "server_prompt_tokens_delta": server_prompt,
        }
        if kv_cache is not None:
            metrics["vllm_kv_cache_usage_snapshot"] = kv_cache

        report = BenchmarkReport(
            name="vllm-streaming-serving",
            metrics=metrics,
            config={
                "endpoints": len(args.endpoint),
                "requests": args.requests,
                "concurrency": concurrency,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            },
            git_sha=os.environ.get("GITHUB_SHA", "unknown"),
            model=args.model,
        )
        report_path = output_dir / f"concurrency-{concurrency}.json"
        report.write_json(report_path)
        (output_dir / f"concurrency-{concurrency}.requests.json").write_text(
            json.dumps(
                [asdict(item) for item in measurements],
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (output_dir / f"concurrency-{concurrency}.metrics.prom").write_text(
            after_text,
            encoding="utf-8",
        )
        phase_summaries.append(report.payload())

    (output_dir / "summary.json").write_text(
        json.dumps(phase_summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(phase_summaries, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
