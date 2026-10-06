"""Paired real-endpoint benchmark for verl workload-aware routing.

Runs identical request traces against multiple OpenAI-compatible vLLM/SGLang
endpoints using least-inflight vs predicted-outstanding-work placement.

This file does not fabricate endpoints or results. It is an execution harness
for the real-GPU evidence gate.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt: str
    prompt_tokens: int
    max_tokens: int
    seed: int
    session_id: str | None = None

    @property
    def predicted_work(self) -> int:
        return self.prompt_tokens + self.max_tokens


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def load_trace(path: Path) -> list[Request]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        rows.append(Request(
            request_id=str(raw["request_id"]),
            prompt=str(raw["prompt"]),
            prompt_tokens=int(raw["prompt_tokens"]),
            max_tokens=int(raw["max_tokens"]),
            seed=int(raw["seed"]),
            session_id=(
                None if raw.get("session_id") is None
                else str(raw["session_id"])
            ),
        ))
    if not rows:
        raise ValueError("trace is empty")
    if len({r.request_id for r in rows}) != len(rows):
        raise ValueError("request_id values must be unique")
    return rows


class Router:
    def __init__(self, endpoints: list[str], mode: str):
        if mode not in {"least_inflight", "workload_aware"}:
            raise ValueError(mode)
        self.endpoints = list(endpoints)
        self.mode = mode
        self.inflight = {e: 0 for e in endpoints}
        self.work = {e: 0 for e in endpoints}
        self.sticky: dict[str, str] = {}
        self.lock = asyncio.Lock()

    async def acquire(self, request: Request) -> str:
        async with self.lock:
            if request.session_id is not None:
                endpoint = self.sticky.get(request.session_id)
                if endpoint in self.inflight:
                    self._add(endpoint, request)
                    return endpoint

            if self.mode == "least_inflight":
                best = min(self.inflight.values())
                candidates = [e for e, n in self.inflight.items() if n == best]
            else:
                best = min(self.work.values())
                candidates = [e for e, n in self.work.items() if n == best]
                min_inflight = min(self.inflight[e] for e in candidates)
                candidates = [e for e in candidates if self.inflight[e] == min_inflight]

            endpoint = candidates[0]
            if request.session_id is not None:
                self.sticky[request.session_id] = endpoint
            self._add(endpoint, request)
            return endpoint

    def _add(self, endpoint: str, request: Request) -> None:
        self.inflight[endpoint] += 1
        self.work[endpoint] += request.predicted_work

    async def release(self, endpoint: str, request: Request) -> None:
        async with self.lock:
            self.inflight[endpoint] = max(0, self.inflight[endpoint] - 1)
            self.work[endpoint] = max(0, self.work[endpoint] - request.predicted_work)


def _post_completion(
    endpoint: str,
    model: str,
    request: Request,
    temperature: float,
    timeout_s: float,
) -> dict[str, Any]:
    payload = json.dumps({
        "model": model,
        "prompt": request.prompt,
        "n": 1,
        "temperature": temperature,
        "seed": request.seed,
        "max_tokens": request.max_tokens,
        "logprobs": 1,
    }).encode("utf-8")
    http_request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(http_request, timeout=timeout_s) as response:
        body = json.loads(response.read().decode("utf-8"))
    latency = time.perf_counter() - started
    choices = body.get("choices") or []
    if len(choices) != 1:
        raise RuntimeError(f"expected one completion, got {len(choices)}")
    usage = body.get("usage") or {}
    token_logprobs = (choices[0].get("logprobs") or {}).get("token_logprobs") or []
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = len([x for x in token_logprobs if x is not None])
    return {
        "latency_s": latency,
        "text": str(choices[0].get("text", "")),
        "completion_tokens": int(completion_tokens or 0),
    }


async def run_arm(
    *,
    arm: str,
    endpoints: list[str],
    model: str,
    trace: list[Request],
    concurrency: int,
    temperature: float,
    timeout_s: float,
) -> dict[str, Any]:
    router = Router(endpoints, arm)
    sem = asyncio.Semaphore(concurrency)
    records: list[dict[str, Any]] = []
    per_endpoint = {
        e: {"requests": 0, "predicted_work": 0, "completion_tokens": 0}
        for e in endpoints
    }

    async def one(request: Request) -> None:
        async with sem:
            endpoint = await router.acquire(request)
            started = time.perf_counter()
            try:
                result = await asyncio.to_thread(
                    _post_completion,
                    endpoint,
                    model,
                    request,
                    temperature,
                    timeout_s,
                )
                wall = time.perf_counter() - started
                record = {
                    "request_id": request.request_id,
                    "session_id": request.session_id,
                    "endpoint": endpoint,
                    "prompt_tokens": request.prompt_tokens,
                    "max_tokens": request.max_tokens,
                    "predicted_work": request.predicted_work,
                    "latency_s": wall,
                    "server_http_latency_s": result["latency_s"],
                    "completion_tokens": result["completion_tokens"],
                    "output_sha256": hashlib.sha256(
                        result["text"].encode("utf-8")
                    ).hexdigest(),
                }
                records.append(record)
                per_endpoint[endpoint]["requests"] += 1
                per_endpoint[endpoint]["predicted_work"] += request.predicted_work
                per_endpoint[endpoint]["completion_tokens"] += result["completion_tokens"]
            finally:
                await router.release(endpoint, request)

    started = time.perf_counter()
    await asyncio.gather(*(one(r) for r in trace))
    wall_s = time.perf_counter() - started
    latencies = [r["latency_s"] for r in records]
    total_completion_tokens = sum(r["completion_tokens"] for r in records)
    work_values = [v["predicted_work"] for v in per_endpoint.values()]
    return {
        "arm": arm,
        "wall_s": wall_s,
        "requests_per_s": len(trace) / max(wall_s, 1e-12),
        "completion_tokens_per_s": total_completion_tokens / max(wall_s, 1e-12),
        "latency_mean_s": statistics.fmean(latencies),
        "latency_p50_s": percentile(latencies, .50),
        "latency_p95_s": percentile(latencies, .95),
        "latency_p99_s": percentile(latencies, .99),
        "predicted_work_imbalance_ratio": (
            max(work_values) / max(1, min(work_values))
        ),
        "per_endpoint": per_endpoint,
        "records": sorted(records, key=lambda r: r["request_id"]),
    }


def _output_map(run: dict[str, Any]) -> dict[str, str]:
    return {r["request_id"]: r["output_sha256"] for r in run["records"]}


async def warmup(endpoint: str, model: str, timeout_s: float) -> None:
    req = Request("warmup", "Return only: 1", 4, 1, 1)
    await asyncio.to_thread(_post_completion, endpoint, model, req, 0.0, timeout_s)


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", action="append", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout-s", type=float, default=300)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if len(args.endpoint) < 2:
        raise ValueError("at least two endpoints are required")
    if args.concurrency <= 0 or args.repeats <= 0:
        raise ValueError("concurrency and repeats must be positive")

    trace = load_trace(args.trace)
    for endpoint in args.endpoint:
        await warmup(endpoint, args.model, args.timeout_s)

    runs = []
    for repeat in range(args.repeats):
        order = (
            ["least_inflight", "workload_aware"]
            if repeat % 2 == 0
            else ["workload_aware", "least_inflight"]
        )
        paired = {}
        for arm in order:
            paired[arm] = await run_arm(
                arm=arm,
                endpoints=args.endpoint,
                model=args.model,
                trace=trace,
                concurrency=args.concurrency,
                temperature=args.temperature,
                timeout_s=args.timeout_s,
            )
        least = paired["least_inflight"]
        aware = paired["workload_aware"]
        paired["output_parity"] = _output_map(least) == _output_map(aware)
        paired["relative"] = {
            "requests_per_s_ratio": (
                aware["requests_per_s"] / max(least["requests_per_s"], 1e-12)
            ),
            "p95_latency_ratio": (
                aware["latency_p95_s"] / max(least["latency_p95_s"], 1e-12)
            ),
            "p99_latency_ratio": (
                aware["latency_p99_s"] / max(least["latency_p99_s"], 1e-12)
            ),
            "work_imbalance_ratio": (
                aware["predicted_work_imbalance_ratio"]
                / max(least["predicted_work_imbalance_ratio"], 1e-12)
            ),
        }
        runs.append({"repeat": repeat, "order": order, **paired})

    summary = {}
    for metric in (
        "requests_per_s_ratio",
        "p95_latency_ratio",
        "p99_latency_ratio",
        "work_imbalance_ratio",
    ):
        values = [r["relative"][metric] for r in runs]
        summary[metric] = {
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
        }

    report = {
        "schema_version": 1,
        "model": args.model,
        "endpoints": args.endpoint,
        "trace": str(args.trace),
        "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        "requests": len(trace),
        "concurrency": args.concurrency,
        "temperature": args.temperature,
        "repeats": args.repeats,
        "runs": runs,
        "summary": summary,
        "claim_boundary": (
            "Real endpoint measurements only when this file is executed against "
            "actual GPU vLLM/SGLang servers. This committed harness contains no "
            "performance result by itself."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
