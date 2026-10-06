"""Deterministic trace benchmark for the verl workload-aware router prototype.

This benchmarks scheduling policy only. The service-time model is synthetic and
must not be reported as a GPU serving speedup.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from upstream.verl.workload_aware_router_prototype import (
    WorkloadAwareRequestLoadBalancer,
)


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt_tokens: int
    decode_tokens: int

    @property
    def work(self) -> int:
        return self.prompt_tokens + self.decode_tokens


class LeastInflight:
    def __init__(self, servers):
        self.counts = {s: 0 for s in servers}

    def acquire(self, req):
        m = min(self.counts.values())
        sid = next(s for s, v in self.counts.items() if v == m)
        self.counts[sid] += 1
        return sid

    def release(self, sid, req):
        self.counts[sid] -= 1


class WorkAware:
    def __init__(self, servers):
        self.lb = WorkloadAwareRequestLoadBalancer(
            {s: s for s in servers},
            default_decode_tokens=0,
        )

    def acquire(self, req):
        sid, _ = self.lb.acquire_server(
            req.request_id,
            prompt_tokens=req.prompt_tokens,
            decode_budget=req.decode_tokens,
        )
        return sid

    def release(self, sid, req):
        self.lb.release_server(sid, request_id=req.request_id)


def percentile(values, q):
    xs = sorted(values)
    if not xs:
        return 0.0
    x = (len(xs) - 1) * q
    lo = math.floor(x)
    hi = math.ceil(x)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - x) + xs[hi] * (x - lo)


def make_trace(n, seed):
    rng = random.Random(seed)
    trace = []
    for i in range(n):
        # Heterogeneous agent/RL-like traffic: most requests are moderate,
        # with a long-tail of large prompt/response budgets.
        if rng.random() < 0.15:
            prompt = rng.randint(8_000, 32_000)
            decode = rng.randint(1_024, 4_096)
        else:
            prompt = rng.randint(128, 2_048)
            decode = rng.randint(64, 768)
        trace.append(Request(f"r{i}", prompt, decode))
    return trace


def simulate(policy, trace, servers, token_service_s):
    completions = []
    assigned_work = {sid: 0 for sid in servers}

    # All requests are available together, matching a rollout burst. The router
    # decides placement; each server processes its assigned requests FIFO.
    assigned = {sid: [] for sid in servers}
    for req in trace:
        sid = policy.acquire(req)
        assigned[sid].append(req)
        assigned_work[sid] += req.work

    for sid, rows in assigned.items():
        t = 0.0
        for req in rows:
            t += req.work * token_service_s
            completions.append(t)
            policy.release(sid, req)

    return {
        "makespan_s": max(completions or [0.0]),
        "p50_completion_s": percentile(completions, .50),
        "p95_completion_s": percentile(completions, .95),
        "p99_completion_s": percentile(completions, .99),
        "work_per_server": assigned_work,
        "work_imbalance_ratio": (
            max(assigned_work.values()) / max(1, min(assigned_work.values()))
        ),
    }


def run(n=512, servers=8, seed=17, token_service_us=10.0):
    trace = make_trace(n, seed)
    ids = [f"s{i}" for i in range(servers)]
    least = simulate(
        LeastInflight(ids), trace, ids, token_service_us / 1_000_000
    )
    aware = simulate(
        WorkAware(ids), trace, ids, token_service_us / 1_000_000
    )
    return {
        "trace": {
            "requests": n,
            "servers": servers,
            "seed": seed,
            "token_service_us": token_service_us,
        },
        "least_inflight": least,
        "workload_aware": aware,
        "relative": {
            "makespan_reduction": (
                (least["makespan_s"] - aware["makespan_s"])
                / least["makespan_s"]
                if least["makespan_s"] else 0.0
            ),
            "p95_reduction": (
                (least["p95_completion_s"] - aware["p95_completion_s"])
                / least["p95_completion_s"]
                if least["p95_completion_s"] else 0.0
            ),
        },
        "claim_boundary": (
            "Synthetic service-time trace benchmark only. "
            "No GPU serving performance claim."
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path)
    p.add_argument("--requests", type=int, default=512)
    p.add_argument("--servers", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    report = run(args.requests, args.servers, args.seed)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
