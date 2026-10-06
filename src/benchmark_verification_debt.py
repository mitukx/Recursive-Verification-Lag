"""Deterministic control-plane benchmark for verification-aware async RL.

This is a queueing/contract simulation, not a GPU throughput benchmark.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from src.rvl_systems.lab.verification_debt import (
    VerificationDebtConfig,
    VerificationDebtController,
    VerificationDebtSignals,
)


@dataclass(frozen=True)
class SimulationConfig:
    groups: int = 64
    generation_s: float = 0.01
    verification_s: float = 0.04
    learner_s: float = 0.01
    capacity: int = 32
    max_policy_lag: int = 16
    tick_s: float = 0.001
    timeout_s: float = 60.0

    def __post_init__(self):
        if min(self.groups,self.capacity,self.max_policy_lag+1) <= 0:
            raise ValueError("invalid integer simulation config")
        if min(self.generation_s,self.verification_s,self.learner_s,self.tick_s,self.timeout_s) <= 0:
            raise ValueError("invalid timing simulation config")


@dataclass
class Item:
    generated_at: float
    behavior_version: int
    verified_at: float | None = None


def _percentile(values, q):
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs)-1)*q
    lo = int(pos)
    hi = min(lo+1,len(xs)-1)
    return xs[lo] + (xs[hi]-xs[lo])*(pos-lo)


def simulate(arm: str, cfg: SimulationConfig, debt_cfg: VerificationDebtConfig | None = None):
    if arm not in {"sync","async_policy_only","verification_aware"}:
        raise ValueError("unknown arm")
    controller = VerificationDebtController(debt_cfg or VerificationDebtConfig())
    now = 0.0
    generated = 0
    consumed = 0
    policy_version = 0
    pending: list[Item] = []
    ready: list[Item] = []
    stale = 0
    verification_busy_until = 0.0
    learner_busy_until = 0.0
    next_generation_at = 0.0
    verification_current: Item | None = None
    learner_current: Item | None = None
    queue_delays = []
    policy_lags = []
    debt_scores = []
    peak_backlog = 0
    throttle_events = 0
    learner_idle_ticks = 0
    generated_times = []

    while consumed + stale < cfg.groups:
        if now > cfg.timeout_s:
            raise RuntimeError("simulation timed out")

        if verification_current is not None and now + 1e-12 >= verification_busy_until:
            verification_current.verified_at = now
            queue_delays.append(now-verification_current.generated_at)
            ready.append(verification_current)
            verification_current = None

        if learner_current is not None and now + 1e-12 >= learner_busy_until:
            lag = max(0,policy_version-learner_current.behavior_version)
            policy_lags.append(lag)
            if lag > cfg.max_policy_lag:
                stale += 1
            else:
                consumed += 1
                policy_version += 1
            learner_current = None

        if verification_current is None and pending:
            verification_current = pending.pop(0)
            verification_busy_until = now + cfg.verification_s

        if learner_current is None and ready:
            learner_current = ready.pop(0)
            learner_busy_until = now + cfg.learner_s
        elif learner_current is None and not ready and (pending or verification_current):
            learner_idle_ticks += 1

        active = len(pending) + len(ready)
        active += int(verification_current is not None) + int(learner_current is not None)
        backlog = len(pending) + int(verification_current is not None)
        peak_backlog = max(peak_backlog,backlog)

        max_policy_lag = max(
            [max(0,policy_version-x.behavior_version)
             for x in pending + ready
             + ([verification_current] if verification_current else [])
             + ([learner_current] if learner_current else [])]
            or [0]
        )
        oldest_age = max(
            [now-x.generated_at for x in pending + ([verification_current] if verification_current else [])]
            or [0.0]
        )
        assessment = controller.assess(VerificationDebtSignals(
            pending=len(pending),
            verifying=int(verification_current is not None),
            stale_rewards=0,
            max_policy_lag=max_policy_lag,
            max_verifier_lag=0,
            oldest_unverified_age_s=max(0.0,oldest_age),
        ))
        debt_scores.append(assessment.score)

        can_generate = (
            generated < cfg.groups
            and active < cfg.capacity
            and now + 1e-12 >= next_generation_at
        )
        if arm == "sync":
            can_generate = can_generate and backlog == 0
        elif arm == "verification_aware" and assessment.action != "admit_generation":
            if can_generate:
                throttle_events += 1
            can_generate = False

        if can_generate:
            pending.append(Item(now,policy_version))
            generated += 1
            generated_times.append(now)
            next_generation_at = now + cfg.generation_s

        now += cfg.tick_s

    elapsed = now
    return {
        "arm":arm,
        "config":asdict(cfg),
        "debt_config":asdict(controller.config),
        "generated":generated,
        "consumed":consumed,
        "stale":stale,
        "elapsed_s":elapsed,
        "groups_per_s":consumed/elapsed if elapsed else 0.0,
        "peak_verification_backlog":peak_backlog,
        "mean_queue_to_verified_s":sum(queue_delays)/len(queue_delays) if queue_delays else 0.0,
        "p95_queue_to_verified_s":_percentile(queue_delays,.95),
        "max_policy_lag":max(policy_lags or [0]),
        "mean_policy_lag":sum(policy_lags)/len(policy_lags) if policy_lags else 0.0,
        "max_verification_debt":max(debt_scores or [0.0]),
        "throttle_events":throttle_events,
        "learner_idle_fraction":learner_idle_ticks/max(1,math.ceil(elapsed/cfg.tick_s)),
        "last_generation_s":max(generated_times or [0.0]),
        "claim_boundary":"Deterministic queue/control simulation only; no GPU, model-quality, or wall-clock systems claim.",
    }


def benchmark(cfg: SimulationConfig, debt_cfg: VerificationDebtConfig | None = None):
    runs = {arm:simulate(arm,cfg,debt_cfg) for arm in (
        "sync","async_policy_only","verification_aware"
    )}
    return {
        "config":asdict(cfg),
        "runs":runs,
        "checks":{
            "aware_backlog_bounded_vs_async":
                runs["verification_aware"]["peak_verification_backlog"]
                < runs["async_policy_only"]["peak_verification_backlog"],
            "sync_backlog_at_most_one":
                runs["sync"]["peak_verification_backlog"] <= 1,
            "all_groups_accounted":
                all(r["consumed"]+r["stale"]==cfg.groups for r in runs.values()),
        },
        "claim_boundary":"Synthetic queueing evidence; real serving/learner measurements are required by Issue #66.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("artifacts/verification-debt-benchmark.json"))
    p.add_argument("--groups",type=int,default=64)
    p.add_argument("--generation-ms",type=float,default=10)
    p.add_argument("--verification-ms",type=float,default=40)
    p.add_argument("--learner-ms",type=float,default=10)
    p.add_argument("--capacity",type=int,default=32)
    args = p.parse_args()
    cfg = SimulationConfig(
        groups=args.groups,
        generation_s=args.generation_ms/1000,
        verification_s=args.verification_ms/1000,
        learner_s=args.learner_ms/1000,
        capacity=args.capacity,
    )
    report = benchmark(cfg)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
