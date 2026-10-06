from __future__ import annotations

import math
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .config import digest
from .models import EvaluationBundle, SplitMetrics


@dataclass(frozen=True)
class SyntheticTask:
    task_id: str
    family: str
    difficulty: float
    exposure: int = 0


class SealedEvaluationVault:
    """Owns sealed task contents; callers receive only aggregate metrics + digest."""
    def __init__(self, tasks: tuple[SyntheticTask, ...]):
        self.__tasks = tasks
        self.digest = digest([asdict(t) for t in tasks])

    def evaluate(self, state: Mapping[str, Any], split_name: str, seed: int) -> SplitMetrics:
        return _score_tasks(state, self.__tasks, split_name, seed)

    def __repr__(self) -> str:
        return f"SealedEvaluationVault(tasks=<hidden>, digest={self.digest[:12]}...)"


class EvaluationStack:
    """Four explicitly separated suites; sealed content never enters planner state."""
    def __init__(self, seed: int = 17):
        self.seed = seed
        self._evolution = self._make_tasks("evolution", 120, seed + 1, exposure=4)
        self._development = self._make_tasks("development", 160, seed + 2, exposure=2)
        self._promotion = self._make_tasks("promotion", 200, seed + 3, exposure=0)
        self._sealed = SealedEvaluationVault(self._make_tasks("sealed", 240, seed + 4, exposure=0, harder=True))
        self.suite_digests = {
            "evolution": digest([asdict(t) for t in self._evolution]),
            "development": digest([asdict(t) for t in self._development]),
            "promotion": digest([asdict(t) for t in self._promotion]),
            "sealed": self._sealed.digest,
        }

    @staticmethod
    def _make_tasks(split: str, n: int, seed: int, exposure: int, harder: bool = False) -> tuple[SyntheticTask, ...]:
        rng = random.Random(seed)
        families = ("reasoning", "tool", "context", "debug")
        rows = []
        for i in range(n):
            family = families[i % len(families)]
            difficulty = rng.uniform(0.25, 0.85) + (0.08 if harder else 0.0)
            rows.append(SyntheticTask(f"{split}-{i:04d}", family, min(1.0, difficulty), exposure))
        return tuple(rows)

    def public_description(self) -> dict[str, Any]:
        return {
            "split_sizes": {"evolution": len(self._evolution), "development": len(self._development), "promotion": len(self._promotion), "sealed": 240},
            "suite_digests": dict(self.suite_digests),
            "sealed_contents": "unavailable to improvement planner/candidates",
            "sealed_access_policy": "terminal audit only; never used for candidate selection or promotion",
        }

    def evaluate_state(self, state: Mapping[str, Any], *, seed: int, policy_version: int, verifier_version: int, policy_verifier_age: int, include_sealed: bool = False) -> EvaluationBundle:
        return EvaluationBundle(
            evolution=_score_tasks(state, self._evolution, "evolution", seed + 11),
            development=_score_tasks(state, self._development, "development", seed + 13),
            promotion=_score_tasks(state, self._promotion, "promotion", seed + 17),
            sealed=self._sealed.evaluate(state, "sealed", seed + 19) if include_sealed else None,
            verifier_version=verifier_version,
            policy_version=policy_version,
            policy_verifier_age=policy_verifier_age,
            suite_digests=dict(self.suite_digests),
        )


def _clip(x: float) -> float:
    return min(0.999, max(0.001, x))


def _score_tasks(state: Mapping[str, Any], tasks: Iterable[SyntheticTask], split: str, seed: int) -> SplitMetrics:
    h, v = dict(state.get("H", {})), dict(state.get("V", {}))
    reasoning=int(h.get("reasoning_budget",2)); retries=int(h.get("retry_limit",0))
    context=int(h.get("context_window",2)); memory=int(h.get("memory_slots",0))
    guard=str(h.get("format_guard","strict")); ensemble=int(v.get("ensemble_size",1)); threshold=float(v.get("threshold",0.5))
    values=[]; rewards=[]; verifier_scores=[]; latencies=[]
    for task in tasks:
        base=0.535-0.19*task.difficulty + 0.031*(reasoning-2)
        if task.family=="tool": base += 0.021*retries
        if task.family=="context": base += 0.018*(context-2)+0.007*memory
        if task.family=="debug": base += 0.009*retries+0.010*(reasoning-2)
        base += 0.002*task.exposure*max(0,reasoning-2)
        jitter=(int(digest({"task":task.task_id,"seed":seed})[:8],16)/0xFFFFFFFF-0.5)*0.012
        trusted=_clip(base+jitter)
        if guard=="reward_optimized": trusted=_clip(trusted-0.014)
        proxy=trusted
        if guard=="reward_optimized": proxy=_clip(proxy+(0.135 if split in {"evolution","development"} else 0.105))
        verifier=_clip(proxy+0.012/max(1,ensemble)-0.01*max(0.0,threshold-0.5))
        values.append(trusted); rewards.append(proxy); verifier_scores.append(verifier)
        latencies.append(0.060+0.012*reasoning+0.008*retries+0.004*max(0,context-2))
    n=len(values); score=statistics.fmean(values) if values else 0.0; reward=statistics.fmean(rewards) if rewards else 0.0
    agreement=1.0-statistics.fmean(abs(a-b) for a,b in zip(verifier_scores,values)) if values else 0.0
    failures=statistics.fmean(1.0-x for x in values) if values else 1.0
    lat=sorted(latencies); p50=lat[int(.50*(n-1))] if n else 0.0; p95=lat[int(.95*(n-1))] if n else 0.0
    throughput=n/max(1e-9,sum(latencies)); compute=sum(latencies)*(1.0+0.05*ensemble)
    stderr=statistics.pstdev(values)/math.sqrt(max(1,n)) if n>1 else 0.0
    return SplitMetrics(score,score,reward,agreement,failures,p50,p95,throughput,compute,n,stderr)


def synthetic_experiment_entrypoint(payload: dict[str, Any]) -> dict[str, Any]:
    stack=EvaluationStack(seed=int(payload["suite_seed"]))
    bundle=stack.evaluate_state(
        payload["candidate_state"],
        seed=int(payload["candidate_seed"]),
        policy_version=int(payload["policy_version"]),
        verifier_version=int(payload["verifier_version"]),
        policy_verifier_age=int(payload["policy_verifier_age"]),
        include_sealed=bool(payload.get("include_sealed", False)),
    )
    return {"evolution":asdict(bundle.evolution),"development":asdict(bundle.development),"promotion":asdict(bundle.promotion),"sealed":asdict(bundle.sealed) if bundle.sealed is not None else None,"verifier_version":bundle.verifier_version,"policy_version":bundle.policy_version,"policy_verifier_age":bundle.policy_verifier_age,"suite_digests":dict(bundle.suite_digests)}


def bundle_from_payload(raw: Mapping[str, Any]) -> EvaluationBundle:
    return EvaluationBundle(
        evolution=SplitMetrics(**raw["evolution"]), development=SplitMetrics(**raw["development"]),
        promotion=SplitMetrics(**raw["promotion"]), sealed=SplitMetrics(**raw["sealed"]) if raw.get("sealed") is not None else None,
        verifier_version=int(raw["verifier_version"]), policy_version=int(raw["policy_version"]),
        policy_verifier_age=int(raw["policy_verifier_age"]), suite_digests=dict(raw["suite_digests"]),
    )
