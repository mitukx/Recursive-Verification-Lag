from __future__ import annotations

import asyncio
import hashlib
import inspect
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Awaitable, Callable, Protocol

from .rollout import RolloutRequest
from .types import Generation


class EvalRunner(Protocol):
    async def run(self, requests: list[RolloutRequest]) -> list[Generation]:
        ...


RewardFn = Callable[["EvalTask", Generation], float | Awaitable[float]]


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    prompt: str
    slice: str = "default"
    split: str = "eval"
    estimated_tokens: int | None = None
    deadline_s: float | None = None

    def __post_init__(self) -> None:
        if not self.task_id or not self.prompt:
            raise ValueError("task_id and prompt must be non-empty")
        if not self.slice:
            raise ValueError("slice must be non-empty")
        if self.split not in {"eval", "validation", "test"}:
            raise ValueError("training examples cannot enter evaluation fabric")
        if self.estimated_tokens is not None and self.estimated_tokens <= 0:
            raise ValueError("estimated_tokens must be positive")
        if self.deadline_s is not None and self.deadline_s <= 0:
            raise ValueError("deadline_s must be positive")


@dataclass(frozen=True)
class EvalResult:
    candidate: str
    task_id: str
    slice: str
    seed: int
    status: str
    reward: float | None
    latency_s: float
    token_count: int
    response: str | None
    response_sha256: str | None
    error_type: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"ok", "infra_failure", "evaluator_failure"}:
            raise ValueError("invalid evaluation status")
        if self.latency_s < 0 or self.token_count < 0:
            raise ValueError("invalid evaluation measurements")
        if self.status == "ok":
            if (
                self.reward is None
                or not math.isfinite(self.reward)
                or not 0 <= self.reward <= 1
            ):
                raise ValueError("successful result requires finite reward in [0,1]")
            if self.response is None or self.response_sha256 is None:
                raise ValueError("successful result requires response provenance")
        elif self.reward is not None:
            raise ValueError("failed evaluation must not carry a reward")


@dataclass(frozen=True)
class PairwiseComparison:
    candidate_a: str
    candidate_b: str
    complete_pairs: int
    incomplete_pairs: int
    mean_reward_a: float | None
    mean_reward_b: float | None
    mean_delta_b_minus_a: float | None
    wins_a: int
    wins_b: int
    ties: int
    exact_sign_test_p: float | None
    slice_deltas_b_minus_a: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def exact_two_sided_sign_test(wins_a: int, wins_b: int) -> float | None:
    """Exact two-sided binomial sign test over non-tied paired outcomes."""
    if wins_a < 0 or wins_b < 0:
        raise ValueError("win counts must be non-negative")
    n = wins_a + wins_b
    if n == 0:
        return None
    tail = min(wins_a, wins_b)
    mass = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, 2.0 * mass)


class EvaluationFabric:
    """Fail-closed asynchronous paired evaluation across model/fleet runners.

    Runners may be AsyncRolloutEngine instances, LeastLoadedScheduler fleets, or
    any object implementing run(list[RolloutRequest]). Every candidate sees the
    same task identity and seed. Infrastructure/evaluator failures are kept
    separate from model reward and make the overall run incomplete.
    """

    def __init__(
        self,
        runners: dict[str, EvalRunner],
        reward_fn: RewardFn,
        *,
        max_inflight: int = 32,
        seed: int = 20261006,
        temperature: float = 0.0,
    ) -> None:
        if len(runners) < 2:
            raise ValueError("paired evaluation requires at least two candidates")
        if any(not name for name in runners):
            raise ValueError("candidate names must be non-empty")
        if max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        self.runners = dict(runners)
        self.reward_fn = reward_fn
        self.max_inflight = max_inflight
        self.seed = seed
        self.temperature = temperature
        self._semaphore = asyncio.Semaphore(max_inflight)

    @staticmethod
    def validate_tasks(tasks: list[EvalTask]) -> None:
        if not tasks:
            raise ValueError("evaluation suite must be non-empty")
        ids = [task.task_id for task in tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate evaluation task IDs")

    async def _reward(self, task: EvalTask, generation: Generation) -> float:
        value = self.reward_fn(task, generation)
        if inspect.isawaitable(value):
            value = await value
        reward = float(value)
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("evaluator reward must be finite and in [0,1]")
        return reward

    async def _one(
        self,
        candidate: str,
        runner: EvalRunner,
        task: EvalTask,
        index: int,
    ) -> EvalResult:
        paired_seed = self.seed + index
        request = RolloutRequest(
            prompt_id=task.task_id,
            prompt=task.prompt,
            samples=1,
            temperature=self.temperature,
            seed=paired_seed,
            deadline_s=task.deadline_s,
            estimated_tokens=task.estimated_tokens,
            workload_id=f"eval:{candidate}",
        )
        started = time.perf_counter()
        async with self._semaphore:
            try:
                generations = await runner.run([request])
                latency = time.perf_counter() - started
            except Exception as exc:
                return EvalResult(
                    candidate, task.task_id, task.slice, paired_seed,
                    "infra_failure", None, time.perf_counter() - started, 0,
                    None, None, type(exc).__name__,
                )
        if len(generations) != 1:
            return EvalResult(
                candidate, task.task_id, task.slice, paired_seed,
                "infra_failure", None, latency, 0, None, None,
                "GenerationCardinalityError",
            )
        generation = generations[0]
        if generation.prompt_id != task.task_id:
            return EvalResult(
                candidate, task.task_id, task.slice, paired_seed,
                "infra_failure", None, latency, generation.token_count, None, None,
                "PromptIdentityMismatch",
            )
        response_hash = hashlib.sha256(
            generation.response.encode("utf-8")
        ).hexdigest()
        try:
            reward = await self._reward(task, generation)
        except Exception as exc:
            return EvalResult(
                candidate, task.task_id, task.slice, paired_seed,
                "evaluator_failure", None, latency, generation.token_count,
                generation.response, response_hash, type(exc).__name__,
            )
        return EvalResult(
            candidate=candidate,
            task_id=task.task_id,
            slice=task.slice,
            seed=paired_seed,
            status="ok",
            reward=reward,
            latency_s=latency,
            token_count=generation.token_count,
            response=generation.response,
            response_sha256=response_hash,
        )

    @staticmethod
    def compare(
        candidate_a: str,
        candidate_b: str,
        tasks: list[EvalTask],
        results: list[EvalResult],
    ) -> PairwiseComparison:
        by_key = {(r.candidate, r.task_id): r for r in results}
        complete: list[tuple[EvalTask, EvalResult, EvalResult]] = []
        incomplete = 0
        for task in tasks:
            left = by_key.get((candidate_a, task.task_id))
            right = by_key.get((candidate_b, task.task_id))
            if (
                left is None
                or right is None
                or left.status != "ok"
                or right.status != "ok"
            ):
                incomplete += 1
                continue
            if left.seed != right.seed:
                raise RuntimeError("paired candidate seeds diverged")
            complete.append((task, left, right))

        if not complete:
            return PairwiseComparison(
                candidate_a, candidate_b, 0, incomplete, None, None, None,
                0, 0, 0, None, {},
            )

        left_rewards = [float(left.reward) for _, left, _ in complete]
        right_rewards = [float(right.reward) for _, _, right in complete]
        wins_a = sum(a > b for a, b in zip(left_rewards, right_rewards))
        wins_b = sum(b > a for a, b in zip(left_rewards, right_rewards))
        ties = len(complete) - wins_a - wins_b

        slices: dict[str, list[float]] = {}
        for task, left, right in complete:
            slices.setdefault(task.slice, []).append(
                float(right.reward) - float(left.reward)
            )
        return PairwiseComparison(
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            complete_pairs=len(complete),
            incomplete_pairs=incomplete,
            mean_reward_a=fmean(left_rewards),
            mean_reward_b=fmean(right_rewards),
            mean_delta_b_minus_a=fmean(
                b - a for a, b in zip(left_rewards, right_rewards)
            ),
            wins_a=wins_a,
            wins_b=wins_b,
            ties=ties,
            exact_sign_test_p=exact_two_sided_sign_test(wins_a, wins_b),
            slice_deltas_b_minus_a={
                key: fmean(values) for key, values in sorted(slices.items())
            },
        )

    async def run(self, tasks: list[EvalTask]) -> dict:
        self.validate_tasks(tasks)
        suite_payload = [asdict(task) for task in tasks]
        suite_sha = _digest(suite_payload)
        jobs = [
            self._one(candidate, runner, task, index)
            for candidate, runner in sorted(self.runners.items())
            for index, task in enumerate(tasks)
        ]
        started = time.perf_counter()
        results = list(await asyncio.gather(*jobs))
        wall_s = time.perf_counter() - started

        candidate_summaries = {}
        for candidate, runner in sorted(self.runners.items()):
            rows = [r for r in results if r.candidate == candidate]
            ok = [r for r in rows if r.status == "ok"]
            telemetry = None
            if (
                hasattr(runner, "telemetry")
                and hasattr(runner.telemetry, "snapshot")
            ):
                telemetry = runner.telemetry.snapshot()
            candidate_summaries[candidate] = {
                "completed": len(ok),
                "failures": len(rows) - len(ok),
                "mean_reward": (
                    fmean(float(r.reward) for r in ok) if ok else None
                ),
                "tokens": sum(r.token_count for r in ok),
                "latency_s_p50": _percentile(
                    [r.latency_s for r in rows], 0.50
                ),
                "latency_s_p95": _percentile(
                    [r.latency_s for r in rows], 0.95
                ),
                "telemetry": telemetry,
            }

        comparisons = [
            self.compare(a, b, tasks, results).to_dict()
            for a, b in itertools.combinations(sorted(self.runners), 2)
        ]
        failure_count = sum(r.status != "ok" for r in results)
        expected = len(tasks) * len(self.runners)
        semantic = {
            "suite_sha256": suite_sha,
            "candidates": sorted(self.runners),
            "results": [
                asdict(r)
                for r in sorted(results, key=lambda x: (x.candidate, x.task_id))
            ],
            "comparisons": comparisons,
        }
        return {
            "schema": 1,
            "complete": failure_count == 0 and len(results) == expected,
            "suite_sha256": suite_sha,
            "semantic_sha256": _digest(semantic),
            "task_count": len(tasks),
            "candidate_count": len(self.runners),
            "expected_results": expected,
            "successful_results": expected - failure_count,
            "failure_count": failure_count,
            "wall_s": wall_s,
            "candidate_summaries": candidate_summaries,
            "comparisons": comparisons,
            "results": [
                asdict(r)
                for r in sorted(results, key=lambda x: (x.candidate, x.task_id))
            ],
            "claim_boundary": (
                "Infrastructure/evaluator failures make the run incomplete and "
                "are never silently converted into model reward."
            ),
        }
