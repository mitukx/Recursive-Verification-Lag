from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from .backends import InferenceBackend
from .rollout import RolloutRequest
from .telemetry import Telemetry
from .types import Generation
from .worker_health import WorkerHealth


class WorkerUnavailableError(RuntimeError):
    """A previously admitted worker became unavailable before execution."""


@dataclass
class WorkerSlot:
    name: str
    backend: InferenceBackend
    max_inflight: int = 1
    latency_ewma_s: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        self.inflight = 0
        self.semaphore = asyncio.Semaphore(self.max_inflight)


class LeastLoadedScheduler:
    """SLO-aware rollout scheduler with adaptive routing and fault tolerance.

    The scheduler supports:
    - bounded queueing and per-request end-to-end deadlines;
    - latency-aware worker selection using an EWMA service-time estimate;
    - cross-worker failover on request failures;
    - optional tail-latency hedging, where a slow primary is raced against a
      distinct healthy worker and the losing request is cancelled;
    - circuit-breaker health state, including optional half-open recovery.
    """

    def __init__(
        self,
        workers: list[WorkerSlot],
        *,
        queue_limit: int = 64,
        request_timeout_s: float = 120.0,
        max_attempts_per_request: int | None = None,
        latency_ewma_alpha: float = 0.2,
        hedge_after_s: float | None = None,
        telemetry: Telemetry | None = None,
        health: WorkerHealth | None = None,
    ) -> None:
        if not workers:
            raise ValueError("at least one worker is required")
        if queue_limit <= 0:
            raise ValueError("queue_limit must be positive")
        if request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        if max_attempts_per_request is not None and max_attempts_per_request <= 0:
            raise ValueError("max_attempts_per_request must be positive")
        if not 0.0 < latency_ewma_alpha <= 1.0:
            raise ValueError("latency_ewma_alpha must be in (0, 1]")
        if hedge_after_s is not None and hedge_after_s <= 0:
            raise ValueError("hedge_after_s must be positive")
        self.workers = workers
        self.queue_limit = queue_limit
        self.request_timeout_s = request_timeout_s
        self.max_attempts_per_request = min(
            max_attempts_per_request or len(workers),
            len(workers),
        )
        self.latency_ewma_alpha = latency_ewma_alpha
        self.hedge_after_s = hedge_after_s
        self.telemetry = telemetry or Telemetry()
        self.health = health or WorkerHealth()
        self._queue_sem = asyncio.Semaphore(queue_limit)
        self._pick_lock = asyncio.Lock()

    def _fallback_latency_s(self) -> float:
        known = [
            w.latency_ewma_s
            for w in self.workers
            if w.latency_ewma_s is not None
        ]
        return min(known) if known else 1e-6

    def _worker_score(self, worker: WorkerSlot) -> tuple[float, float, str]:
        predicted = (
            worker.latency_ewma_s
            if worker.latency_ewma_s is not None
            else self._fallback_latency_s()
        )
        completion_cost = predicted * (worker.inflight + 1) / worker.max_inflight
        return completion_cost, worker.inflight / worker.max_inflight, worker.name

    async def _pick_worker(self, excluded: set[str] | None = None) -> WorkerSlot:
        excluded = excluded or set()
        async with self._pick_lock:
            candidates = [
                w
                for w in self.workers
                if w.name not in excluded and self.health.is_available(w.name)
            ]
            while candidates:
                worker = min(candidates, key=self._worker_score)
                if self.health.reserve(worker.name):
                    worker.inflight += 1
                    self.telemetry.increment(
                        f"scheduler.dispatch.{worker.name}",
                        1,
                    )
                    self.telemetry.observe("scheduler.inflight", worker.inflight)
                    return worker
                candidates.remove(worker)

            healthy = [
                w for w in self.workers if self.health.is_available(w.name)
            ]
            if not healthy:
                self.telemetry.increment("scheduler.no_healthy_worker", 1)
                raise RuntimeError("no healthy rollout worker is available")
            self.telemetry.increment("scheduler.no_untried_worker", 1)
            raise RuntimeError("no untried healthy rollout worker is available")

    async def _release_worker(self, worker: WorkerSlot) -> None:
        async with self._pick_lock:
            worker.inflight -= 1
            if worker.inflight < 0:
                raise RuntimeError("worker inflight count underflow")

    def _record_latency(self, worker: WorkerSlot, elapsed_s: float) -> None:
        if worker.latency_ewma_s is None:
            worker.latency_ewma_s = elapsed_s
        else:
            alpha = self.latency_ewma_alpha
            worker.latency_ewma_s = (
                alpha * elapsed_s
                + (1.0 - alpha) * worker.latency_ewma_s
            )
        self.telemetry.observe(
            f"scheduler.worker_latency_s.{worker.name}",
            elapsed_s,
        )
        self.telemetry.observe(
            f"scheduler.worker_latency_ewma_s.{worker.name}",
            worker.latency_ewma_s,
        )

    def _record_failure(self, worker: WorkerSlot) -> None:
        quarantined = self.health.record_failure(worker.name)
        if quarantined:
            self.telemetry.increment("scheduler.worker_quarantines", 1)

    @staticmethod
    def _remaining(deadline_at: float | None) -> float | None:
        if deadline_at is None:
            return None
        return deadline_at - time.perf_counter()

    def _attempt_timeout(
        self,
        deadline_at: float | None,
    ) -> tuple[float, bool]:
        remaining = self._remaining(deadline_at)
        if remaining is None:
            return self.request_timeout_s, False
        if remaining <= 0:
            raise TimeoutError("rollout request deadline exceeded")
        return (
            min(self.request_timeout_s, remaining),
            remaining <= self.request_timeout_s,
        )

    async def _attempt(
        self,
        request: RolloutRequest,
        worker: WorkerSlot,
        *,
        deadline_at: float | None,
    ) -> list[Generation]:
        started = time.perf_counter()
        try:
            timeout_s, deadline_limited = self._attempt_timeout(deadline_at)
        except TimeoutError:
            self.telemetry.increment("scheduler.deadline_exceeded", 1)
            self.health.cancel_reservation(worker.name)
            await self._release_worker(worker)
            raise

        semaphore_acquired = False
        backend_started = False
        attempt_deadline = started + timeout_s
        try:
            try:
                await asyncio.wait_for(
                    worker.semaphore.acquire(),
                    timeout=max(
                        attempt_deadline - time.perf_counter(),
                        1e-12,
                    ),
                )
                semaphore_acquired = True
                if not self.health.can_execute(worker.name):
                    self.telemetry.increment(
                        "scheduler.stale_reservations",
                        1,
                    )
                    self.health.cancel_reservation(worker.name)
                    raise WorkerUnavailableError(
                        f"rollout worker {worker.name} became unavailable"
                    )
            except TimeoutError:
                self.telemetry.increment(
                    "scheduler.capacity_wait_timeouts",
                    1,
                )
                if deadline_limited:
                    self.telemetry.increment(
                        "scheduler.deadline_exceeded",
                        1,
                    )
                self.health.cancel_reservation(worker.name)
                raise

            backend_budget = attempt_deadline - time.perf_counter()
            remaining = self._remaining(deadline_at)
            if remaining is not None:
                backend_budget = min(backend_budget, remaining)
            if backend_budget <= 0:
                self.telemetry.increment(
                    "scheduler.deadline_exceeded",
                    1,
                )
                self.health.cancel_reservation(worker.name)
                raise TimeoutError("rollout request deadline exceeded")

            backend_started = True
            async with asyncio.timeout(backend_budget):
                result = await worker.backend.generate(
                    request.prompt_id,
                    request.prompt,
                    n=request.samples,
                    temperature=request.temperature,
                    seed=request.seed,
                )
            elapsed = time.perf_counter() - started
            self._record_latency(worker, elapsed)
            self.health.record_success(worker.name)
            return result
        except asyncio.CancelledError:
            self.telemetry.increment("scheduler.cancelled_attempts", 1)
            self.health.cancel_reservation(worker.name)
            raise
        except TimeoutError:
            if backend_started:
                elapsed = time.perf_counter() - started
                self._record_latency(worker, elapsed)
                self.telemetry.increment("scheduler.timeouts", 1)
                if deadline_limited or (
                    deadline_at is not None
                    and self._remaining(deadline_at) <= 0
                ):
                    self.telemetry.increment(
                        "scheduler.deadline_exceeded",
                        1,
                    )
                self._record_failure(worker)
            raise
        except WorkerUnavailableError:
            raise
        except Exception:
            self.telemetry.increment("scheduler.failures", 1)
            self._record_failure(worker)
            raise
        finally:
            if semaphore_acquired:
                worker.semaphore.release()
            await self._release_worker(worker)

    async def _finish_success(
        self,
        result: list[Generation],
        *,
        attempts: int,
        had_failure: bool,
    ) -> list[Generation]:
        self.telemetry.increment("scheduler.completed", 1)
        self.telemetry.increment("scheduler.samples", len(result))
        self.telemetry.observe(
            "scheduler.attempts_per_request",
            attempts,
        )
        if had_failure:
            self.telemetry.increment(
                "scheduler.failover_successes",
                1,
            )
        return result

    async def _race_hedge(
        self,
        request: RolloutRequest,
        *,
        primary: WorkerSlot,
        attempted: set[str],
        deadline_at: float | None,
    ) -> tuple[
        list[Generation] | None,
        int,
        Exception | None,
        bool,
    ]:
        """Run the first attempt and optionally race a hedge."""
        primary_task = asyncio.create_task(
            self._attempt(
                request,
                primary,
                deadline_at=deadline_at,
            )
        )
        remaining = self._remaining(deadline_at)
        wait_s = self.hedge_after_s
        assert wait_s is not None
        if remaining is not None:
            wait_s = min(wait_s, max(remaining, 0.0))

        done, _ = await asyncio.wait(
            {primary_task},
            timeout=wait_s,
        )
        if primary_task in done:
            try:
                return primary_task.result(), 1, None, False
            except Exception as exc:
                return None, 1, exc, True

        remaining = self._remaining(deadline_at)
        if remaining is not None and remaining <= 0:
            try:
                await primary_task
            except Exception as exc:
                return None, 1, exc, True
            raise AssertionError(
                "primary unexpectedly succeeded after expired deadline"
            )

        try:
            secondary = await self._pick_worker(attempted)
        except RuntimeError:
            try:
                return await primary_task, 1, None, False
            except Exception as exc:
                return None, 1, exc, True

        attempted.add(secondary.name)
        self.telemetry.increment("scheduler.hedge_launched", 1)
        secondary_task = asyncio.create_task(
            self._attempt(
                request,
                secondary,
                deadline_at=deadline_at,
            )
        )
        pending: set[asyncio.Task[list[Generation]]] = {
            primary_task,
            secondary_task,
        }
        last_error: Exception | None = None
        had_failure = False

        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                try:
                    result = task.result()
                except Exception as exc:
                    last_error = exc
                    had_failure = True
                    continue

                if task is secondary_task:
                    self.telemetry.increment(
                        "scheduler.hedge_wins",
                        1,
                    )
                else:
                    self.telemetry.increment(
                        "scheduler.hedge_primary_wins",
                        1,
                    )

                for loser in pending:
                    loser.cancel()
                if pending:
                    await asyncio.gather(
                        *pending,
                        return_exceptions=True,
                    )
                    self.telemetry.increment(
                        "scheduler.hedge_cancellations",
                        len(pending),
                    )
                return result, 2, last_error, had_failure

        return None, 2, last_error, True

    async def dispatch(
        self,
        request: RolloutRequest,
    ) -> list[Generation]:
        started = time.perf_counter()
        deadline_at = (
            started + request.deadline_s
            if request.deadline_s is not None
            else None
        )
        if self._queue_sem.locked():
            self.telemetry.increment(
                "scheduler.backpressure_events",
                1,
            )

        queue_acquired = False
        try:
            if deadline_at is None:
                await self._queue_sem.acquire()
            else:
                remaining = self._remaining(deadline_at)
                if remaining is None or remaining <= 0:
                    raise TimeoutError(
                        "rollout request deadline exceeded in queue"
                    )
                try:
                    await asyncio.wait_for(
                        self._queue_sem.acquire(),
                        timeout=remaining,
                    )
                except TimeoutError:
                    self.telemetry.increment(
                        "scheduler.deadline_exceeded",
                        1,
                    )
                    self.telemetry.increment(
                        "scheduler.queue_deadline_exceeded",
                        1,
                    )
                    self.telemetry.increment(
                        "scheduler.request_failures",
                        1,
                    )
                    raise TimeoutError(
                        "rollout request deadline exceeded in queue"
                    )
            queue_acquired = True
            self.telemetry.observe(
                "scheduler.queue_wait_s",
                time.perf_counter() - started,
            )

            attempted: set[str] = set()
            last_error: Exception | None = None
            had_failure = False
            attempts = 0

            primary = await self._pick_worker(attempted)
            attempted.add(primary.name)
            if (
                self.hedge_after_s is not None
                and self.max_attempts_per_request >= 2
            ):
                (
                    result,
                    used,
                    error,
                    hedge_had_failure,
                ) = await self._race_hedge(
                    request,
                    primary=primary,
                    attempted=attempted,
                    deadline_at=deadline_at,
                )
                attempts += used
                had_failure = (
                    had_failure or hedge_had_failure
                )
                last_error = error
                if result is not None:
                    self.telemetry.observe(
                        "scheduler.end_to_end_latency_s",
                        time.perf_counter() - started,
                    )
                    return await self._finish_success(
                        result,
                        attempts=attempts,
                        had_failure=had_failure,
                    )
            else:
                attempts += 1
                try:
                    result = await self._attempt(
                        request,
                        primary,
                        deadline_at=deadline_at,
                    )
                    self.telemetry.observe(
                        "scheduler.end_to_end_latency_s",
                        time.perf_counter() - started,
                    )
                    return await self._finish_success(
                        result,
                        attempts=attempts,
                        had_failure=False,
                    )
                except Exception as exc:
                    last_error = exc
                    had_failure = True

            while attempts < self.max_attempts_per_request:
                remaining = self._remaining(deadline_at)
                if remaining is not None and remaining <= 0:
                    if not isinstance(last_error, TimeoutError):
                        last_error = TimeoutError(
                            "rollout request deadline exceeded"
                        )
                        self.telemetry.increment(
                            "scheduler.deadline_exceeded",
                            1,
                        )
                    break

                self.telemetry.increment(
                    "scheduler.failover_attempts",
                    1,
                )
                try:
                    worker = await self._pick_worker(attempted)
                except RuntimeError:
                    break
                attempted.add(worker.name)
                attempts += 1
                try:
                    result = await self._attempt(
                        request,
                        worker,
                        deadline_at=deadline_at,
                    )
                    self.telemetry.observe(
                        "scheduler.end_to_end_latency_s",
                        time.perf_counter() - started,
                    )
                    return await self._finish_success(
                        result,
                        attempts=attempts,
                        had_failure=True,
                    )
                except Exception as exc:
                    last_error = exc
                    had_failure = True

            self.telemetry.increment(
                "scheduler.request_failures",
                1,
            )
            self.telemetry.observe(
                "scheduler.attempts_per_request",
                max(attempts, 1),
            )
            self.telemetry.observe(
                "scheduler.end_to_end_latency_s",
                time.perf_counter() - started,
            )
            if last_error is None:
                raise RuntimeError(
                    "request failed before reaching a rollout worker"
                )
            raise last_error
        finally:
            if queue_acquired:
                self._queue_sem.release()

    async def run(
        self,
        requests: list[RolloutRequest],
    ) -> list[Generation]:
        nested = await asyncio.gather(
            *(self.dispatch(req) for req in requests)
        )
        return [
            item
            for batch in nested
            for item in batch
        ]
