from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .backends import InferenceBackend
from .rollout import RolloutRequest
from .telemetry import Telemetry
from .types import Generation
from .worker_health import WorkerHealth


@dataclass
class WorkerSlot:
    name: str
    backend: InferenceBackend
    max_inflight: int = 1

    def __post_init__(self) -> None:
        if self.max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        self.inflight = 0
        self.semaphore = asyncio.Semaphore(self.max_inflight)


class LeastLoadedScheduler:
    """Bounded-queue scheduler with health-aware cross-worker failover.

    Each request is attempted on at most max_attempts_per_request distinct
    workers. A failed worker is excluded from subsequent attempts for that
    request, while the shared WorkerHealth state can quarantine it globally.
    """

    def __init__(
        self,
        workers: list[WorkerSlot],
        *,
        queue_limit: int = 64,
        request_timeout_s: float = 120.0,
        max_attempts_per_request: int | None = None,
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
        self.workers = workers
        self.queue_limit = queue_limit
        self.request_timeout_s = request_timeout_s
        self.max_attempts_per_request = min(
            max_attempts_per_request or len(workers),
            len(workers),
        )
        self.telemetry = telemetry or Telemetry()
        self.health = health or WorkerHealth()
        self._queue_sem = asyncio.Semaphore(queue_limit)
        self._pick_lock = asyncio.Lock()

    async def _pick_worker(self, excluded: set[str] | None = None) -> WorkerSlot:
        excluded = excluded or set()
        async with self._pick_lock:
            healthy = [w for w in self.workers if self.health.is_available(w.name)]
            if not healthy:
                self.telemetry.increment("scheduler.no_healthy_worker", 1)
                raise RuntimeError("no healthy rollout worker is available")
            candidates = [w for w in healthy if w.name not in excluded]
            if not candidates:
                self.telemetry.increment("scheduler.no_untried_worker", 1)
                raise RuntimeError("no untried healthy rollout worker is available")
            worker = min(
                candidates,
                key=lambda w: (w.inflight / w.max_inflight, w.inflight, w.name),
            )
            worker.inflight += 1
            self.telemetry.increment(f"scheduler.dispatch.{worker.name}", 1)
            self.telemetry.observe("scheduler.inflight", worker.inflight)
            return worker

    async def _release_worker(self, worker: WorkerSlot) -> None:
        async with self._pick_lock:
            worker.inflight -= 1
            if worker.inflight < 0:
                raise RuntimeError("worker inflight count underflow")

    def _record_failure(self, worker: WorkerSlot) -> None:
        quarantined = self.health.record_failure(worker.name)
        if quarantined:
            self.telemetry.increment("scheduler.worker_quarantines", 1)

    async def dispatch(self, request: RolloutRequest) -> list[Generation]:
        if self._queue_sem.locked():
            self.telemetry.increment("scheduler.backpressure_events", 1)

        async with self._queue_sem:
            attempted: set[str] = set()
            last_error: Exception | None = None

            for attempt in range(1, self.max_attempts_per_request + 1):
                try:
                    worker = await self._pick_worker(attempted)
                except RuntimeError:
                    if last_error is None:
                        raise
                    break

                attempted.add(worker.name)
                try:
                    async with worker.semaphore:
                        async with asyncio.timeout(self.request_timeout_s):
                            result = await worker.backend.generate(
                                request.prompt_id,
                                request.prompt,
                                n=request.samples,
                                temperature=request.temperature,
                                seed=request.seed,
                            )
                    self.health.record_success(worker.name)
                    self.telemetry.increment("scheduler.completed", 1)
                    self.telemetry.increment("scheduler.samples", len(result))
                    self.telemetry.observe("scheduler.attempts_per_request", attempt)
                    if attempt > 1:
                        self.telemetry.increment("scheduler.failover_successes", 1)
                    return result
                except TimeoutError as exc:
                    last_error = exc
                    self.telemetry.increment("scheduler.timeouts", 1)
                    self._record_failure(worker)
                except Exception as exc:
                    last_error = exc
                    self.telemetry.increment("scheduler.failures", 1)
                    self._record_failure(worker)
                finally:
                    await self._release_worker(worker)

                if attempt < self.max_attempts_per_request:
                    self.telemetry.increment("scheduler.failover_attempts", 1)

            self.telemetry.increment("scheduler.request_failures", 1)
            self.telemetry.observe(
                "scheduler.attempts_per_request",
                max(len(attempted), 1),
            )
            if last_error is None:
                raise RuntimeError("request failed before reaching a rollout worker")
            raise last_error

    async def run(self, requests: list[RolloutRequest]) -> list[Generation]:
        nested = await asyncio.gather(*(self.dispatch(req) for req in requests))
        return [item for batch in nested for item in batch]
