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
    """Bounded-queue scheduler with least-loaded healthy-worker selection."""

    def __init__(
        self,
        workers: list[WorkerSlot],
        *,
        queue_limit: int = 64,
        request_timeout_s: float = 120.0,
        telemetry: Telemetry | None = None,
        health: WorkerHealth | None = None,
    ) -> None:
        if not workers:
            raise ValueError("at least one worker is required")
        if queue_limit <= 0:
            raise ValueError("queue_limit must be positive")
        if request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        self.workers = workers
        self.queue_limit = queue_limit
        self.request_timeout_s = request_timeout_s
        self.telemetry = telemetry or Telemetry()
        self.health = health or WorkerHealth()
        self._queue_sem = asyncio.Semaphore(queue_limit)
        self._pick_lock = asyncio.Lock()

    async def _pick_worker(self) -> WorkerSlot:
        async with self._pick_lock:
            candidates = [w for w in self.workers if self.health.is_available(w.name)]
            if not candidates:
                self.telemetry.increment("scheduler.no_healthy_worker", 1)
                raise RuntimeError("no healthy rollout worker is available")
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
            worker = await self._pick_worker()
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
                return result
            except TimeoutError:
                self.telemetry.increment("scheduler.timeouts", 1)
                self._record_failure(worker)
                raise
            except Exception:
                self.telemetry.increment("scheduler.failures", 1)
                self._record_failure(worker)
                raise
            finally:
                await self._release_worker(worker)

    async def run(self, requests: list[RolloutRequest]) -> list[Generation]:
        nested = await asyncio.gather(*(self.dispatch(req) for req in requests))
        return [item for batch in nested for item in batch]
