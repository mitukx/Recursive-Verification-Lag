from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

from .rollout import RolloutRequest
from .telemetry import Telemetry
from .types import Generation


class BatchInferenceBackend(Protocol):
    async def generate_batch(
        self,
        requests: list[RolloutRequest],
    ) -> list[list[Generation]]:
        ...


@dataclass
class _Pending:
    request: RolloutRequest
    future: asyncio.Future[list[Generation]]


class DynamicBatcher:
    """Coalesce compatible rollout requests into bounded micro-batches.

    Requests are compatible when they share sampling count, temperature and
    seed. Keeping the seed in the key preserves the exact InferenceBackend
    contract for backends whose batch API exposes only one seed per batch.
    """

    def __init__(
        self,
        backend: BatchInferenceBackend,
        *,
        max_batch_size: int = 16,
        max_wait_ms: float = 2.0,
        telemetry: Telemetry | None = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000.0
        self.telemetry = telemetry or Telemetry()
        self._lock = asyncio.Lock()
        self._buckets: dict[tuple[int, float, int], list[_Pending]] = defaultdict(list)
        self._timers: dict[tuple[int, float, int], asyncio.Task[None]] = {}
        self._closed = False

    @staticmethod
    def _key(request: RolloutRequest) -> tuple[int, float, int]:
        return (request.samples, float(request.temperature), request.seed)

    async def submit(self, request: RolloutRequest) -> list[Generation]:
        if self._closed:
            raise RuntimeError("dynamic batcher is closed")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[Generation]] = loop.create_future()
        key = self._key(request)
        flush_now = False
        async with self._lock:
            bucket = self._buckets[key]
            bucket.append(_Pending(request, future))
            self.telemetry.observe("batch.queue_depth", len(bucket))
            if len(bucket) == 1:
                self._timers[key] = asyncio.create_task(self._delayed_flush(key))
            if len(bucket) >= self.max_batch_size:
                timer = self._timers.pop(key, None)
                if timer is not None:
                    timer.cancel()
                flush_now = True
        if flush_now:
            asyncio.create_task(self._flush(key))
        return await future

    async def _delayed_flush(self, key: tuple[int, float, int]) -> None:
        try:
            if self.max_wait_s:
                await asyncio.sleep(self.max_wait_s)
            await self._flush(key)
        except asyncio.CancelledError:
            return

    async def _flush(self, key: tuple[int, float, int]) -> None:
        async with self._lock:
            bucket = self._buckets.get(key)
            if not bucket:
                self._timers.pop(key, None)
                return
            batch = bucket[: self.max_batch_size]
            remainder = bucket[self.max_batch_size :]
            if remainder:
                self._buckets[key] = remainder
                self._timers[key] = asyncio.create_task(self._delayed_flush(key))
            else:
                self._buckets.pop(key, None)
                self._timers.pop(key, None)

        requests = [item.request for item in batch]
        self.telemetry.increment("batch.calls", 1)
        self.telemetry.increment("batch.requests", len(requests))
        self.telemetry.observe("batch.size", len(requests))
        try:
            results = await self.backend.generate_batch(requests)
            if len(results) != len(batch):
                raise RuntimeError(
                    f"batch backend returned {len(results)} result groups for "
                    f"{len(batch)} requests"
                )
            for pending, generations in zip(batch, results):
                if len(generations) != pending.request.samples:
                    raise RuntimeError(
                        f"batch backend returned {len(generations)} generations "
                        f"for request expecting {pending.request.samples}"
                    )
                if not pending.future.done():
                    pending.future.set_result(generations)
        except Exception as exc:
            self.telemetry.increment("batch.failures", 1)
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_exception(exc)

    async def close(self) -> None:
        self._closed = True
        async with self._lock:
            timers = list(self._timers.values())
            keys = list(self._buckets)
            self._timers.clear()
        for timer in timers:
            timer.cancel()
        for key in keys:
            await self._flush(key)
