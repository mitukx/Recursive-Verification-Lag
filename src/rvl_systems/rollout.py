from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from .backends import InferenceBackend
from .telemetry import Telemetry
from .types import Generation


@dataclass(frozen=True)
class RolloutRequest:
    prompt_id: str
    prompt: str
    samples: int = 4
    temperature: float = 1.0
    seed: int = 0
    deadline_s: float | None = None

    def __post_init__(self) -> None:
        if self.samples <= 0:
            raise ValueError("samples must be positive")
        if self.deadline_s is not None and self.deadline_s <= 0:
            raise ValueError("deadline_s must be positive")


class AsyncRolloutEngine:
    """Bounded-concurrency rollout scheduler with per-request telemetry."""

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        max_concurrency: int = 8,
        telemetry: Telemetry | None = None,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self.backend = backend
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.telemetry = telemetry or Telemetry()

    async def _generate(self, request: RolloutRequest) -> list[Generation]:
        async with self._semaphore:
            return await self.backend.generate(
                request.prompt_id,
                request.prompt,
                n=request.samples,
                temperature=request.temperature,
                seed=request.seed,
            )

    async def _one(self, request: RolloutRequest) -> list[Generation]:
        start = time.perf_counter()
        if request.deadline_s is None:
            generations = await self._generate(request)
        else:
            async with asyncio.timeout(request.deadline_s):
                generations = await self._generate(request)
        elapsed = time.perf_counter() - start
        self.telemetry.observe("rollout.request_latency_s", elapsed)
        self.telemetry.increment("rollout.requests", 1)
        self.telemetry.increment("rollout.samples", len(generations))
        self.telemetry.increment(
            "rollout.tokens", sum(g.token_count for g in generations)
        )
        return generations

    async def run(self, requests: list[RolloutRequest]) -> list[Generation]:
        nested = await asyncio.gather(*(self._one(r) for r in requests))
        return [g for batch in nested for g in batch]
