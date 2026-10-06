from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from .telemetry import Telemetry
from .types import Generation, VerifiedGeneration


class Verifier(Protocol):
    @property
    def version(self) -> int:
        ...

    async def verify(self, generation: Generation) -> VerifiedGeneration:
        ...

    def refresh(self) -> None:
        ...


@dataclass
class ExactMatchVerifier:
    """Cheap deterministic verifier for RLVR experiments."""

    answers: dict[str, str]
    latency_s: float = 0.0
    telemetry: Telemetry | None = None

    def __post_init__(self) -> None:
        self._version = 0
        self.telemetry = self.telemetry or Telemetry()

    @property
    def version(self) -> int:
        return self._version

    def refresh(self) -> None:
        self._version += 1
        self.telemetry.increment("verifier.refreshes", 1)

    async def verify(self, generation: Generation) -> VerifiedGeneration:
        if generation.prompt_id not in self.answers:
            raise KeyError(f"no trusted answer for {generation.prompt_id}")
        start = time.perf_counter()
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        reward = float(
            generation.response.strip() == self.answers[generation.prompt_id].strip()
        )
        elapsed = time.perf_counter() - start
        self.telemetry.observe("verifier.latency_s", elapsed)
        self.telemetry.increment("verifier.calls", 1)
        return VerifiedGeneration(
            generation=generation,
            reward=reward,
            verifier_latency_s=elapsed,
            verifier_version=self.version,
        )


@dataclass
class FunctionalVerifier:
    """Adapter for custom reward functions without coupling to a benchmark."""

    fn: Callable[[Generation], float]
    telemetry: Telemetry | None = None

    def __post_init__(self) -> None:
        self._version = 0
        self.telemetry = self.telemetry or Telemetry()

    @property
    def version(self) -> int:
        return self._version

    def refresh(self) -> None:
        self._version += 1
        self.telemetry.increment("verifier.refreshes", 1)

    async def verify(self, generation: Generation) -> VerifiedGeneration:
        start = time.perf_counter()
        reward = float(self.fn(generation))
        elapsed = time.perf_counter() - start
        self.telemetry.observe("verifier.latency_s", elapsed)
        self.telemetry.increment("verifier.calls", 1)
        return VerifiedGeneration(
            generation=generation,
            reward=reward,
            verifier_latency_s=elapsed,
            verifier_version=self.version,
        )
