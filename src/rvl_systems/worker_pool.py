from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

from .telemetry import Telemetry

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    backoff_s: float = 0.05

    def __post_init__(self) -> None:
        if self.attempts <= 0:
            raise ValueError("attempts must be positive")
        if self.backoff_s < 0:
            raise ValueError("backoff_s must be non-negative")


async def with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    telemetry: Telemetry | None = None,
    metric_prefix: str = "worker",
) -> T:
    telemetry = telemetry or Telemetry()
    last_error: Exception | None = None
    for attempt in range(1, policy.attempts + 1):
        try:
            result = await fn()
            telemetry.increment(f"{metric_prefix}.success", 1)
            telemetry.observe(f"{metric_prefix}.attempts", attempt)
            return result
        except Exception as exc:  # infrastructure boundary
            last_error = exc
            telemetry.increment(f"{metric_prefix}.failures", 1)
            if attempt == policy.attempts:
                break
            if policy.backoff_s:
                await asyncio.sleep(policy.backoff_s * attempt)
    assert last_error is not None
    raise last_error


class VersionedWorkerPool:
    """Tracks rollout-worker policy versions and rejects stale results."""

    def __init__(self, *, telemetry: Telemetry | None = None) -> None:
        self.telemetry = telemetry or Telemetry()
        self._policy_version = 0

    @property
    def policy_version(self) -> int:
        return self._policy_version

    def publish_new_version(self) -> int:
        self._policy_version += 1
        self.telemetry.increment("weights.publishes", 1)
        return self._policy_version

    def validate_result_version(self, result_version: int) -> None:
        if result_version != self._policy_version:
            self.telemetry.increment("weights.stale_results", 1)
            raise RuntimeError(
                f"stale rollout result: got policy version {result_version}, "
                f"expected {self._policy_version}"
            )
        self.telemetry.increment("weights.accepted_results", 1)
