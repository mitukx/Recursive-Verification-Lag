from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class WorkerHealth:
    """Circuit-breaker state for rollout workers.

    By default quarantine is manual, preserving the original behavior.
    Setting quarantine_cooldown_s enables an automatic half-open probe:
    after the cooldown exactly one scheduler reservation may probe the worker.
    """

    failure_threshold: int = 3
    quarantine_cooldown_s: float | None = None
    consecutive_failures: dict[str, int] = field(default_factory=dict)
    quarantined: set[str] = field(default_factory=set)
    quarantined_at: dict[str, float] = field(default_factory=dict)
    half_open: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if self.quarantine_cooldown_s is not None and self.quarantine_cooldown_s < 0:
            raise ValueError("quarantine_cooldown_s must be non-negative")

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def record_success(self, worker: str) -> None:
        self.consecutive_failures[worker] = 0
        self.quarantined.discard(worker)
        self.quarantined_at.pop(worker, None)
        self.half_open.discard(worker)

    def record_failure(self, worker: str) -> bool:
        failures = self.consecutive_failures.get(worker, 0) + 1
        self.consecutive_failures[worker] = failures
        was_quarantined = worker in self.quarantined
        if failures >= self.failure_threshold or was_quarantined or worker in self.half_open:
            self.quarantined.add(worker)
            self.quarantined_at[worker] = self._now()
            self.half_open.discard(worker)
            return True
        return False

    def is_available(self, worker: str) -> bool:
        if worker not in self.quarantined:
            return True
        if self.quarantine_cooldown_s is None or worker in self.half_open:
            return False
        quarantined_at = self.quarantined_at.get(worker)
        if quarantined_at is None:
            return False
        return self._now() - quarantined_at >= self.quarantine_cooldown_s

    def reserve(self, worker: str) -> bool:
        """Reserve a healthy worker or the single allowed half-open probe."""
        if worker not in self.quarantined:
            return True
        if not self.is_available(worker):
            return False
        self.half_open.add(worker)
        return True

    def cancel_reservation(self, worker: str) -> None:
        """Release a half-open reservation without changing health state."""
        self.half_open.discard(worker)

    def recover(self, worker: str) -> None:
        self.quarantined.discard(worker)
        self.quarantined_at.pop(worker, None)
        self.half_open.discard(worker)
        self.consecutive_failures[worker] = 0
