from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorkerHealth:
    failure_threshold: int = 3
    consecutive_failures: dict[str, int] = field(default_factory=dict)
    quarantined: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")

    def record_success(self, worker: str) -> None:
        self.consecutive_failures[worker] = 0

    def record_failure(self, worker: str) -> bool:
        failures = self.consecutive_failures.get(worker, 0) + 1
        self.consecutive_failures[worker] = failures
        if failures >= self.failure_threshold:
            self.quarantined.add(worker)
            return True
        return False

    def is_available(self, worker: str) -> bool:
        return worker not in self.quarantined

    def recover(self, worker: str) -> None:
        self.quarantined.discard(worker)
        self.consecutive_failures[worker] = 0
