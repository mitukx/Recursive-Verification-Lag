from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveRefreshController:
    """Refresh when policy movement or verifier age exceeds a budget.

    movement_budget is an application-defined policy-shift coordinate. In a
    real RVL experiment this can be KL, max density ratio, or restricted
    verifier-error geometry.
    """

    movement_budget: float = 0.75
    max_stale_steps: int = 4

    def __post_init__(self) -> None:
        if self.movement_budget < 0:
            raise ValueError("movement_budget must be non-negative")
        if self.max_stale_steps <= 0:
            raise ValueError("max_stale_steps must be positive")
        self._stale_steps = 0
        self._accumulated_movement = 0.0

    @property
    def stale_steps(self) -> int:
        return self._stale_steps

    @property
    def accumulated_movement(self) -> float:
        return self._accumulated_movement

    def observe(self, movement: float) -> tuple[bool, str]:
        if movement < 0:
            raise ValueError("movement must be non-negative")
        self._stale_steps += 1
        self._accumulated_movement += movement
        if self._accumulated_movement >= self.movement_budget:
            self.reset()
            return True, "movement"
        if self._stale_steps >= self.max_stale_steps:
            self.reset()
            return True, "staleness"
        return False, "none"

    def reset(self) -> None:
        self._stale_steps = 0
        self._accumulated_movement = 0.0
