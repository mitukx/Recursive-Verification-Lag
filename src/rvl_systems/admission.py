from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque

from .telemetry import Telemetry


class OverloadedError(RuntimeError):
    """Raised when bounded queued work would exceed the configured budget."""


@dataclass
class _Waiter:
    workload_id: str
    cost: int
    future: asyncio.Future[bool]
    enqueued_at: float
    granted: bool = False
    cancelled: bool = False


class AdmissionLease:
    """Idempotent lease returned by FairWorkloadAdmission."""

    def __init__(
        self,
        controller: "FairWorkloadAdmission",
        workload_id: str,
        cost: int,
    ) -> None:
        self._controller = controller
        self.workload_id = workload_id
        self.cost = cost
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._controller._release(
            self.workload_id,
            self.cost,
        )

    async def __aenter__(self) -> "AdmissionLease":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.release()


class FairWorkloadAdmission:
    """Weighted, bounded admission with round-robin workload fairness.

    Work is measured in integer units supplied by the caller. For rollout
    scheduling the default unit is either the request's estimated token count
    or its sample count.

    The controller enforces:
    - a global in-flight work budget;
    - an optional per-workload in-flight budget;
    - a bounded queued-work budget with immediate load shedding;
    - round-robin service across active workload queues;
    - cancellation- and timeout-safe accounting.

    Speculative work (for example a hedged rollout) may use try_acquire().
    Speculation is denied whenever normal work is already queued, so it cannot
    steal capacity from waiting requests.
    """

    def __init__(
        self,
        capacity_units: int,
        *,
        max_queued_units: int | None = None,
        per_workload_capacity_units: int | None = None,
        telemetry: Telemetry | None = None,
    ) -> None:
        if capacity_units <= 0:
            raise ValueError("capacity_units must be positive")
        if max_queued_units is not None and max_queued_units < 0:
            raise ValueError("max_queued_units must be non-negative")
        if (
            per_workload_capacity_units is not None
            and per_workload_capacity_units <= 0
        ):
            raise ValueError(
                "per_workload_capacity_units must be positive"
            )
        if (
            per_workload_capacity_units is not None
            and per_workload_capacity_units > capacity_units
        ):
            raise ValueError(
                "per_workload_capacity_units cannot exceed capacity_units"
            )

        self.capacity_units = capacity_units
        self.max_queued_units = (
            max_queued_units
            if max_queued_units is not None
            else capacity_units * 4
        )
        self.per_workload_capacity_units = (
            per_workload_capacity_units
        )
        self.telemetry = telemetry or Telemetry()

        self._available = capacity_units
        self._queued_units = 0
        self._inflight_by_workload: dict[str, int] = defaultdict(int)
        self._queues: dict[str, Deque[_Waiter]] = defaultdict(deque)
        self._rotation: Deque[str] = deque()
        self._active_workloads: set[str] = set()
        self._last_granted_workload: str | None = None
        self._lock = asyncio.Lock()

    @property
    def in_use_units(self) -> int:
        return self.capacity_units - self._available

    @property
    def queued_units(self) -> int:
        return self._queued_units

    def _validate_request(
        self,
        workload_id: str,
        cost: int,
    ) -> None:
        if not workload_id:
            raise ValueError("workload_id must be non-empty")
        if cost <= 0:
            raise ValueError("cost must be positive")
        if cost > self.capacity_units:
            raise OverloadedError(
                f"request cost {cost} exceeds global capacity "
                f"{self.capacity_units}"
            )
        limit = self.per_workload_capacity_units
        if limit is not None and cost > limit:
            raise OverloadedError(
                f"request cost {cost} exceeds per-workload capacity "
                f"{limit}"
            )

    def _workload_fits(
        self,
        workload_id: str,
        cost: int,
    ) -> bool:
        if cost > self._available:
            return False
        limit = self.per_workload_capacity_units
        if limit is None:
            return True
        return (
            self._inflight_by_workload[workload_id] + cost
            <= limit
        )

    def _activate_workload_locked(
        self,
        workload_id: str,
    ) -> None:
        if workload_id in self._active_workloads:
            return
        self._active_workloads.add(workload_id)
        self._rotation.append(workload_id)

    def _deactivate_workload_locked(
        self,
        workload_id: str,
    ) -> None:
        self._active_workloads.discard(workload_id)
        try:
            self._rotation.remove(workload_id)
        except ValueError:
            pass
        if not self._queues.get(workload_id):
            self._queues.pop(workload_id, None)

    def _observe_state_locked(self) -> None:
        self.telemetry.observe(
            "admission.in_use_units",
            self.in_use_units,
        )
        self.telemetry.observe(
            "admission.queued_units",
            self._queued_units,
        )

    def _drop_cancelled_heads_locked(
        self,
        workload_id: str,
    ) -> None:
        queue = self._queues.get(workload_id)
        if queue is None:
            return
        while queue and (
            queue[0].cancelled
            or queue[0].future.cancelled()
        ):
            waiter = queue.popleft()
            self._queued_units -= waiter.cost
        if not queue:
            self._deactivate_workload_locked(workload_id)

    def _rotate_after_last_grant_locked(self) -> None:
        last = self._last_granted_workload
        if last is None or last not in self._active_workloads:
            return
        while self._rotation and self._rotation[0] != last:
            self._rotation.rotate(-1)
        if self._rotation and self._rotation[0] == last:
            self._rotation.rotate(-1)

    def _drain_locked(self) -> None:
        self._rotate_after_last_grant_locked()

        while self._rotation and self._available > 0:
            round_size = len(self._rotation)
            progressed = False

            for _ in range(round_size):
                workload_id = self._rotation.popleft()
                self._drop_cancelled_heads_locked(workload_id)
                queue = self._queues.get(workload_id)
                if not queue:
                    continue

                waiter = queue[0]
                if self._workload_fits(
                    workload_id,
                    waiter.cost,
                ):
                    queue.popleft()
                    self._queued_units -= waiter.cost
                    self._available -= waiter.cost
                    self._inflight_by_workload[workload_id] += (
                        waiter.cost
                    )
                    waiter.granted = True
                    self._last_granted_workload = workload_id
                    if not waiter.future.done():
                        waiter.future.set_result(True)
                    self.telemetry.increment(
                        "admission.admitted",
                        1,
                    )
                    progressed = True

                if queue:
                    self._rotation.append(workload_id)
                else:
                    self._active_workloads.discard(workload_id)
                    self._queues.pop(workload_id, None)

            if not progressed:
                break
            self._rotate_after_last_grant_locked()

        self._observe_state_locked()

    def _cancel_waiter_locked(
        self,
        waiter: _Waiter,
    ) -> None:
        if waiter.cancelled:
            return
        waiter.cancelled = True

        if waiter.granted:
            waiter.granted = False
            self._available += waiter.cost
            self._inflight_by_workload[waiter.workload_id] -= (
                waiter.cost
            )
            if self._inflight_by_workload[waiter.workload_id] == 0:
                self._inflight_by_workload.pop(
                    waiter.workload_id,
                    None,
                )
            return

        queue = self._queues.get(waiter.workload_id)
        if queue is not None:
            try:
                queue.remove(waiter)
                self._queued_units -= waiter.cost
            except ValueError:
                pass
            if not queue:
                self._deactivate_workload_locked(
                    waiter.workload_id
                )

    async def acquire(
        self,
        workload_id: str,
        cost: int,
        *,
        timeout_s: float | None = None,
    ) -> AdmissionLease:
        self._validate_request(workload_id, cost)
        if timeout_s is not None and timeout_s <= 0:
            raise TimeoutError("admission deadline exceeded")

        loop = asyncio.get_running_loop()
        waiter = _Waiter(
            workload_id=workload_id,
            cost=cost,
            future=loop.create_future(),
            enqueued_at=time.perf_counter(),
        )

        async with self._lock:
            if self._queued_units + cost > self.max_queued_units:
                self.telemetry.increment(
                    "admission.shed",
                    1,
                )
                raise OverloadedError(
                    "queued work budget exhausted: "
                    f"queued={self._queued_units}, cost={cost}, "
                    f"limit={self.max_queued_units}"
                )
            self._queues[workload_id].append(waiter)
            self._queued_units += cost
            self._activate_workload_locked(workload_id)
            self.telemetry.increment(
                "admission.enqueued",
                1,
            )
            self._drain_locked()

        try:
            if timeout_s is None:
                await asyncio.shield(waiter.future)
            else:
                async with asyncio.timeout(timeout_s):
                    await asyncio.shield(waiter.future)
        except TimeoutError:
            async with self._lock:
                self._cancel_waiter_locked(waiter)
                self.telemetry.increment(
                    "admission.timeouts",
                    1,
                )
                self._drain_locked()
            raise
        except asyncio.CancelledError:
            async with self._lock:
                self._cancel_waiter_locked(waiter)
                self.telemetry.increment(
                    "admission.cancellations",
                    1,
                )
                self._drain_locked()
            raise

        wait_s = time.perf_counter() - waiter.enqueued_at
        self.telemetry.observe(
            "admission.wait_s",
            wait_s,
        )
        return AdmissionLease(
            self,
            workload_id,
            cost,
        )

    async def try_acquire(
        self,
        workload_id: str,
        cost: int,
    ) -> AdmissionLease | None:
        """Acquire only if capacity is immediately spare.

        Normal queued work always takes precedence over speculative work.
        """
        self._validate_request(workload_id, cost)
        async with self._lock:
            if self._queued_units > 0 or self._rotation:
                self.telemetry.increment(
                    "admission.speculative_denied",
                    1,
                )
                return None
            if not self._workload_fits(workload_id, cost):
                self.telemetry.increment(
                    "admission.speculative_denied",
                    1,
                )
                return None
            self._available -= cost
            self._inflight_by_workload[workload_id] += cost
            self.telemetry.increment(
                "admission.speculative_admitted",
                1,
            )
            self._observe_state_locked()
        return AdmissionLease(
            self,
            workload_id,
            cost,
        )

    async def _release(
        self,
        workload_id: str,
        cost: int,
    ) -> None:
        async with self._lock:
            current = self._inflight_by_workload.get(
                workload_id,
                0,
            )
            if cost > current:
                raise RuntimeError(
                    "admission release exceeds in-flight allocation"
                )
            self._inflight_by_workload[workload_id] = current - cost
            if self._inflight_by_workload[workload_id] == 0:
                self._inflight_by_workload.pop(
                    workload_id,
                    None,
                )
            self._available += cost
            if self._available > self.capacity_units:
                raise RuntimeError(
                    "admission capacity accounting overflow"
                )
            self.telemetry.increment(
                "admission.released",
                1,
            )
            self._drain_locked()
