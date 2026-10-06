import asyncio
import time
import unittest

from src.rvl_systems.rollout import RolloutRequest
from src.rvl_systems.scheduler import LeastLoadedScheduler, WorkerSlot
from src.rvl_systems.telemetry import Telemetry
from src.rvl_systems.types import Generation
from src.rvl_systems.worker_health import WorkerHealth


class DelayBackend:
    def __init__(
        self,
        delay_s: float,
        response: str,
        *,
        failures: int = 0,
    ):
        self.delay_s = delay_s
        self.response = response
        self.failures = failures

    async def generate(
        self,
        prompt_id,
        prompt,
        *,
        n,
        temperature,
        seed,
    ):
        await asyncio.sleep(self.delay_s)
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError(f"{self.response} failed")
        return [
            Generation(
                prompt_id,
                prompt,
                self.response,
                0.0,
                1,
                self.delay_s,
            )
            for _ in range(n)
        ]


class SLOAwareSchedulerTest(unittest.TestCase):
    def test_latency_estimate_changes_idle_worker_choice(self):
        async def run():
            slow = WorkerSlot(
                "slow",
                DelayBackend(0.001, "slow"),
            )
            fast = WorkerSlot(
                "fast",
                DelayBackend(0.001, "fast"),
            )
            slow.latency_ewma_s = 0.050
            fast.latency_ewma_s = 0.005
            scheduler = LeastLoadedScheduler([slow, fast])
            out = await scheduler.dispatch(
                RolloutRequest("p", "x", samples=1)
            )
            self.assertEqual(out[0].response, "fast")

        asyncio.run(run())

    def test_hedge_beats_slow_primary_and_cancels_loser(self):
        async def run():
            telemetry = Telemetry()
            primary = WorkerSlot(
                "primary",
                DelayBackend(0.050, "slow"),
            )
            backup = WorkerSlot(
                "backup",
                DelayBackend(0.002, "fast"),
            )
            primary.latency_ewma_s = 0.001
            backup.latency_ewma_s = 0.010
            scheduler = LeastLoadedScheduler(
                [primary, backup],
                hedge_after_s=0.005,
                max_attempts_per_request=2,
                telemetry=telemetry,
            )
            start = time.perf_counter()
            out = await scheduler.dispatch(
                RolloutRequest(
                    "p",
                    "x",
                    samples=1,
                    deadline_s=0.030,
                )
            )
            elapsed = time.perf_counter() - start
            self.assertEqual(out[0].response, "fast")
            self.assertLess(elapsed, 0.030)
            snap = telemetry.snapshot()
            self.assertEqual(
                snap["scheduler.hedge_launched"],
                1,
            )
            self.assertEqual(
                snap["scheduler.hedge_wins"],
                1,
            )
            self.assertEqual(
                snap["scheduler.hedge_cancellations"],
                1,
            )

        asyncio.run(run())

    def test_deadline_is_end_to_end_and_bounds_queue_wait(self):
        async def run():
            telemetry = Telemetry()
            worker = WorkerSlot(
                "only",
                DelayBackend(0.050, "ok"),
                max_inflight=1,
            )
            scheduler = LeastLoadedScheduler(
                [worker],
                queue_limit=1,
                telemetry=telemetry,
            )
            first = asyncio.create_task(
                scheduler.dispatch(
                    RolloutRequest(
                        "first",
                        "x",
                        samples=1,
                    )
                )
            )
            await asyncio.sleep(0.005)
            start = time.perf_counter()
            with self.assertRaises(TimeoutError):
                await scheduler.dispatch(
                    RolloutRequest(
                        "second",
                        "x",
                        samples=1,
                        deadline_s=0.010,
                    )
                )
            elapsed = time.perf_counter() - start
            self.assertLess(elapsed, 0.030)
            self.assertGreaterEqual(
                telemetry.snapshot().get(
                    "scheduler.queue_deadline_exceeded",
                    0,
                ),
                1,
            )
            await first

        asyncio.run(run())

    def test_half_open_probe_auto_recovers_worker(self):
        async def run():
            backend = DelayBackend(
                0.0,
                "recovered",
                failures=1,
            )
            health = WorkerHealth(
                failure_threshold=1,
                quarantine_cooldown_s=0.010,
            )
            scheduler = LeastLoadedScheduler(
                [WorkerSlot("worker", backend)],
                health=health,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "failed",
            ):
                await scheduler.dispatch(
                    RolloutRequest(
                        "first",
                        "x",
                        samples=1,
                    )
                )
            self.assertFalse(
                health.is_available("worker")
            )
            await asyncio.sleep(0.015)
            out = await scheduler.dispatch(
                RolloutRequest(
                    "second",
                    "x",
                    samples=1,
                )
            )
            self.assertEqual(
                out[0].response,
                "recovered",
            )
            self.assertTrue(
                health.is_available("worker")
            )
            self.assertNotIn(
                "worker",
                health.half_open,
            )

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
