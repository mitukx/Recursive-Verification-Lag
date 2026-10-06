import asyncio
import unittest

from src.rvl_systems.rollout import RolloutRequest
from src.rvl_systems.scheduler import LeastLoadedScheduler, WorkerSlot
from src.rvl_systems.telemetry import Telemetry
from src.rvl_systems.worker_health import WorkerHealth


class AlwaysFailBackend:
    async def generate(self, *args, **kwargs):
        raise RuntimeError("worker failed")


class HealthyBackend:
    async def generate(self, prompt_id, prompt, *, n, temperature, seed):
        from src.rvl_systems.types import Generation
        return [Generation(prompt_id, prompt, "ok", 0.0, 1, 0.0) for _ in range(n)]


class SchedulerHealthIntegrationTest(unittest.TestCase):
    def test_failed_request_fails_over_to_healthy_worker(self):
        async def run():
            health = WorkerHealth(failure_threshold=1)
            telemetry = Telemetry()
            scheduler = LeastLoadedScheduler(
                [
                    WorkerSlot("bad", AlwaysFailBackend()),
                    WorkerSlot("good", HealthyBackend()),
                ],
                health=health,
                telemetry=telemetry,
            )
            out = await scheduler.dispatch(RolloutRequest("x", "x", samples=1, seed=1))
            self.assertEqual(out[0].response, "ok")
            self.assertFalse(health.is_available("bad"))
            snap = telemetry.snapshot()
            self.assertEqual(snap["scheduler.failures"], 1)
            self.assertEqual(snap["scheduler.failover_attempts"], 1)
            self.assertEqual(snap["scheduler.failover_successes"], 1)
            self.assertEqual(snap["scheduler.completed"], 1)
        asyncio.run(run())

    def test_all_quarantined_fails_fast(self):
        async def run():
            health = WorkerHealth(failure_threshold=1)
            health.record_failure("bad")
            scheduler = LeastLoadedScheduler(
                [WorkerSlot("bad", AlwaysFailBackend())],
                health=health,
            )
            with self.assertRaisesRegex(RuntimeError, "no healthy rollout worker"):
                await scheduler.dispatch(RolloutRequest("x", "x"))
        asyncio.run(run())

    def test_exhausts_distinct_workers_then_raises_last_error(self):
        async def run():
            health = WorkerHealth(failure_threshold=10)
            telemetry = Telemetry()
            scheduler = LeastLoadedScheduler(
                [
                    WorkerSlot("bad-a", AlwaysFailBackend()),
                    WorkerSlot("bad-b", AlwaysFailBackend()),
                ],
                health=health,
                telemetry=telemetry,
            )
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                await scheduler.dispatch(RolloutRequest("x", "x"))
            snap = telemetry.snapshot()
            self.assertEqual(snap["scheduler.failures"], 2)
            self.assertEqual(snap["scheduler.failover_attempts"], 1)
            self.assertEqual(snap["scheduler.request_failures"], 1)
            self.assertEqual(snap["scheduler.attempts_per_request.max"], 2)
        asyncio.run(run())

    def test_attempt_budget_can_disable_failover(self):
        async def run():
            telemetry = Telemetry()
            scheduler = LeastLoadedScheduler(
                [
                    WorkerSlot("bad", AlwaysFailBackend()),
                    WorkerSlot("good", HealthyBackend()),
                ],
                max_attempts_per_request=1,
                telemetry=telemetry,
            )
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                await scheduler.dispatch(RolloutRequest("x", "x"))
            self.assertNotIn("scheduler.failover_successes", telemetry.snapshot())
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
