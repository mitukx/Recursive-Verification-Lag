import asyncio
import unittest

from src.rvl_systems.rollout import RolloutRequest
from src.rvl_systems.scheduler import LeastLoadedScheduler, WorkerSlot
from src.rvl_systems.worker_health import WorkerHealth


class AlwaysFailBackend:
    async def generate(self, *args, **kwargs):
        raise RuntimeError("worker failed")


class HealthyBackend:
    async def generate(self, prompt_id, prompt, *, n, temperature, seed):
        from src.rvl_systems.types import Generation
        return [Generation(prompt_id, prompt, "ok", 0.0, 1, 0.0) for _ in range(n)]


class SchedulerHealthIntegrationTest(unittest.TestCase):
    def test_quarantined_worker_is_removed_from_selection(self):
        async def run():
            health = WorkerHealth(failure_threshold=1)
            scheduler = LeastLoadedScheduler(
                [
                    WorkerSlot("bad", AlwaysFailBackend()),
                    WorkerSlot("good", HealthyBackend()),
                ],
                health=health,
            )
            with self.assertRaises(RuntimeError):
                await scheduler.dispatch(RolloutRequest("x", "x", samples=1, seed=1))
            self.assertFalse(health.is_available("bad"))
            out = await scheduler.dispatch(RolloutRequest("y", "y", samples=1, seed=2))
            self.assertEqual(out[0].response, "ok")
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


if __name__ == "__main__":
    unittest.main()
