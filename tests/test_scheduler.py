import asyncio
import unittest

from src.rvl_systems.backends import ToyTabularBackend
from src.rvl_systems.rollout import RolloutRequest
from src.rvl_systems.scheduler import LeastLoadedScheduler, WorkerSlot
from src.rvl_systems.telemetry import Telemetry


class SchedulerTest(unittest.TestCase):
    def test_balances_across_workers(self):
        async def run():
            telemetry = Telemetry()
            scheduler = LeastLoadedScheduler(
                [
                    WorkerSlot("w0", ToyTabularBackend(base_latency_s=0.002), max_inflight=1),
                    WorkerSlot("w1", ToyTabularBackend(base_latency_s=0.002), max_inflight=1),
                ],
                queue_limit=8,
                telemetry=telemetry,
            )
            reqs = [RolloutRequest(str(i), "x", samples=1, seed=i) for i in range(8)]
            out = await scheduler.run(reqs)
            self.assertEqual(len(out), 8)
            snap = telemetry.snapshot()
            self.assertEqual(snap["scheduler.completed"], 8)
            self.assertGreater(snap["scheduler.dispatch.w0"], 0)
            self.assertGreater(snap["scheduler.dispatch.w1"], 0)
        asyncio.run(run())

    def test_records_backpressure(self):
        async def run():
            telemetry = Telemetry()
            scheduler = LeastLoadedScheduler(
                [WorkerSlot("w", ToyTabularBackend(base_latency_s=0.01), max_inflight=1)],
                queue_limit=1,
                telemetry=telemetry,
            )
            reqs = [RolloutRequest(str(i), "x", samples=1, seed=i) for i in range(4)]
            await scheduler.run(reqs)
            self.assertGreater(telemetry.snapshot().get("scheduler.backpressure_events", 0), 0)
        asyncio.run(run())

    def test_timeout_is_counted(self):
        async def run():
            telemetry = Telemetry()
            scheduler = LeastLoadedScheduler(
                [WorkerSlot("slow", ToyTabularBackend(base_latency_s=0.05), max_inflight=1)],
                queue_limit=2,
                request_timeout_s=0.005,
                telemetry=telemetry,
            )
            with self.assertRaises(TimeoutError):
                await scheduler.dispatch(RolloutRequest("p", "x", samples=1, seed=1))
            self.assertEqual(telemetry.snapshot()["scheduler.timeouts"], 1)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
