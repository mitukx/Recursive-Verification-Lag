import asyncio
import unittest

from src.rvl_systems.admission import OverloadedError
from src.rvl_systems.rollout import RolloutRequest
from src.rvl_systems.scheduler import LeastLoadedScheduler, WorkerSlot
from src.rvl_systems.telemetry import Telemetry
from src.rvl_systems.types import Generation


class DelayBackend:
    def __init__(self, delay_s: float, response: str):
        self.delay_s = delay_s
        self.response = response

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


class SchedulerAdmissionTest(unittest.TestCase):
    def test_weighted_budget_limits_concurrent_work(self):
        async def run():
            telemetry = Telemetry()
            scheduler = LeastLoadedScheduler(
                [
                    WorkerSlot(
                        "w0",
                        DelayBackend(0.010, "ok"),
                        max_inflight=4,
                    )
                ],
                queue_limit=8,
                max_inflight_work_units=4,
                max_queued_work_units=8,
                telemetry=telemetry,
            )
            requests = [
                RolloutRequest(
                    str(i),
                    "x",
                    samples=1,
                    estimated_tokens=2,
                    workload_id="train",
                )
                for i in range(4)
            ]
            out = await scheduler.run(requests)
            self.assertEqual(len(out), 4)
            snap = telemetry.snapshot()
            self.assertLessEqual(
                snap["admission.in_use_units.max"],
                4,
            )

        asyncio.run(run())

    def test_overload_is_shed_instead_of_unbounded_queueing(self):
        async def run():
            scheduler = LeastLoadedScheduler(
                [
                    WorkerSlot(
                        "w0",
                        DelayBackend(0.030, "ok"),
                        max_inflight=1,
                    )
                ],
                queue_limit=8,
                max_inflight_work_units=1,
                max_queued_work_units=1,
            )
            first = asyncio.create_task(
                scheduler.dispatch(
                    RolloutRequest(
                        "a",
                        "x",
                        samples=1,
                        estimated_tokens=1,
                        workload_id="train",
                    )
                )
            )
            await asyncio.sleep(0.003)
            second = asyncio.create_task(
                scheduler.dispatch(
                    RolloutRequest(
                        "b",
                        "x",
                        samples=1,
                        estimated_tokens=1,
                        workload_id="eval",
                    )
                )
            )
            await asyncio.sleep(0.003)
            with self.assertRaises(OverloadedError):
                await scheduler.dispatch(
                    RolloutRequest(
                        "c",
                        "x",
                        samples=1,
                        estimated_tokens=1,
                        workload_id="other",
                    )
                )
            await first
            await second

        asyncio.run(run())

    def test_hedge_respects_spare_work_budget(self):
        async def run():
            telemetry = Telemetry()
            primary = WorkerSlot(
                "primary",
                DelayBackend(0.020, "primary"),
            )
            backup = WorkerSlot(
                "backup",
                DelayBackend(0.001, "backup"),
            )
            primary.latency_ewma_s = 0.001
            backup.latency_ewma_s = 0.010
            scheduler = LeastLoadedScheduler(
                [primary, backup],
                hedge_after_s=0.002,
                max_attempts_per_request=2,
                max_inflight_work_units=1,
                max_queued_work_units=2,
                telemetry=telemetry,
            )
            out = await scheduler.dispatch(
                RolloutRequest(
                    "p",
                    "x",
                    samples=1,
                    estimated_tokens=1,
                    workload_id="train",
                )
            )
            self.assertEqual(out[0].response, "primary")
            self.assertEqual(
                telemetry.snapshot().get(
                    "scheduler.hedge_budget_denied",
                    0,
                ),
                1,
            )

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
