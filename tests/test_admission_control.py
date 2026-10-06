import asyncio
import unittest

from src.rvl_systems.admission import (
    FairWorkloadAdmission,
    OverloadedError,
)


class FairWorkloadAdmissionTest(unittest.TestCase):
    def test_round_robin_prevents_same_workload_monopoly(self):
        async def run():
            controller = FairWorkloadAdmission(
                1,
                max_queued_units=4,
            )
            first = await controller.acquire("train-a", 1)

            order: list[str] = []

            async def waiter(workload: str, label: str):
                lease = await controller.acquire(workload, 1)
                order.append(label)
                await asyncio.sleep(0)
                await lease.release()

            a2 = asyncio.create_task(
                waiter("train-a", "a2")
            )
            await asyncio.sleep(0)
            b1 = asyncio.create_task(
                waiter("eval-b", "b1")
            )
            await asyncio.sleep(0)

            await first.release()
            await asyncio.gather(a2, b1)
            self.assertEqual(order, ["b1", "a2"])

        asyncio.run(run())

    def test_bounded_queue_sheds_excess_work(self):
        async def run():
            controller = FairWorkloadAdmission(
                1,
                max_queued_units=1,
            )
            active = await controller.acquire("a", 1)
            queued_task = asyncio.create_task(
                controller.acquire("b", 1)
            )
            await asyncio.sleep(0)

            with self.assertRaises(OverloadedError):
                await controller.acquire("c", 1)

            await active.release()
            queued = await queued_task
            await queued.release()

        asyncio.run(run())

    def test_cancelled_waiter_does_not_leak_capacity(self):
        async def run():
            controller = FairWorkloadAdmission(
                1,
                max_queued_units=2,
            )
            active = await controller.acquire("a", 1)
            queued = asyncio.create_task(
                controller.acquire("b", 1)
            )
            await asyncio.sleep(0)
            queued.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await queued

            await active.release()
            next_lease = await controller.acquire("c", 1)
            self.assertEqual(controller.in_use_units, 1)
            await next_lease.release()
            self.assertEqual(controller.in_use_units, 0)
            self.assertEqual(controller.queued_units, 0)

        asyncio.run(run())

    def test_per_workload_cap_preserves_global_headroom(self):
        async def run():
            controller = FairWorkloadAdmission(
                4,
                max_queued_units=8,
                per_workload_capacity_units=2,
            )
            a = await controller.acquire("a", 2)
            b = await controller.acquire("b", 2)
            self.assertEqual(controller.in_use_units, 4)
            await a.release()
            await b.release()

        asyncio.run(run())

    def test_speculation_cannot_jump_normal_queue(self):
        async def run():
            controller = FairWorkloadAdmission(
                2,
                max_queued_units=4,
            )
            active = await controller.acquire("a", 2)
            queued = asyncio.create_task(
                controller.acquire("b", 1)
            )
            await asyncio.sleep(0)
            speculative = await controller.try_acquire("hedge", 1)
            self.assertIsNone(speculative)
            await active.release()
            lease = await queued
            await lease.release()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
