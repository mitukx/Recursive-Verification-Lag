import asyncio
import unittest

from src.rvl_systems.dynamic_batch import DynamicBatcher
from src.rvl_systems.rollout import RolloutRequest
from src.rvl_systems.telemetry import Telemetry
from src.rvl_systems.types import Generation


class RecordingBatchBackend:
    def __init__(self):
        self.calls = []

    async def generate_batch(self, requests):
        self.calls.append(list(requests))
        return [
            [
                Generation(r.prompt_id, r.prompt, "ok", 0.0, 1, 0.0)
                for _ in range(r.samples)
            ]
            for r in requests
        ]


class BrokenBatchBackend:
    async def generate_batch(self, requests):
        return []


class DynamicBatchingTest(unittest.TestCase):
    def test_coalesces_concurrent_compatible_requests(self):
        async def run():
            backend = RecordingBatchBackend()
            telemetry = Telemetry()
            batcher = DynamicBatcher(
                backend,
                max_batch_size=4,
                max_wait_ms=5,
                telemetry=telemetry,
            )
            try:
                tasks = [
                    asyncio.create_task(
                        batcher.submit(
                            RolloutRequest(
                                str(i),
                                f"prompt {i}",
                                samples=1,
                                temperature=0.8,
                                seed=7,
                            )
                        )
                    )
                    for i in range(7)
                ]
                results = await asyncio.gather(*tasks)
                self.assertEqual(len(results), 7)
                self.assertEqual(sum(len(call) for call in backend.calls), 7)
                self.assertTrue(all(len(call) <= 4 for call in backend.calls))
                self.assertLess(len(backend.calls), 7)
                snap = telemetry.snapshot()
                self.assertEqual(snap["batch.requests"], 7)
                self.assertGreaterEqual(snap["batch.size.max"], 2)
            finally:
                await batcher.close()
        asyncio.run(run())

    def test_incompatible_sampling_keys_are_not_mixed(self):
        async def run():
            backend = RecordingBatchBackend()
            batcher = DynamicBatcher(backend, max_batch_size=8, max_wait_ms=2)
            try:
                await asyncio.gather(
                    batcher.submit(RolloutRequest("a", "a", temperature=0.7, seed=1)),
                    batcher.submit(RolloutRequest("b", "b", temperature=0.8, seed=1)),
                    batcher.submit(RolloutRequest("c", "c", temperature=0.7, seed=2)),
                )
                self.assertEqual(len(backend.calls), 3)
            finally:
                await batcher.close()
        asyncio.run(run())

    def test_cardinality_failure_reaches_all_callers(self):
        async def run():
            batcher = DynamicBatcher(BrokenBatchBackend(), max_wait_ms=0)
            try:
                results = await asyncio.gather(
                    batcher.submit(RolloutRequest("a", "a")),
                    batcher.submit(RolloutRequest("b", "b")),
                    return_exceptions=True,
                )
                self.assertTrue(all(isinstance(x, RuntimeError) for x in results))
            finally:
                await batcher.close()
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
