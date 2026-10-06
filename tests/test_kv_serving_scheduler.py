import asyncio
import unittest

from src.rvl_systems.rollout import RolloutRequest
from src.rvl_systems.scheduler import LeastLoadedScheduler, WorkerSlot
from src.rvl_systems.types import Generation


class DelayBackend:
    def __init__(self, response: str, delay_s: float = 0.0):
        self.response = response
        self.delay_s = delay_s

    async def generate(self, prompt_id, prompt, *, n, temperature, seed):
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return [
            Generation(prompt_id, prompt, self.response, 0.0, 1, self.delay_s)
            for _ in range(n)
        ]


class KVServingSchedulerTest(unittest.TestCase):
    def test_prefill_heavy_request_prefers_prefill_worker(self):
        async def run():
            prefill = WorkerSlot(
                "prefill", DelayBackend("prefill"),
                prefill_tokens_per_s=10000,
                decode_tokens_per_s=100,
            )
            decode = WorkerSlot(
                "decode", DelayBackend("decode"),
                prefill_tokens_per_s=100,
                decode_tokens_per_s=10000,
            )
            scheduler = LeastLoadedScheduler([prefill, decode])
            out = await scheduler.dispatch(
                RolloutRequest(
                    "p", "x", samples=1,
                    prompt_tokens_estimate=1000,
                    decode_tokens_estimate=1,
                )
            )
            self.assertEqual(out[0].response, "prefill")

        asyncio.run(run())

    def test_decode_heavy_request_prefers_decode_worker(self):
        async def run():
            prefill = WorkerSlot(
                "prefill", DelayBackend("prefill"),
                prefill_tokens_per_s=10000,
                decode_tokens_per_s=100,
            )
            decode = WorkerSlot(
                "decode", DelayBackend("decode"),
                prefill_tokens_per_s=100,
                decode_tokens_per_s=10000,
            )
            scheduler = LeastLoadedScheduler([prefill, decode])
            out = await scheduler.dispatch(
                RolloutRequest(
                    "p", "x", samples=4,
                    prompt_tokens_estimate=1,
                    decode_tokens_estimate=500,
                )
            )
            self.assertEqual(out[0].response, "decode")

        asyncio.run(run())

    def test_kv_capacity_excludes_worker_and_releases_reservation(self):
        async def run():
            small = WorkerSlot(
                "small", DelayBackend("small"), max_kv_tokens=100
            )
            large = WorkerSlot(
                "large", DelayBackend("large"), max_kv_tokens=10000
            )
            scheduler = LeastLoadedScheduler([small, large])
            out = await scheduler.dispatch(
                RolloutRequest(
                    "p", "x", samples=2,
                    prompt_tokens_estimate=100,
                    decode_tokens_estimate=50,
                )
            )
            self.assertEqual(out[0].response, "large")
            self.assertEqual(small.active_kv_tokens, 0)
            self.assertEqual(large.active_kv_tokens, 0)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
