import asyncio
import time
import unittest

from src.rvl_systems.types import Generation, VerifiedGeneration
from src.rvl_systems.verifier_rpc import AdaptiveVerifierFleet


def gen(i=0):
    return Generation(str(i), "prompt", "42", -0.1, 1, 0.0, {})


class FakeVerifierClient:
    def __init__(self, worker_id, latency_s, capacity=1, version=0, fail=False):
        self.worker_id = worker_id
        self.latency_s = latency_s
        self.capacity = capacity
        self.version = version
        self.fail = fail
        self.calls = 0

    async def ping(self):
        return {
            "ok": True,
            "worker_id": self.worker_id,
            "healthy": True,
            "capacity": self.capacity,
            "inflight": 0,
            "verifier_version": self.version,
            "service_time_hint_s": self.latency_s,
        }

    async def verify(self, generation, *, expected_verifier_version):
        self.calls += 1
        if self.fail:
            raise RuntimeError("injected failure")
        if expected_verifier_version != self.version:
            raise RuntimeError("version mismatch")
        await asyncio.sleep(self.latency_s)
        return VerifiedGeneration(
            generation, 1.0, self.latency_s, self.version,
            {"rpc_worker_id": self.worker_id},
        )


class AdaptiveVerifierFleetTests(unittest.TestCase):
    def test_capacity_latency_aware_router_prefers_fast_capacity(self):
        async def run():
            slow = FakeVerifierClient("slow", 0.020, capacity=1)
            fast = FakeVerifierClient("fast", 0.002, capacity=2)
            fleet = AdaptiveVerifierFleet(
                [slow, fast],
                expected_verifier_version=0,
                default_service_time_s=0.01,
            )
            await fleet.refresh_health()
            rows = await asyncio.gather(*(fleet.verify(gen(i)) for i in range(12)))
            self.assertTrue(all(r.reward == 1.0 for r in rows))
            self.assertGreater(fast.calls, slow.calls)
            snap = fleet.routing_snapshot()
            self.assertEqual(sum(x["completed"] for x in snap), 12)
        asyncio.run(run())

    def test_deadline_failure_quarantines_and_fails_over(self):
        async def run():
            slow = FakeVerifierClient("slow", 0.050, capacity=1)
            fast = FakeVerifierClient("fast", 0.001, capacity=1)
            fleet = AdaptiveVerifierFleet(
                [slow, fast],
                expected_verifier_version=0,
                request_deadline_s=0.010,
                failure_threshold=1,
            )
            await fleet.refresh_health()
            # Force the slow worker to be selected first by making its health hint smaller.
            fleet._service_time[0] = 0.0001
            out = await fleet.verify(gen())
            self.assertEqual(out.metadata["rpc_worker_id"], "fast")
            self.assertEqual(fleet.quarantined, (True, False))
        asyncio.run(run())

    def test_routing_snapshot_exposes_predicted_completion_state(self):
        async def run():
            client = FakeVerifierClient("v", 0.001, capacity=3)
            fleet = AdaptiveVerifierFleet([client], expected_verifier_version=0)
            await fleet.refresh_health()
            before = fleet.routing_snapshot()[0]
            self.assertEqual(before["capacity"], 3)
            await fleet.verify(gen())
            after = fleet.routing_snapshot()[0]
            self.assertEqual(after["completed"], 1)
            self.assertGreater(after["service_time_ewma_s"], 0)
            self.assertGreater(after["predicted_completion_s"], 0)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
