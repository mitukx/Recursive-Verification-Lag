import asyncio
import unittest

from src.rvl_systems.backends import ToyTabularBackend
from src.rvl_systems.lab.judges import MultiVerifier
from src.rvl_systems.lab.serving import VersionedServingFleet
from src.rvl_systems.rollout import RolloutRequest
from src.rvl_systems.scheduler import WorkerSlot
from src.rvl_systems.types import Generation


class ServingTests(unittest.IsolatedAsyncioTestCase):
    async def test_inflight_episode_survives_atomic_version_activation(self):
        fleet = VersionedServingFleet()
        async def probe(worker,version):
            return True
        await fleet.prepare(0,[WorkerSlot("old",ToyTabularBackend(actions=("old",)))],probe)
        await fleet.activate(0)
        async with fleet.episode() as old:
            await fleet.prepare(1,[WorkerSlot("new",ToyTabularBackend(actions=("new",)))],probe)
            await fleet.activate(1)
            rows = await old.generate(RolloutRequest("old-task","prompt",samples=1))
            self.assertEqual(rows[0].response,"old")
            self.assertEqual(rows[0].metadata["policy_version"],0)
            async with fleet.episode() as new:
                rows = await new.generate(RolloutRequest("new-task","prompt",samples=1))
                self.assertEqual(rows[0].response,"new")
            with self.assertRaises(RuntimeError):
                await fleet.retire(0)
        await fleet.retire(0)
        with self.assertRaises(RuntimeError):
            await old.generate(RolloutRequest("closed","prompt"))
        self.assertEqual(fleet.status(),{"active":1,"episode_references":{1:0}})

    async def test_bad_health_and_mutable_backend_reuse_rejected(self):
        fleet = VersionedServingFleet()
        backend = ToyTabularBackend()
        async def healthy(w,v):
            return True
        async def unhealthy(w,v):
            return False
        with self.assertRaises(RuntimeError):
            await fleet.prepare(0,[WorkerSlot("bad",backend)],unhealthy)
        await fleet.prepare(0,[WorkerSlot("good",backend)],healthy)
        with self.assertRaises(ValueError):
            await fleet.prepare(1,[WorkerSlot("reused",backend)],healthy)

    async def test_exception_releases_episode_reference(self):
        fleet = VersionedServingFleet()
        async def probe(w,v):
            return True
        await fleet.prepare(0,[WorkerSlot("a",ToyTabularBackend())],probe)
        await fleet.activate(0)
        with self.assertRaises(ValueError):
            async with fleet.episode():
                raise ValueError("actor failure")
        self.assertEqual(fleet.references[0],0)


class EnsembleTests(unittest.IsolatedAsyncioTestCase):
    def generation(self):
        return Generation("p","task","candidate",0,1,0)

    async def test_disagreement_trusted_audit_and_fail_closed(self):
        async def public(g):
            return 1.0
        async def critic(g):
            return .25
        async def trusted(g):
            return 0.0
        verifier = MultiVerifier({"public":public,"critic":critic},trusted=trusted)
        result = await verifier.verify(self.generation())
        self.assertEqual(result.reward,.25)
        self.assertEqual(result.metadata["disagreement"],.75)
        self.assertEqual(await verifier.audit(self.generation()),0)
        self.assertFalse(result.metadata["trusted"])

    async def test_timeout_and_nonfinite_grader_abstain(self):
        async def slow(g):
            await asyncio.sleep(1)
            return 1
        async def invalid(g):
            return float("nan")
        verifier = MultiVerifier({"slow":slow,"invalid":invalid},timeout_s=.001)
        r = await verifier.verify(self.generation())
        self.assertEqual(r.reward,0)
        self.assertEqual(set(r.metadata["failed_graders"]),{"slow","invalid"})

    async def test_required_graders_are_concurrency_bounded(self):
        active = 0
        maximum = 0
        async def judge(g):
            nonlocal active,maximum
            active += 1
            maximum = max(maximum,active)
            await asyncio.sleep(.001)
            active -= 1
            return 1.0
        v = MultiVerifier({str(i):judge for i in range(8)},max_concurrency=2)
        await v.verify(self.generation())
        self.assertEqual(maximum,2)


class GPUMeasurementTests(unittest.TestCase):
    def test_mfu_requires_explicit_flop_model_and_peak(self):
        from src.rvl_systems.lab.measurement import mfu_estimate,parse_gpu_csv
        self.assertIsNone(mfu_estimate(training_tokens=100,elapsed_s=2))
        self.assertEqual(mfu_estimate(training_tokens=100,elapsed_s=2,
                                     flops_per_token=10,peak_flops_per_s=1000),.5)
        rows = parse_gpu_csv("GPU-abc, 90, 1234, 250\nGPU-def, 50, 2000, [N/A]\n")
        self.assertEqual(rows[0]["utilization_pct"],90)
        self.assertIsNone(rows[1]["power_w"])


if __name__ == "__main__":
    unittest.main()
