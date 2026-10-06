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

    async def test_calibrated_critic_cannot_override_public_failure(self):
        from dataclasses import asdict
        from src.rvl_systems.lab.judges import CalibratedMultiVerifier
        async def failed(g):
            return 0.0
        v = CalibratedMultiVerifier({"executable":failed},trusted=failed,feature=lambda g:"collision")
        v.fit([{"generation":asdict(self.generation()),"reward":1,"proxy":1}]*100)
        result = await v.verify(self.generation())
        self.assertGreater(result.metadata["critic_score"],.9)
        self.assertEqual(result.reward,0)

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


class ToolAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_tool_loop_preserves_version_and_hides_trusted_tests(self):
        import json
        import tempfile
        from pathlib import Path
        from src.rvl_systems.lab.coding import CodingTask,IOTest
        from src.rvl_systems.lab.tool_agent import CodingToolAgent
        task = CodingTask("double","Implement double(x)=2*x","double",
                          (IOTest((0,),0),),(IOTest((-917,),-1834),))
        calls = [{"tool":"inspect"},{"tool":"edit","source":"def double(x): return 2*x"},
                 {"tool":"public_test"},{"tool":"finish"}]
        class Lease:
            version = 3
            def __init__(self):
                self.prompts = []
            async def generate(self,request):
                self.prompts.append(request.prompt)
                return [Generation(request.prompt_id,request.prompt,json.dumps(calls[len(self.prompts)-1]),
                                   -.1,4,0,{"policy_version":3})]
        async def public(g):
            return 1.0
        agent = CodingToolAgent(public)
        lease = Lease()
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp)/"episode.json"
            episode = await agent.run(task,lease,"episode",journal=journal)
            self.assertTrue(episode.completed)
            self.assertEqual(episode.policy_version,3)
            self.assertEqual(len(episode.generations),4)
            self.assertTrue(all("-917" not in p for p in lease.prompts))
            restored = await agent.run(task,lease,"episode",journal=journal)
            self.assertEqual(restored.generations,episode.generations)
            turns = agent.training_turns(episode,1,2)
            self.assertEqual(len(turns),4)
            self.assertTrue(all(s.generation.response.startswith("{") for s in turns))


class EpisodeCreditTests(unittest.TestCase):
    def test_terminal_credit_is_balanced_across_variable_length_episodes(self):
        from src.rvl_systems.lab.coding_lm import episode_advantages
        from src.rvl_systems.types import VerifiedGeneration
        g = Generation("p","prompt","action",-.1,1,0)
        samples = [VerifiedGeneration(g,1,0,0,{"episode_id":"good"}) for _ in range(3)]
        samples += [VerifiedGeneration(g,0,0,0,{"episode_id":"bad"})]
        advantages = episode_advantages(samples)
        self.assertAlmostEqual(sum(advantages[:3]),-advantages[3])
        self.assertGreater(advantages[0],0)
        self.assertLess(advantages[3],0)


class TokenServingTests(unittest.TestCase):
    def response(self):
        return {"model":"immutable-v3","prompt_token_ids":[1,2],
                "choices":[{"index":0,"text":"answer","token_ids":[3,4],
                            "logprobs":{"token_logprobs":[-.2,-.3]}}]}

    def test_remote_behavior_tokens_are_consumable_by_real_trainer(self):
        from src.rvl_systems.lab.token_serving import TokenServingBackend
        from src.rvl_systems.hf_trainer import HFCausalLMGRPOTrainer
        from src.rvl_systems.types import VerifiedGeneration
        backend = TokenServingBackend("http://localhost","immutable-v3")
        g = backend.parse(self.response(),"p","prompt",1,.1)[0]
        p,r,lp = HFCausalLMGRPOTrainer._metadata(VerifiedGeneration(g,1,0,0))
        self.assertEqual((p,r,lp),([1,2],[3,4],[-.2,-.3]))

    def test_vllm_per_choice_prompt_token_contract(self):
        from src.rvl_systems.lab.token_serving import TokenServingBackend
        raw = self.response()
        raw["choices"][0]["prompt_token_ids"] = raw.pop("prompt_token_ids")
        g = TokenServingBackend("http://localhost","immutable-v3").parse(raw,"p","prompt",1,.1)[0]
        self.assertEqual(g.metadata["prompt_token_ids"],[1,2])

    def test_identity_missing_prompt_ids_and_partial_logprobs_rejected(self):
        from src.rvl_systems.lab.token_serving import TokenServingBackend
        backend = TokenServingBackend("http://localhost","immutable-v3")
        cases = []
        raw = self.response()
        raw["model"] = "wrong-version"
        cases.append(raw)
        raw = self.response()
        del raw["prompt_token_ids"]
        cases.append(raw)
        raw = self.response()
        raw["choices"][0]["logprobs"]["token_logprobs"] = [None,-.3]
        cases.append(raw)
        for body in cases:
            with self.assertRaises(ValueError):
                backend.parse(body,"p","prompt",1,.1)


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
