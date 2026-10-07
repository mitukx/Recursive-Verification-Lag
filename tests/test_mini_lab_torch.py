"""Optional real-model acceptance tests: no external downloads."""
import asyncio
import copy
import math
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    TORCH = True
except ImportError:
    TORCH = False

from src.rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
from src.rvl_systems.lab.lm_runtime import AsyncHFLab
from src.rvl_systems.lab.token_replay import TokenReplay
from src.rvl_systems.lab.judges import CalibratedMultiVerifier
from src.rvl_systems.lab.control import ControlConfig
from src.rvl_systems.types import Generation, VerifiedGeneration
from src.rvl_systems.verifier import FunctionalVerifier


@unittest.skipUnless(TORCH,"optional PyTorch/Transformers dependencies")
class TorchAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    def model(self):
        torch.manual_seed(17)
        return GPT2LMHeadModel(GPT2Config(vocab_size=32,n_positions=32,n_embd=16,
            n_layer=1,n_head=2,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0))

    def samples(self,model):
        model.eval()
        out = []
        for i in range(4):
            p,r = [1,2],[3+i,7+i]
            with torch.no_grad():
                z = model(torch.tensor([p+r])).logits[0,1:3]
                lp = torch.log_softmax(z.float(),-1).gather(1,torch.tensor(r)[:,None]).squeeze(1).tolist()
            out.append(VerifiedGeneration(Generation("p","prompt","response",sum(lp),2,0,
                {"prompt_token_ids":p,"response_token_ids":r,"response_token_logprobs":lp}),
                float(i%2),0,0))
        return out

    async def test_behavior_sampling_distribution_matches_same_version_policy(self):
        from src.rvl_systems.hf_backend import HFLocalBackend
        class Tokenizer:
            eos_token_id = None
            pad_token_id = 0
            def __call__(self,prompt,return_tensors):
                return {"input_ids":torch.tensor([[1,2]])}
            def decode(self,ids,skip_special_tokens):
                return " ".join(str(i) for i in ids)
        backend = HFLocalBackend("offline",max_new_tokens=3,device="cpu",precision="fp32")
        backend._model = self.model()
        backend._torch = torch
        backend._tokenizer = Tokenizer()
        backend._device,backend._resolved_precision = "cpu","fp32"
        # Match the Qwen setting that the first T4 pilot revealed as a mismatch,
        # plus a suppression processor and beam defaults that must not leak in.
        backend.model.generation_config.repetition_penalty = 1.1
        backend.model.generation_config.suppress_tokens = [3]
        backend.model.generation_config.num_beams = 2
        inherited = copy.deepcopy(backend.model.generation_config.to_dict())
        for temperature in (0.7, 1.0, 1.3):
            with self.subTest(temperature=temperature):
                generations = await backend.generate("p","prompt",n=2,temperature=temperature,seed=17)
                trainer = HFCausalLMGRPOTrainer(copy.deepcopy(backend.model))
                for g in generations:
                    sample = VerifiedGeneration(g,1,0,0)
                    row = trainer._sample_objective(sample,1)
                    self.assertLess(float(row[2].detach()),1e-5)
                    self.assertLess(float(row[3].detach()),1e-6)
                self.assertEqual(backend.model.generation_config.to_dict(), inherited)
        greedy = (await backend.generate("p","prompt",n=1,temperature=0.,seed=17))[0]
        with self.assertRaisesRegex(ValueError, "greedy"):
            trainer._sample_objective(VerifiedGeneration(greedy,1,0,0),1)

    async def test_additional_eos_shared_with_pad_keeps_its_action_logprob(self):
        from src.rvl_systems.hf_backend import HFLocalBackend
        from types import SimpleNamespace
        class Tokenizer:
            eos_token_id = 8
            pad_token_id = 2
            def __call__(self, prompt, return_tensors):
                return {"input_ids": torch.tensor([[1, 2]])}
            def decode(self, ids, skip_special_tokens):
                return " ".join(map(str, ids))
        model = self.model()
        model.generation_config.eos_token_id = [8, 2]
        # Force the alternate EOS in a controlled real-tensor protocol fixture.
        model.generate = lambda **kwargs: SimpleNamespace(
            sequences=torch.tensor([[1, 2, 2]]), scores=(torch.zeros(1,32),))
        backend = HFLocalBackend("offline", max_new_tokens=3, device="cpu", precision="fp32")
        backend._model, backend._torch, backend._tokenizer = model, torch, Tokenizer()
        backend._device, backend._resolved_precision = "cpu", "fp32"
        generation = (await backend.generate("p", "prompt", n=1, temperature=1., seed=17))[0]
        self.assertEqual(generation.metadata["response_token_ids"], [2])
        self.assertEqual(generation.token_count, 1)
        self.assertEqual(len(generation.metadata["response_token_logprobs"]), 1)

    async def test_coding_episodes_connect_terminal_credit_rvl_and_optimizer(self):
        # Scripted tool text exercises contracts. The real tiny tensor model
        # supplies token logprobs and receives optimizer updates; no claim that
        # the fixture learned Python syntax or semantic reasoning.
        import json
        from src.rvl_systems.lab.coding import CodingTask,IOTest
        from src.rvl_systems.lab.coding_lm import CodingAsyncHFLab
        model,owner = self.model(),self
        class Backend:
            model_name = "offline-coding-fixture"
            resolved_device = "cpu"
            _torch = torch
            def __init__(self):
                self.model = copy.deepcopy(model)
            def ensure_loaded(self):
                pass
            async def generate(self,pid,prompt,*,n,temperature,seed):
                await asyncio.sleep(.001)
                turn = int(pid.rsplit("-",1)[-1])
                index = (seed//7919)%4
                g = owner.samples(self.model)[index].generation
                if turn == 0:
                    source = "def f(x): return 2*x" if index%2 else "def f(x): return 0"
                    action = {"tool":"edit","source":source}
                else:
                    action = {"tool":"finish"}
                return [replace(g,prompt_id=pid,prompt=prompt,response=json.dumps(action))]
        from dataclasses import replace
        task = CodingTask("f","Implement f(x)=2*x","f",(IOTest((0,),0),),(IOTest((3,),6),))
        async def public(g):
            return 1.0
        async def proxy(g):
            return 1.0 if g.metadata["completed"] else 0.0
        async def trusted(g):
            return float("2*x" in g.metadata["final_source"] and g.metadata["completed"])
        def verifier():
            return CalibratedMultiVerifier({"proxy":proxy},trusted=trusted,
                                           feature=lambda g:g.metadata["final_source"])
        control = ControlConfig(audit_budget=4,refit_labels=2,risk_threshold=0)
        with tempfile.TemporaryDirectory() as tmp:
            lab = CodingAsyncHFLab(tmp,Backend(),verifier(),{"f":task},public,
                                   max_steps=2,learning_rate=1e-3,control=control)
            try:
                report = await asyncio.wait_for(lab.run_tasks(samples=4),30)
                self.assertEqual(report["version"],1)
                self.assertEqual(report["trusted_audits"],2)
                self.assertGreater(report["verifier_version"],0)
                self.assertGreater(report["parameter_l1_change"],0)
                raw = json.loads(lab.replay.db.execute("SELECT payload FROM groups").fetchone()[0])
                self.assertEqual(len(raw),8)
                self.assertTrue(all("episode_id" in s["metadata"] for s in raw))
                self.assertTrue(all(s["generation"]["response"].startswith("{") for s in raw))
                self.assertTrue((Path(tmp)/"coding-episodes"/"group-00000000-plan.json").exists())
            finally:
                lab.close()

    async def test_hf_trainer_transaction_restores_model_optimizer_and_rng(self):
        model = self.model()
        trainer = HFCausalLMGRPOTrainer(model,config=HFTTrainerConfig(learning_rate=1e-3))
        samples = self.samples(model)
        before = trainer.snapshot_training_state()
        torch.manual_seed(1234)
        trainer.train_step(samples)
        self.assertTrue(any(
            not torch.equal(value.cpu(), before["model"][key])
            for key, value in model.state_dict().items()
        ))
        trainer.restore_training_state(before)
        for key,value in model.state_dict().items():
            self.assertTrue(torch.equal(value.cpu(),before["model"][key]))
        restored = trainer.optimizer.state_dict()
        self.assertEqual(restored["param_groups"],before["optimizer"]["param_groups"])
        self.assertEqual(set(restored["state"]),set(before["optimizer"]["state"]))
        self.assertTrue(torch.equal(torch.get_rng_state(),before["cpu_rng"]))

    async def test_microbatch_gradient_matches_full_graph_reference(self):
        model = self.model()
        samples = self.samples(model)
        reference = copy.deepcopy(model)
        config = HFTTrainerConfig(learning_rate=1e-3)
        trainer = HFCausalLMGRPOTrainer(model,config=config)
        full = HFCausalLMGRPOTrainer(reference,config=config)
        advantages = [-1,1,-1,1]
        reference.eval()
        full.optimizer.zero_grad()
        loss = -torch.stack([full._sample_objective(s,a)[0] for s,a in zip(samples,advantages)]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(),config.max_grad_norm,error_if_nonfinite=True)
        full.optimizer.step()
        trainer.train_step(samples,advantages=advantages)
        for a,b in zip(model.parameters(),reference.parameters()):
            self.assertTrue(torch.allclose(a,b,atol=1e-6,rtol=1e-5))

    async def test_token_replay_immutable_behavior_and_atomic_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TokenReplay(Path(tmp)/"replay.sqlite",capacity=1)
            try:
                samples = self.samples(self.model())
                store.bind({"seed":17})
                with self.assertRaises(ValueError):
                    store.bind({"seed":19})
                self.assertTrue(store.put("one",0,samples))
                self.assertFalse(store.put("two",0,samples))
                rid,version,recovered = store.next(0,1)
                self.assertEqual(recovered,samples)
                store.commit(rid,Path(tmp)/"checkpoint.pt",1)
                self.assertEqual(store.checkpoint(),(str(Path(tmp)/"checkpoint.pt"),1))
                self.assertIsNone(store.next(1,1))
            finally:
                store.close()

    async def test_verification_aware_replay_requires_fresh_verifier_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TokenReplay(Path(tmp)/"replay.sqlite",capacity=2)
            try:
                generations = [s.generation for s in self.samples(self.model())]
                self.assertTrue(store.put_pending("pending",0,generations,now=10))
                self.assertIsNone(store.next(0,1,0,0))
                token,rid,version,recovered = store.claim_verification(0,1,0,0,now=11)
                self.assertEqual((rid,version),("pending",0))
                self.assertEqual(recovered,generations)
                verified = [VerifiedGeneration(g,float(i%2),.01,0) for i,g in enumerate(recovered)]
                store.complete_verification(rid,token,verified,now=12)
                ready = store.next(0,1,0,0)
                self.assertIsNotNone(ready)
                self.assertEqual(ready[2],verified)
                self.assertEqual(store.requeue_stale_verifications(1,0),1)
                self.assertIsNone(store.next(0,1,1,0))
                token,rid,version,recovered = store.claim_verification(0,1,1,0,now=13)
                refreshed = [VerifiedGeneration(g,float(i%2),.01,1) for i,g in enumerate(recovered)]
                store.complete_verification(rid,token,refreshed,now=14)
                self.assertEqual(store.next(0,1,1,0)[2],refreshed)
            finally:
                store.close()

    async def test_verification_lease_recovered_immediately_on_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"replay.sqlite"
            store = TokenReplay(path,capacity=2)
            generations = [s.generation for s in self.samples(self.model())]
            self.assertTrue(store.put_pending("pending",0,generations,now=10))
            token,rid,version,recovered = store.claim_verification(
                0,1,0,0,lease_s=3600,now=11
            )
            self.assertEqual(rid,"pending")
            store.close()
            store = TokenReplay(path,capacity=2)
            try:
                self.assertEqual(store.recover_verification_leases(),1)
                claimed = store.claim_verification(0,1,0,0,now=12)
                self.assertIsNotNone(claimed)
                self.assertNotEqual(claimed[0],token)
            finally:
                store.close()

    async def test_verification_failure_quarantines_without_training(self):
        model = self.model()
        owner = self
        class Backend:
            model_name = "offline-verification-failure"
            resolved_device = "cpu"
            _torch = torch
            def __init__(self):
                self.model = copy.deepcopy(model)
            def ensure_loaded(self):
                pass
            async def generate(self,pid,prompt,*,n,temperature,seed):
                return [s.generation for s in owner.samples(self.model)]
        class FailingVerifier:
            version = 0
            async def verify(self,generation):
                raise RuntimeError("injected verifier outage")
        with tempfile.TemporaryDirectory() as tmp:
            lab = AsyncHFLab(
                tmp,Backend(),FailingVerifier(),learning_rate=1e-3,
                max_verification_attempts=2,
            )
            try:
                report = await asyncio.wait_for(lab.run({"a":"prompt"},samples=4),30)
                self.assertEqual(report["version"],0)
                self.assertEqual(report["parameter_l1_change"],0)
                self.assertEqual(report["quarantined_verification_groups"],1)
                self.assertEqual(report["verification_backlog"],0)
                self.assertEqual(report["replay_counts"].get("consumed",0),0)
            finally:
                lab.close()

    async def test_verifier_fleet_scores_groups_concurrently(self):
        model = self.model()
        owner = self
        class Backend:
            model_name = "offline-verifier-fleet"
            resolved_device = "cpu"
            _torch = torch
            def __init__(self):
                self.model = copy.deepcopy(model)
            def ensure_loaded(self):
                pass
            async def generate(self,pid,prompt,*,n,temperature,seed):
                await asyncio.sleep(.001)
                return [s.generation for s in owner.samples(self.model)]
        class SlowVerifier:
            def __init__(self):
                self.version = 0
                self.active = 0
                self.peak = 0
            async def verify(self,generation):
                self.active += 1
                self.peak = max(self.peak,self.active)
                try:
                    await asyncio.sleep(.02)
                    return VerifiedGeneration(generation,1.0,.02,self.version)
                finally:
                    self.active -= 1
        with tempfile.TemporaryDirectory() as tmp:
            verifier = SlowVerifier()
            lab = AsyncHFLab(
                tmp,Backend(),verifier,learning_rate=1e-3,
                verification_workers=2,capacity=8,
            )
            try:
                report = await asyncio.wait_for(
                    lab.run({f"p{i}":"prompt" for i in range(4)},samples=4),30
                )
                self.assertGreaterEqual(verifier.peak,2)
                self.assertEqual(report["replay_counts"].get("consumed",0),4)
                self.assertEqual(report["verification_backlog"],0)
                self.assertTrue(all(
                    row["worker_id"] in (0,1)
                    for row in report["verification_history"]
                ))
            finally:
                lab.close()

    async def test_async_isolated_models_real_update_and_resume(self):
        model = self.model()
        owner = self
        class Backend:
            model_name = "offline-random-gpt2"
            resolved_device = "cpu"
            _torch = torch
            def __init__(self):
                self.model = copy.deepcopy(model)
            def ensure_loaded(self):
                pass
            async def generate(self,pid,prompt,*,n,temperature,seed):
                await asyncio.sleep(.005)
                return [s.generation for s in owner.samples(self.model)]
        with tempfile.TemporaryDirectory() as tmp:
            async def proxy(g):
                return 1.0
            async def trusted(g):
                return float(g.metadata["response_token_ids"][0]%2)
            def make_verifier():
                return CalibratedMultiVerifier({"public":proxy},trusted=trusted,
                    feature=lambda g:str(g.metadata["response_token_ids"][0]%2))
            control = ControlConfig(audit_budget=4,refit_labels=2,risk_threshold=0)
            lab = AsyncHFLab(tmp,Backend(),make_verifier(),
                             learning_rate=1e-3,control=control)
            try:
                first = await asyncio.wait_for(lab.run({"a":"prompt","b":"prompt"},samples=4),30)
                self.assertGreater(first["parameter_l1_change"],0)
                self.assertIsNot(lab.backend.model,lab.learner_model)
                self.assertEqual(first["version"],2)
                self.assertEqual(first["trusted_audits"],4)
                self.assertGreater(first["verifier_version"],0)
                for s in lab.metrics:
                    self.assertGreaterEqual(s["policy_lag"],0)
            finally:
                lab.close()
            resumed = AsyncHFLab(tmp,Backend(),make_verifier(),learning_rate=1e-3,control=control)
            try:
                second = await asyncio.wait_for(resumed.run({"a":"prompt","b":"prompt"},samples=4),30)
                self.assertEqual(second["version"],2)
                self.assertEqual(second["parameter_l1_change"],0)
            finally:
                resumed.close()


if __name__ == "__main__":
    unittest.main()
