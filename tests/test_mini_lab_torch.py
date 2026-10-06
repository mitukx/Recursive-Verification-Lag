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
        generations = await backend.generate("p","prompt",n=2,temperature=1.0,seed=17)
        trainer = HFCausalLMGRPOTrainer(copy.deepcopy(backend.model))
        for g in generations:
            sample = VerifiedGeneration(g,1,0,0)
            row = trainer._sample_objective(sample,1)
            self.assertLess(float(row[2]),1e-5)
            self.assertLess(float(row[3]),1e-6)

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
