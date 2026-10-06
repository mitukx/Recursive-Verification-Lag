"""Real causal-LM asynchronous actor/learner reference, isolated model copies."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
from pathlib import Path

from ..hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
from .token_replay import TokenReplay


class AsyncHFLab:
    """Independent inference/learner models with durable grouped token replay.

    A single inference actor is intentional: HFLocalBackend seeds the global
    torch RNG. Learner dropout is disabled and no inference model is mutated
    during an in-flight group. Multi-node serving uses separate worker services.
    Reward functions are caller-supplied; no hidden tests leak via this class.
    """
    def __init__(self, root, backend, verifier, *, learning_rate=1e-5,
                 max_policy_lag=4, capacity=4):
        import fcntl
        self.root = Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock = (self.root/"driver.lock").open("a+")
        try:
            fcntl.flock(self.lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError("another LM driver owns this run")
        backend.ensure_loaded()
        self.backend = backend
        self.torch = backend._torch
        self.learner_model = copy.deepcopy(backend.model)
        self.trainer = HFCausalLMGRPOTrainer(self.learner_model,config=HFTTrainerConfig(learning_rate=learning_rate))
        self.verifier = verifier
        self.replay = TokenReplay(self.root/"token-replay.sqlite",capacity)
        self.version = 0
        self.actor_version = -1
        self.max_policy_lag = max_policy_lag
        self.metrics = []
        checkpoint = self.replay.checkpoint()
        if checkpoint:
            path,self.version = checkpoint
            state = self.torch.load(path,map_location=backend.resolved_device,weights_only=True)
            self.learner_model.load_state_dict(state["model"])
            self.trainer.optimizer.load_state_dict(state["optimizer"])
            self.torch.set_rng_state(state["rng"].cpu())
        self.published = (self.version,self._weights())
        self.ready,self.space = asyncio.Event(),asyncio.Event()
        self.actor_done = False

    def _weights(self):
        return {k:v.detach().cpu().clone() for k,v in self.learner_model.state_dict().items()}

    def close(self):
        import fcntl
        self.replay.close()
        fcntl.flock(self.lock,fcntl.LOCK_UN)
        self.lock.close()

    async def _actor(self,prompts,samples,seed):
        try:
            for i,(pid,prompt) in enumerate(prompts.items()):
                rid = f"group-{i:08d}"
                if self.replay.db.execute("SELECT 1 FROM groups WHERE id=?",(rid,)).fetchone():
                    continue
                version,weights = self.published
                if self.actor_version != version:
                    self.backend.model.load_state_dict(weights)
                    self.actor_version = version
                generations = await self.backend.generate(pid,prompt,n=samples,temperature=1.0,seed=seed+i)
                verified = list(await asyncio.gather(*(self.verifier.verify(g) for g in generations)))
                while not self.replay.put(rid,version,verified):
                    self.space.clear()
                    self.ready.set()
                    await self.space.wait()
                self.ready.set()
                await asyncio.sleep(0)
        finally:
            self.actor_done = True
            self.ready.set()

    def _save_checkpoint(self,rid):
        state = {"model":self._weights(),"optimizer":self.trainer.optimizer.state_dict(),
                 "rng":self.torch.get_rng_state()}
        temp = self.root/f"checkpoint-{self.version}.tmp"
        self.torch.save(state,temp)
        checksum = hashlib.sha256(temp.read_bytes()).hexdigest()
        final = self.root/f"checkpoint-{self.version}-{checksum}.pt"
        temp.replace(final)
        self.replay.commit(rid,final,self.version)

    async def _learner(self):
        while True:
            group = self.replay.next(self.version,self.max_policy_lag)
            if group is None:
                self.space.set()
                if self.actor_done:
                    break
                self.ready.clear()
                await self.ready.wait()
                continue
            rid,behavior_version,samples = group
            start = time.perf_counter()
            metrics = await asyncio.to_thread(self.trainer.train_step,samples)
            self.version += 1
            self._save_checkpoint(rid)
            self.published = (self.version,self._weights())
            metrics.update({"version":self.version,"behavior_version":behavior_version,
                            "policy_lag":self.version-1-behavior_version,
                            "elapsed_s":time.perf_counter()-start,
                            "response_tokens":sum(s.generation.token_count for s in samples)})
            self.metrics.append(metrics)
            self.space.set()
            await asyncio.sleep(0)

    async def run(self,prompts,*,samples=4,seed=17):
        self.replay.bind({"model":self.backend.model_name,"prompts":list(prompts.items()),
                          "samples":samples,"seed":seed,"max_policy_lag":self.max_policy_lag,
                          "learning_rate":self.trainer.config.learning_rate})
        before = self._weights()
        async with asyncio.TaskGroup() as group:
            group.create_task(self._actor(prompts,samples,seed))
            group.create_task(self._learner())
        delta = sum(float((self.published[1][k]-v).abs().sum()) for k,v in before.items())
        report = {"version":self.version,"parameter_l1_change":delta,"history":self.metrics,
                  "replay_counts":dict(self.replay.db.execute("SELECT status,COUNT(*) FROM groups GROUP BY status")),
                  "limitations":["one local HF inference actor","no GPU scale validation",
                                 "reward semantics supplied by caller","no RVL verifier fitting in this LM adapter yet"]}
        (self.root/"lm-report.json").write_text(json.dumps(report,indent=2,sort_keys=True))
        return report
