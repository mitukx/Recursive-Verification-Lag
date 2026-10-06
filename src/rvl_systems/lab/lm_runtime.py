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
                 max_policy_lag=4, capacity=4, control=None):
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
        from .control import ControlConfig, RVLControlPlane
        self.controller = RVLControlPlane(control or ControlConfig(audit_budget=32,refit_labels=4,risk_threshold=.3))
        self.labels = []
        self.estimated_shift = 0.0
        self.rvl_enabled = hasattr(verifier,"rescore")
        checkpoint = self.replay.checkpoint()
        if checkpoint:
            path,self.version = checkpoint
            state = self.torch.load(path,map_location=backend.resolved_device,weights_only=True)
            self.learner_model.load_state_dict(state["model"])
            self.trainer.optimizer.load_state_dict(state["optimizer"])
            self.torch.set_rng_state(state["rng"].cpu())
            self.labels = state.get("labels",[])
            self.estimated_shift = state.get("estimated_shift",0.0)
            if state.get("controller"):
                self.controller.restore(state["controller"])
            if self.rvl_enabled and state.get("verifier"):
                self.verifier.restore(state["verifier"])
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
                 "rng":self.torch.get_rng_state(),"labels":self.labels,
                 "controller":self.controller.state(),"estimated_shift":self.estimated_shift,
                 "verifier":self.verifier.state() if self.rvl_enabled else None}
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
            if self.rvl_enabled:
                samples = await self._intervene(rid,behavior_version,samples)
            metrics = await asyncio.to_thread(self.trainer.train_step,samples)
            self.version += 1
            self.estimated_shift += metrics.get("behavior_kl_estimate",0.0)
            self._save_checkpoint(rid)
            self.published = (self.version,self._weights())
            metrics.update({"version":self.version,"behavior_version":behavior_version,
                            "policy_lag":self.version-1-behavior_version,
                            "elapsed_s":time.perf_counter()-start,
                            "response_tokens":sum(s.generation.token_count for s in samples)})
            self.metrics.append(metrics)
            self.space.set()
            await asyncio.sleep(0)

    async def _intervene(self,rid,behavior_version,samples):
        from dataclasses import asdict, replace
        from types import SimpleNamespace
        snapshot = SimpleNamespace(version=self.version,logits={},estimated_shift=self.estimated_shift)
        samples = [self.verifier.rescore(s) for s in samples]
        records = [SimpleNamespace(trajectory_id=f"{rid}:{i}",policy_version=behavior_version,generation=s.generation,index=i)
                   for i,s in enumerate(samples)]
        rows = [(record,self.verifier.verdict(sample)) for record,sample in zip(records,samples)]
        selected = self.controller.select_audits(rows,self.verifier,snapshot,self.version)
        for record in selected:
            sample = samples[record.index]
            y = await self.verifier.audit(record.generation)
            self.labels.append({"generation":asdict(record.generation),"reward":y,"proxy":sample.reward})
            samples[record.index] = replace(sample,reward=y,metadata={**sample.metadata,"trusted":True})
        if self.controller.should_refit(len(self.labels)):
            self.verifier.fit(self.labels)
            self.controller.refitted(len(self.labels),snapshot)
            self.estimated_shift = 0.0
            # Relabel queued experiences without regenerating behavior tokens.
            for pending_id,raw in self.replay.db.execute("SELECT id,payload FROM groups WHERE status='ready'"):
                from ..types import Generation, VerifiedGeneration
                old = []
                for entry in json.loads(raw):
                    entry["generation"] = Generation(**entry["generation"])
                    old.append(self.verifier.rescore(VerifiedGeneration(**entry)))
                from .contracts import canonical
                self.replay.db.execute("UPDATE groups SET payload=? WHERE id=?",
                                       (canonical([asdict(s) for s in old]),pending_id))
        return [self.verifier.rescore(s) for s in samples]

    async def run(self,prompts,*,samples=4,seed=17):
        self.replay.bind({"model":self.backend.model_name,"prompts":list(prompts.items()),
                          "samples":samples,"seed":seed,"max_policy_lag":self.max_policy_lag,
                          "learning_rate":self.trainer.config.learning_rate,
                          "controller":vars(self.controller.config),"rvl_enabled":self.rvl_enabled})
        before = self._weights()
        async with asyncio.TaskGroup() as group:
            group.create_task(self._actor(prompts,samples,seed))
            group.create_task(self._learner())
        delta = sum(float((self.published[1][k]-v).abs().sum()) for k,v in before.items())
        report = {"version":self.version,"parameter_l1_change":delta,"history":self.metrics,
                  "trusted_audits":len(self.labels),"verifier_version":self.verifier.version,
                  "rvl_enabled":self.rvl_enabled,
                  "replay_counts":dict(self.replay.db.execute("SELECT status,COUNT(*) FROM groups GROUP BY status")),
                  "limitations":["one local HF inference actor","no GPU scale validation",
                                 "reward semantics supplied by caller","residual calibration is feature dependent"]}
        (self.root/"lm-report.json").write_text(json.dumps(report,indent=2,sort_keys=True))
        return report
