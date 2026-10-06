"""Real causal-LM verification-aware asynchronous actor/verifier/learner reference."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from ..hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
from .token_replay import TokenReplay
from .verification_debt import VerificationDebtConfig, VerificationDebtController


class AsyncHFLab:
    """Independent inference, verification, and learner stages with durable replay.

    Generation is admitted to replay before reward computation. A separate
    verification worker turns immutable behavior trajectories into versioned
    reward-bearing samples. The learner consumes only groups that satisfy both
    policy-lag and verifier-lag bounds. Verifier refits are serialized against
    scoring, so a replay group never mixes verifier versions.

    The local reference still uses one HF inference actor because HFLocalBackend
    seeds the global torch RNG. Multi-node serving uses separate worker services.
    """

    def __init__(self, root, backend, verifier, *, learning_rate=1e-5,
                 max_policy_lag=4, max_verifier_lag=0, capacity=4, control=None,
                 verification_lease_s=30, max_verification_attempts=3,
                 verification_debt=None):
        import fcntl
        if min(max_policy_lag,max_verifier_lag) < 0:
            raise ValueError("lag bounds must be non-negative")
        if verification_lease_s <= 0 or max_verification_attempts <= 0:
            raise ValueError("invalid verification retry configuration")
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
        self.trainer = HFCausalLMGRPOTrainer(
            self.learner_model,
            config=HFTTrainerConfig(learning_rate=learning_rate),
        )
        self.verifier = verifier
        self.replay = TokenReplay(self.root/"token-replay.sqlite",capacity)
        self.replay.recover_verification_leases()
        self.version = 0
        self.actor_version = -1
        self.max_policy_lag = max_policy_lag
        self.max_verifier_lag = max_verifier_lag
        self.verification_lease_s = verification_lease_s
        self.max_verification_attempts = max_verification_attempts
        self.metrics = []
        self.verification_metrics = []
        self.verification_debt_controller = VerificationDebtController(
            verification_debt or VerificationDebtConfig()
        )
        self.verification_debt_history = []
        from .control import ControlConfig, RVLControlPlane
        self.controller = RVLControlPlane(
            control or ControlConfig(audit_budget=32,refit_labels=4,risk_threshold=.3)
        )
        self.labels = []
        self.estimated_shift = 0.0
        self.rvl_enabled = hasattr(verifier,"rescore")
        checkpoint = self.replay.checkpoint()
        if checkpoint:
            path,self.version = checkpoint
            state = self.torch.load(
                path,map_location=backend.resolved_device,weights_only=True
            )
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
        self.verifier_done = False
        self.verifier_lock = asyncio.Lock()
        self.verification_progress = asyncio.Event()

    def _weights(self):
        return {
            k:v.detach().cpu().clone()
            for k,v in self.learner_model.state_dict().items()
        }

    def close(self):
        import fcntl
        self.replay.close()
        fcntl.flock(self.lock,fcntl.LOCK_UN)
        self.lock.close()

    def _verification_debt_assessment(self):
        signals = self.replay.verification_debt_signals(
            self.version,self.verifier.version
        )
        return self.verification_debt_controller.assess(signals)

    async def _await_generation_admission(self):
        while True:
            assessment = self._verification_debt_assessment()
            self.verification_debt_history.append({
                "policy_version":self.version,
                "verifier_version":self.verifier.version,
                "score":assessment.score,
                "level":assessment.level,
                "action":assessment.action,
                "signals":vars(assessment.signals),
            })
            if assessment.action == "admit_generation":
                return
            self.verification_progress.clear()
            assessment = self._verification_debt_assessment()
            if assessment.action == "admit_generation":
                continue
            await self.verification_progress.wait()

    async def _actor(self,prompts,samples,seed):
        try:
            for i,(pid,prompt) in enumerate(prompts.items()):
                await self._await_generation_admission()
                rid = f"group-{i:08d}"
                if self.replay.db.execute(
                    "SELECT 1 FROM groups WHERE id=?",(rid,)
                ).fetchone():
                    continue
                version,weights = self.published
                if self.actor_version != version:
                    self.backend.model.load_state_dict(weights)
                    self.actor_version = version
                generations = await self.backend.generate(
                    pid,prompt,n=samples,temperature=1.0,seed=seed+i
                )
                while not self.replay.put_pending(rid,version,generations):
                    self.space.clear()
                    self.ready.set()
                    await self.space.wait()
                self.ready.set()
                await asyncio.sleep(0)
        finally:
            self.actor_done = True
            self.ready.set()

    @staticmethod
    def _inherit_generation_metadata(sample):
        metadata = dict(sample.metadata)
        for key in ("episode_id","coding_task_id","completed","policy_version"):
            if key in sample.generation.metadata:
                metadata.setdefault(key,sample.generation.metadata[key])
        return replace(sample,metadata=metadata)

    async def _verification_worker(self):
        while True:
            claim = self.replay.claim_verification(
                self.version,self.max_policy_lag,self.verifier.version,
                self.max_verifier_lag,lease_s=self.verification_lease_s,
            )
            if claim is None:
                if self.actor_done and self.replay.active_count() == 0:
                    self.verifier_done = True
                    self.ready.set()
                    return
                self.ready.clear()
                await self.ready.wait()
                continue
            token,rid,behavior_version,generations = claim
            started = time.perf_counter()
            try:
                async with self.verifier_lock:
                    verifier_version_before = self.verifier.version
                    results = list(await asyncio.gather(
                        *(self.verifier.verify(g) for g in generations),
                        return_exceptions=True,
                    ))
                    failures = [x for x in results if isinstance(x,Exception)]
                    if failures:
                        raise RuntimeError(
                            f"verification group failed in {len(failures)} sample(s)"
                        ) from failures[0]
                    samples = [
                        self._inherit_generation_metadata(s) for s in results
                    ]
                    versions = {s.verifier_version for s in samples}
                    if versions != {verifier_version_before}:
                        raise RuntimeError(
                            "verification group mixed or changed verifier version"
                        )
                self.replay.complete_verification(rid,token,samples)
                meta = self.replay.verification_metadata(rid)
                elapsed = time.perf_counter()-started
                self.verification_metrics.append({
                    "group_id":rid,
                    "behavior_version":behavior_version,
                    "verifier_version":verifier_version_before,
                    "verification_latency_s":elapsed,
                    "queue_to_verified_s":max(
                        0.0,(meta["verified_at"] or 0)-meta["generated_at"]
                    ),
                    "verification_attempts":meta["verification_attempts"],
                })
                self.ready.set()
                self.verification_progress.set()
            except Exception:
                status = self.replay.fail_verification(
                    rid,token,max_attempts=self.max_verification_attempts
                )
                self.verification_metrics.append({
                    "group_id":rid,
                    "behavior_version":behavior_version,
                    "verifier_version":self.verifier.version,
                    "verification_latency_s":time.perf_counter()-started,
                    "status":status,
                })
                self.ready.set()
                self.verification_progress.set()
            await asyncio.sleep(0)

    def _save_checkpoint(self,rid):
        state = {
            "model":self._weights(),
            "optimizer":self.trainer.optimizer.state_dict(),
            "rng":self.torch.get_rng_state(),
            "labels":self.labels,
            "controller":self.controller.state(),
            "estimated_shift":self.estimated_shift,
            "verifier":self.verifier.state() if self.rvl_enabled else None,
        }
        temp = self.root/f"checkpoint-{self.version}.tmp"
        self.torch.save(state,temp)
        checksum = hashlib.sha256(temp.read_bytes()).hexdigest()
        final = self.root/f"checkpoint-{self.version}-{checksum}.pt"
        temp.replace(final)
        self.replay.commit(rid,final,self.version)

    async def _learner(self):
        while True:
            group = self.replay.next(
                self.version,self.max_policy_lag,
                self.verifier.version,self.max_verifier_lag,
            )
            if group is None:
                self.space.set()
                if self.actor_done and self.replay.active_count() == 0:
                    break
                self.ready.clear()
                await self.ready.wait()
                continue
            rid,behavior_version,samples = group
            start = time.perf_counter()
            verifier_version_at_admission = min(
                s.verifier_version for s in samples
            )
            if self.rvl_enabled:
                async with self.verifier_lock:
                    samples = await self._intervene(
                        rid,behavior_version,samples
                    )
            if all("episode_id" in s.metadata for s in samples):
                from .coding_lm import episode_advantages
                metrics = await asyncio.to_thread(
                    self.trainer.train_step,
                    samples,
                    advantages=episode_advantages(samples),
                )
            else:
                metrics = await asyncio.to_thread(
                    self.trainer.train_step,samples
                )
            self.version += 1
            self.estimated_shift += metrics.get("behavior_kl_estimate",0.0)
            meta = self.replay.verification_metadata(rid)
            self._save_checkpoint(rid)
            self.published = (self.version,self._weights())
            current_verifier_version = self.verifier.version
            metrics.update({
                "version":self.version,
                "behavior_version":behavior_version,
                "policy_lag":self.version-1-behavior_version,
                "verifier_version":current_verifier_version,
                "verifier_version_at_admission":verifier_version_at_admission,
                "verifier_lag":current_verifier_version-verifier_version_at_admission,
                "verification_age_s":(
                    0.0 if meta["verified_at"] is None
                    else max(0.0,time.time()-meta["verified_at"])
                ),
                "verification_attempts":meta["verification_attempts"],
                "verification_backlog":self.replay.verification_backlog(
                    current_verifier_version,self.max_verifier_lag
                ),
                "elapsed_s":time.perf_counter()-start,
                "response_tokens":sum(
                    s.generation.token_count for s in samples
                ),
            })
            self.metrics.append(metrics)
            self.space.set()
            self.ready.set()
            self.verification_progress.set()
            await asyncio.sleep(0)

    def _rescore_current(self,sample):
        sample = self.verifier.rescore(sample)
        if sample.metadata.get("trusted") and sample.verifier_version != self.verifier.version:
            sample = replace(sample,verifier_version=self.verifier.version)
        return sample

    async def _intervene(self,rid,behavior_version,samples):
        from dataclasses import asdict
        from types import SimpleNamespace
        snapshot = SimpleNamespace(
            version=self.version,logits={},estimated_shift=self.estimated_shift
        )
        samples = [self._rescore_current(s) for s in samples]
        records = [
            SimpleNamespace(
                trajectory_id=f"{rid}:{i}",
                policy_version=behavior_version,
                generation=s.generation,
                index=i,
            )
            for i,s in enumerate(samples)
        ]
        rows = []
        seen_episodes = set()
        for record,sample in zip(records,samples):
            episode = sample.metadata.get("episode_id")
            if episode is not None and episode in seen_episodes:
                continue
            if episode is not None:
                seen_episodes.add(episode)
            rows.append((record,self.verifier.verdict(sample)))
        selected = self.controller.select_audits(
            rows,self.verifier,snapshot,self.version
        )
        for record in selected:
            sample = samples[record.index]
            y = await self.verifier.audit(record.generation)
            self.labels.append({
                "generation":asdict(record.generation),
                "reward":y,
                "proxy":sample.reward,
            })
            episode = sample.metadata.get("episode_id")
            targets = [record.index] if episode is None else [
                i for i,s in enumerate(samples)
                if s.metadata.get("episode_id")==episode
            ]
            for index in targets:
                samples[index] = replace(
                    samples[index],
                    reward=y,
                    metadata={**samples[index].metadata,"trusted":True},
                )
        if self.controller.should_refit(len(self.labels)):
            self.verifier.fit(self.labels)
            self.controller.refitted(len(self.labels),snapshot)
            self.estimated_shift = 0.0
        samples = [self._rescore_current(s) for s in samples]
        self.replay.rewrite_verified(rid,samples)
        if self.rvl_enabled:
            self.replay.requeue_stale_verifications(
                self.verifier.version,self.max_verifier_lag
            )
            self.ready.set()
        return samples

    async def run(self,prompts,*,samples=4,seed=17):
        self.replay.bind({
            "model":self.backend.model_name,
            "prompts":list(prompts.items()),
            "samples":samples,
            "seed":seed,
            "max_policy_lag":self.max_policy_lag,
            "learning_rate":self.trainer.config.learning_rate,
            "controller":vars(self.controller.config),
            "rvl_enabled":self.rvl_enabled,
            "precision":getattr(self.backend,"resolved_precision","unspecified"),
            "generation_limit":getattr(self.backend,"max_new_tokens",None),
            "task_manifest":getattr(self,"task_manifest",None),
        })
        self.replay.bind_verification({
            "max_verifier_lag":self.max_verifier_lag,
            "verification_debt_config":vars(self.verification_debt_controller.config),
            "verification_lease_s":self.verification_lease_s,
            "max_verification_attempts":self.max_verification_attempts,
            "verifier_type":type(self.verifier).__name__,
        })
        before = self._weights()
        async with asyncio.TaskGroup() as group:
            group.create_task(self._actor(prompts,samples,seed))
            group.create_task(self._verification_worker())
            group.create_task(self._learner())
        delta = sum(
            float((self.published[1][k]-v).abs().sum())
            for k,v in before.items()
        )
        counts = self.replay.counts()
        report = {
            "version":self.version,
            "parameter_l1_change":delta,
            "history":self.metrics,
            "verification_history":self.verification_metrics,
            "verification_debt_history":self.verification_debt_history,
            "verification_debt_config":vars(self.verification_debt_controller.config),
            "max_verification_debt":max(
                [row["score"] for row in self.verification_debt_history] or [0.0]
            ),
            "verification_debt_throttle_events":sum(
                row["action"]!="admit_generation"
                for row in self.verification_debt_history
            ),
            "trusted_audits":len(self.labels),
            "verifier_version":self.verifier.version,
            "rvl_enabled":self.rvl_enabled,
            "max_policy_lag":self.max_policy_lag,
            "max_verifier_lag":self.max_verifier_lag,
            "replay_counts":counts,
            "verification_backlog":self.replay.verification_backlog(
                self.verifier.version,self.max_verifier_lag
            ),
            "quarantined_verification_groups":counts.get("quarantined",0),
            "limitations":[
                "one local HF inference actor",
                "one local verification worker",
                "no GPU scale validation of verification-aware pipeline",
                "reward semantics supplied by caller",
                "residual calibration is feature dependent",
            ],
        }
        (self.root/"lm-report.json").write_text(
            json.dumps(report,indent=2,sort_keys=True)
        )
        return report
