from __future__ import annotations

import asyncio
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .agent import Snapshot, TabularLearner, WeightRegistry, rollout, trusted_reward
from .contracts import Task, digest
from .control import ControlConfig, Curriculum, RVLControlPlane
from .store import ReplayStore
from .verification import VerifierEnsemble


@dataclass(frozen=True)
class LabConfig:
    episodes: int = 256
    actors: int = 4
    batch_size: int = 8
    capacity: int = 32
    max_policy_lag: int = 16
    max_steps: int = 24
    episode_deadline_s: float = 7200
    tool_latency_s: float = 0.001
    seed: int = 17
    learning_rate: float = 0.12
    attack_every: int = 4
    distill: bool = True
    deterministic: bool = False

    def __post_init__(self):
        if min(self.episodes,self.actors,self.batch_size,self.capacity,self.max_steps,self.attack_every) <= 0:
            raise ValueError("invalid lab size")
        if self.capacity < self.batch_size or self.max_policy_lag < 0:
            raise ValueError("invalid backpressure/staleness configuration")
        if self.episode_deadline_s <= 0 or self.tool_latency_s < 0 or self.learning_rate <= 0:
            raise ValueError("invalid runtime configuration")
        if self.deterministic and self.actors != 1:
            raise ValueError("deterministic execution requires one actor")


def percentile(values, q):
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs)-1)*q
    i = int(pos)
    return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(pos-i)


class MiniLab:
    """Concurrent actor tasks and independent learner; no per-round barrier.

    The reference learner is CPU/tabular. SQLite is a single-machine store.
    Network serving and distributed tensor learning have separate acceptance
    tests; they do not silently make this a multi-node training service.
    """
    def __init__(self, root, config=None, control=None):
        self.root = Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.cfg = config or LabConfig()
        # Single control-plane owner; other processes may use ReplayStore.
        import fcntl
        self._lock = (self.root/"driver.lock").open("a+")
        try:
            fcntl.flock(self._lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise RuntimeError("another lab driver owns this run")
        self.store = ReplayStore(self.root/"replay.sqlite",self.cfg.capacity)
        self.learner = TabularLearner(self.cfg.learning_rate)
        self.verifier = VerifierEnsemble()
        self.controller = RVLControlPlane(control or ControlConfig())
        self.curriculum = Curriculum()
        self.registry = WeightRegistry(self.root/"weights")
        self.next_index = self.batches = self.tokens = 0
        self.latencies, self.metrics = [], []
        self.backpressure = self.done_actors = 0
        self.wakeup, self.progress = asyncio.Event(), asyncio.Event()
        self._reserved = set()
        self.serving = self.learner.snapshot()
        state = self.store.load_state("lab")
        if state:
            expected = {**asdict(self.cfg), "episodes": None}
            if state["config"] != expected or state["control_config"] != asdict(self.controller.config):
                self.close()
                raise ValueError("resume configuration mismatch")
            for obj,key in ((self.learner,"learner"),(self.verifier,"verifier"),
                            (self.controller,"controller"),(self.curriculum,"curriculum")):
                obj.restore(state[key])
            self.next_index,self.batches,self.tokens = state["next_index"],state["batches"],state["tokens"]
            self.metrics = state["metrics"]
            s = state["serving"]
            self.serving = Snapshot(s["version"],{k:tuple(v) for k,v in s["logits"].items()})
        self.registry.publish(self.serving)
        # Owner lock makes abandoned local learner leases safely reclaimable.
        self.store.db.execute("UPDATE replay SET status='ready',lease_token=NULL WHERE status='leased'")

    def close(self):
        import fcntl
        self.store.close()
        fcntl.flock(self._lock,fcntl.LOCK_UN)
        self._lock.close()

    def _state(self):
        return {"config":{**asdict(self.cfg),"episodes":None},
                "control_config":asdict(self.controller.config),
                "learner":self.learner.state(),"verifier":self.verifier.state(),
                "controller":self.controller.state(),"curriculum":self.curriculum.state(),
                "serving":asdict(self.serving),
                "next_index":self.next_index,"batches":self.batches,"tokens":self.tokens,
                "metrics":self.metrics}

    def _reserve(self):
        for key,payload in self.store.db.execute("SELECT key,payload FROM state WHERE key LIKE 'job-%' ORDER BY key"):
            raw = json.loads(payload)
            if not raw["done"] and key not in self._reserved:
                self._reserved.add(key)
                return key,raw
        if self.next_index >= self.cfg.episodes:
            return None
        i = self.next_index
        self.next_index += 1
        task = self.curriculum.generate(i,self.cfg.seed+i*1009)
        key = f"job-{i:08d}"
        raw = {"index":i,"task":asdict(task),"snapshot":asdict(self.registry.active),"done":False}
        self.store.db.execute("BEGIN IMMEDIATE")
        try:
            self.store.save_state(key,raw)
            self.store.save_state("lab",self._state())
            self.store.db.execute("COMMIT")
        except BaseException:
            self.store.db.execute("ROLLBACK")
            raise
        self._reserved.add(key)
        return key,raw

    async def _actor(self, actor_id):
        try:
            while True:
                job = self._reserve()
                if job is None:
                    break
                key,raw = job
                t = dict(raw["task"])
                t["public_inputs"] = tuple(t["public_inputs"])
                task = Task(**t)
                s = raw["snapshot"]
                snapshot = Snapshot(s["version"], {k:tuple(v) for k,v in s["logits"].items()})
                i = raw["index"]
                trajectory = await rollout(task,snapshot,self.cfg.seed+i*7919,
                    f"episode-{i:08d}",max_steps=self.cfg.max_steps,
                    deadline_s=self.cfg.episode_deadline_s,tool_latency_s=self.cfg.tool_latency_s,
                    checkpoint=self.root/"episodes"/f"{i:08d}.json")
                verdict = self.verifier.score(trajectory)
                while not self.store.put(trajectory,verdict):
                    if self.store.db.execute("SELECT 1 FROM replay WHERE id=?",(trajectory.trajectory_id,)).fetchone():
                        break  # Crash after insert but before marking job done.
                    self.backpressure += 1
                    self.progress.clear()
                    self.wakeup.set()
                    await self.progress.wait()
                raw["done"] = True
                self.store.save_state(key,raw)
                self._reserved.remove(key)
                self.latencies.append(trajectory.elapsed_s)
                self.wakeup.set()
                if self.cfg.deterministic and (i+1)%self.cfg.batch_size == 0:
                    while True:
                        status = self.store.db.execute("SELECT status FROM replay WHERE id=?",
                                                       (trajectory.trajectory_id,)).fetchone()[0]
                        if status not in ("ready","leased"):
                            break
                        self.progress.clear()
                        await self.progress.wait()
        finally:
            self.done_actors += 1
            self.wakeup.set()

    def evaluate(self, episodes=48):
        from .agent import AffineToolEnvironment
        import random
        snapshot = self.learner.snapshot()
        values = []
        for i in range(episodes):
            task = Task(f"eval-{i}",i%3,(i%3+1)*(-1 if i%2 else 1),10+i,split="eval")
            env,rng = AffineToolEnvironment(task),random.Random(100000+i)
            for turn in range(self.cfg.max_steps):
                a = rng.choices(range(7),weights=snapshot.probabilities(env.context(turn)))[0]
                env.step(a)
                if env.done:
                    break
            values.append(trusted_reward(task,env.program,env.done))
        return sum(values)/len(values)

    def _train_batch(self, claimed):
        snapshot = self.learner.snapshot()
        rows = [(t,self.verifier.score(t) if not v.trusted else v) for _,t,v in claimed]
        audited = {}
        for t in self.controller.select_audits(rows,self.verifier,snapshot,self.batches):
            verdict = self.verifier.audit(t)
            audited[t.trajectory_id] = verdict
            self.store.trusted_label(t.trajectory_id,verdict.reward)
            self.store.relabel(t.trajectory_id,verdict)
            self.curriculum.observe(t,verdict.reward)
        # Red-team search consumes the SAME trusted budget as ordinary auditing.
        if self.batches%self.cfg.attack_every == 0 and self.controller.spent < self.controller.config.audit_budget and self.controller.config.mode != "never":
            attack = self.curriculum.attack(claimed[0][1].task,snapshot,self.batches,self.verifier)
            verdict = self.verifier.audit(attack)
            self.controller.spent += 1
            attacks = self.store.load_state("attack_labels") or []
            attacks.append({"trajectory":attack.to_dict(),"reward":verdict.reward})
            self.store.save_state("attack_labels",attacks)
            self.curriculum.observe_attack(verdict.scores[0],verdict.reward)
            self.store.event("attack",id=attack.trajectory_id,proxy=verdict.scores[0],trusted=verdict.reward)
        from .contracts import Trajectory
        labels = self.store.labeled()
        labels += [(Trajectory.from_dict(a["trajectory"]),a["reward"])
                   for a in (self.store.load_state("attack_labels") or [])]
        refit = self.controller.should_refit(len(labels))
        if refit:
            self.verifier.fit(labels)
            self.controller.refitted(len(labels),snapshot)
            for t in self.controller.reevaluate_ids(self.store.candidates(),self.verifier,snapshot):
                self.store.relabel(t.trajectory_id,self.verifier.score(t))
                self.store.event("reevaluate",id=t.trajectory_id,verifier_version=self.verifier.version)
        batch = []
        for _,t,v in claimed:
            verdict = audited.get(t.trajectory_id)
            if verdict is None:
                verdict = v if v.trusted else self.verifier.score(t)
            self.store.relabel(t.trajectory_id,verdict)
            batch.append((t,verdict.reward))
        train = self.learner.train(batch)
        if self.cfg.distill:
            self.learner.distill([t for t,y in labels if y == 1][-8:])
        self.batches += 1
        self.tokens += sum(len(t.steps) for _,t,_ in claimed)
        evaluation = self.evaluate()
        previous = self.store.load_state("best")
        accepted = previous is None or evaluation >= previous["reward"]-0.15
        if accepted and (previous is None or evaluation > previous["reward"]):
            # Deep JSON copy is persisted before further parameter updates.
            self.store.save_state("best",{"reward":evaluation,"learner":self.learner.state()})
        if accepted:
            self.serving = self.learner.snapshot()
        self.metrics.append({"batch":self.batches,"version":self.learner.version,
            "eval_reward":evaluation,"mean_training_reward":statistics.fmean(y for _,y in batch),
            "audits":self.controller.spent,"verifier_version":self.verifier.version,
            "refit":refit,"checkpoint_accepted":accepted,
            "max_policy_lag":max(snapshot.version-t.policy_version for _,t,_ in claimed),**train})
        self.store.event("learn",**self.metrics[-1])
        for token,t,_ in claimed:
            self.store.finish(t.trajectory_id,token)
        self.store.save_state("lab",self._state())
        return accepted

    async def _learner_loop(self):
        while True:
            ready = self.store.counts().get("ready",0)
            done = self.done_actors == self.cfg.actors
            if ready < self.cfg.batch_size and not done:
                self.wakeup.clear()
                await self.wakeup.wait()
                continue
            claimed = self.store.claim(self.cfg.batch_size,self.learner.version,self.cfg.max_policy_lag)
            if not claimed:
                self.progress.set()
                if done:
                    break
                self.wakeup.clear()
                await self.wakeup.wait()
                continue
            self.store.db.execute("BEGIN IMMEDIATE")
            try:
                accepted = self._train_batch(claimed)
                self.store.db.execute("COMMIT")
            except BaseException:
                self.store.db.execute("ROLLBACK")
                raise
            if accepted:
                self.registry.publish(self.serving)
            self.progress.set()
            await asyncio.sleep(0)

    async def run(self):
        start = time.perf_counter()
        initial = self.evaluate()
        async with asyncio.TaskGroup() as group:
            group.create_task(self._learner_loop())
            for i in range(self.cfg.actors):
                group.create_task(self._actor(i))
        elapsed = time.perf_counter()-start
        self.store.save_state("lab",self._state())
        state = self._state()
        semantic = {k:v for k,v in state.items() if k != "config"}
        def normalize(value):
            if isinstance(value,float):
                return round(value,10)
            if isinstance(value,dict):
                return {k:normalize(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):
                return [normalize(v) for v in value]
            return value
        result = {"schema":1,"system":"cpu-affine-tool-lab","config":asdict(self.cfg),
                  "control":asdict(self.controller.config),"initial_eval_reward":initial,
                  "final_eval_reward":self.evaluate(),"batches":self.batches,
                  "trusted_calls":self.verifier.audits,"exploits_detected":self.verifier.exploits,
                  "attack_successes":self.curriculum.attack_successes,
                  "replay_counts":self.store.counts(),"backpressure_waits":self.backpressure,
                  "elapsed_s":elapsed,"tool_actions_per_s":self.tokens/max(elapsed,1e-9),
                  "latency_s":{"p50":percentile(self.latencies,.5),
                               "p95":percentile(self.latencies,.95),"p99":percentile(self.latencies,.99)},
                  "semantic_sha256":digest(normalize(semantic)),
                  "semantic_float_precision":10,"history":self.metrics,
                  "hardware":{"platform":platform.platform(),"python":platform.python_version(),
                              "gpu_utilization":None,"mfu":None},
                  "limitations":["restricted DSL and tabular policy","single-machine SQLite",
                                 "tool actions are not LM tokens","no GPU measurements"]}
        (self.root/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
        return result
