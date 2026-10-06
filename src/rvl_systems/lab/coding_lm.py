"""Asynchronous model-driven coding episodes -> verification-aware RL training."""
from __future__ import annotations

import asyncio
from dataclasses import replace

from .lm_runtime import AsyncHFLab
from .tool_agent import CodingToolAgent


def episode_advantages(samples):
    """Normalize terminal returns across episodes, then balance episode length."""
    import math
    groups = {}
    for sample in samples:
        episode = sample.metadata["episode_id"]
        groups.setdefault(episode,[]).append(sample.reward)
    rewards = {episode:sum(values)/len(values) for episode,values in groups.items()}
    mean = sum(rewards.values())/len(rewards)
    std = math.sqrt(sum((y-mean)**2 for y in rewards.values())/len(rewards)+1e-6)
    return [((rewards[s.metadata["episode_id"]]-mean)/std) *
            len(samples)/(len(groups)*len(groups[s.metadata["episode_id"]])) for s in samples]


class CodingAsyncHFLab(AsyncHFLab):
    def __init__(self,root,backend,verifier,tasks,public_grader,*,max_steps=128,deadline_s=7200,**kwargs):
        super().__init__(root,backend,verifier,**kwargs)
        self.tasks = tasks
        from dataclasses import asdict
        self.task_manifest = {"tasks":{k:asdict(t) for k,t in tasks.items()},
                              "max_steps":max_steps,"deadline_s":deadline_s}
        self.agent = CodingToolAgent(public_grader,max_steps=max_steps,deadline_s=deadline_s)

    async def _actor(self,prompts,samples,seed):
        try:
            for i,(task_id,_) in enumerate(prompts.items()):
                await self._await_generation_admission()
                rid = f"group-{i:08d}"
                if self.replay.db.execute("SELECT 1 FROM groups WHERE id=?",(rid,)).fetchone():
                    continue
                import hashlib
                import json
                from .contracts import canonical
                plan_path = self.root/"coding-episodes"/f"{rid}-plan.json"
                if plan_path.exists():
                    plan = json.loads(plan_path.read_text())
                    path = self.root/plan["artifact"]
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=plan["sha256"]:
                        raise ValueError("pinned episode weights corrupted")
                    version = plan["version"]
                    weights = self.torch.load(path,map_location="cpu",weights_only=True)
                else:
                    version,weights = self.published
                    temp = self.root/f"actor-v{version}.tmp"
                    self.torch.save(weights,temp)
                    checksum = hashlib.sha256(temp.read_bytes()).hexdigest()
                    final = self.root/f"actor-v{version}-{checksum}.pt"
                    temp.replace(final)
                    plan_path.parent.mkdir(parents=True,exist_ok=True)
                    temp_plan = plan_path.with_suffix(".tmp")
                    temp_plan.write_text(canonical({"version":version,"artifact":final.name,"sha256":checksum}))
                    temp_plan.replace(plan_path)
                if self.actor_version != version:
                    self.backend.model.load_state_dict(weights)
                    self.actor_version = version
                owner = self
                class Lease:
                    def __init__(self):
                        self.version = version
                    async def generate(self,request):
                        rows = await owner.backend.generate(
                            request.prompt_id,request.prompt,
                            n=1,temperature=request.temperature,seed=request.seed
                        )
                        return [replace(g,metadata={**g.metadata,"policy_version":version}) for g in rows]
                generations = []
                task = self.tasks[task_id]
                for j in range(samples):
                    episode = await self.agent.run(
                        task,Lease(),f"{rid}-episode-{j}",
                        seed=seed+i*1009+j*7919,
                        journal=self.root/"coding-episodes"/f"{rid}-{j}.json",
                    )
                    for g in episode.generations:
                        generations.append(replace(
                            g,
                            metadata={
                                **g.metadata,
                                "final_source":episode.source,
                                "coding_task_id":task_id,
                                "episode_id":episode.episode_id,
                                "completed":episode.completed,
                            },
                        ))
                if not generations:
                    raise RuntimeError("coding episode produced no model actions")
                while not self.replay.put_pending(rid,version,generations):
                    self.space.clear()
                    self.ready.set()
                    await self.space.wait()
                self.ready.set()
                await asyncio.sleep(0)
        finally:
            self.actor_done = True
            self.ready.set()

    async def run_tasks(self,*,samples=4,seed=17):
        return await self.run({key:t.prompt for key,t in self.tasks.items()},samples=samples,seed=seed)
