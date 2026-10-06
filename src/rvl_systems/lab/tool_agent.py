"""Bounded, resumable model-driven coding/tool-use episodes."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ..rollout import RolloutRequest
from ..types import Generation, VerifiedGeneration
from .contracts import canonical, digest


@dataclass(frozen=True)
class AgentEpisode:
    episode_id: str
    policy_version: int
    generations: tuple[Generation,...]
    source: str
    completed: bool
    elapsed_s: float


class CodingToolAgent:
    """JSON inspect/edit/public_test/finish loop with evaluator-owned tests.

    A serving lease pins one version for the ENTIRE episode. Journal entries
    contain public tool results only. Hidden tests are called after completion
    by a trusted evaluator, never included in model context.
    """
    def __init__(self,public_grader,*,max_steps=128,deadline_s=7200,max_context_chars=64000):
        if min(max_steps,deadline_s,max_context_chars) <= 0:
            raise ValueError("invalid agent budgets")
        self.public_grader = public_grader
        self.max_steps,self.deadline_s,self.max_context_chars = max_steps,deadline_s,max_context_chars

    async def run(self,task,lease,episode_id,*,seed=17,journal=None):
        start = time.perf_counter()
        identity = digest({"task":task.task_id,"prompt":task.prompt,"version":lease.version,"seed":seed})
        history,generations,source,done = [],[],"",False
        if journal is not None and Path(journal).exists():
            saved = json.loads(Path(journal).read_text())
            if saved["identity"] != identity:
                raise ValueError("tool journal provenance mismatch")
            history,source,done = saved["history"],saved["source"],saved["done"]
            generations = [Generation(**r) for r in saved["generations"]]
        async with asyncio.timeout(self.deadline_s):
            for turn in range(len(generations),self.max_steps):
                if done:
                    break
                prompt = (task.prompt+'\nReturn one JSON tool call: {"tool":"inspect"}, '
                          '{"tool":"edit","source":"Python code"}, {"tool":"public_test"}, or {"tool":"finish"}.\n'
                          "Public tool history:\n"+canonical(history))
                if len(prompt)>self.max_context_chars:
                    break  # Explicit context budget; do not silently truncate provenance.
                request = RolloutRequest(f"{episode_id}-turn-{turn}",prompt,samples=1,
                                         seed=seed+turn,deadline_s=self.deadline_s)
                rows = await lease.generate(request)
                if len(rows)!=1:
                    raise ValueError("agent requires one completion per turn")
                g = rows[0]
                if g.metadata.get("policy_version") != lease.version:
                    raise ValueError("agent received mismatched policy version")
                generations.append(g)
                try:
                    action = json.loads(g.response)
                    tool = action["tool"]
                    if tool=="inspect":
                        observation = {"source":source,"task":task.prompt}
                    elif tool=="edit":
                        candidate_source = action["source"]
                        if not isinstance(candidate_source,str) or len(candidate_source)>32000:
                            raise ValueError("invalid source size")
                        source = candidate_source
                        observation = {"edited":True}
                    elif tool=="public_test":
                        candidate = Generation(task.task_id,task.prompt,source,0,0,0)
                        observation = {"public_reward":await self.public_grader(candidate)}
                    elif tool=="finish":
                        done = True
                        observation = {"submitted":True}
                    else:
                        raise ValueError("unsupported tool")
                except (ValueError,KeyError,TypeError):
                    observation = {"tool_error":"invalid JSON tool call"}
                history.append({"turn":turn,"action":g.response,"observation":observation})
                if journal is not None:
                    path = Path(journal)
                    path.parent.mkdir(parents=True,exist_ok=True)
                    temp = path.with_suffix(".tmp")
                    temp.write_text(canonical({"identity":identity,"history":history,
                        "generations":[asdict(g) for g in generations],"source":source,"done":done}))
                    temp.replace(path)
        return AgentEpisode(episode_id,lease.version,tuple(generations),source,done,
                            time.perf_counter()-start)

    @staticmethod
    def training_turns(episode,terminal_reward,verifier_version):
        # Train the model's sampled JSON actions, not edited code that wasn't
        # sampled as a standalone response. Caller supplies episode advantages.
        return [VerifiedGeneration(g,terminal_reward,0,verifier_version,
                {"episode_id":episode.episode_id,"credit":"terminal_return"})
                for g in episode.generations]
