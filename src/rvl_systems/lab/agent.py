from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from .contracts import Step, Task, Trajectory, canonical, digest


def softmax(logits):
    offset = max(logits)
    values = [math.exp(x - offset) for x in logits]
    total = sum(values)
    return [x / total for x in values]


@dataclass(frozen=True)
class Snapshot:
    version: int
    logits: dict[str, tuple[float, ...]]

    def probabilities(self, context):
        return softmax(self.logits.get(context, (0.0,) * 7))


class TabularLearner:
    """Clipped per-decision off-policy REINFORCE, not a frontier LM."""
    def __init__(self, learning_rate=0.12, clip_eps=0.2):
        self.logits = {}
        self.version = 0
        self.learning_rate = learning_rate
        self.clip_eps = clip_eps
        self.baselines = {}

    def snapshot(self):
        return Snapshot(self.version, {k: tuple(v) for k, v in self.logits.items()})

    def train(self, batch):
        # Compute all gradients against one fixed target policy before mutation.
        gradients, ratios, losses = {}, [], []
        for trajectory, reward in batch:
            key = str(trajectory.task.family)
            baseline = self.baselines.get(key, 0.5)
            advantage = max(-1.0, min(1.0, reward - baseline))
            for step in trajectory.steps:
                z = self.logits.get(step.context, [0.0] * 7)
                p = softmax(z)
                ratio = math.exp(max(-20.0, min(20.0, math.log(p[step.action]) - step.behavior_logprob)))
                ratios.append(ratio)
                clipped = max(1-self.clip_eps, min(1+self.clip_eps, ratio))
                losses.append(-min(ratio * advantage, clipped * advantage))
                if (advantage > 0 and ratio > 1+self.clip_eps) or (advantage < 0 and ratio < 1-self.clip_eps):
                    continue
                g = gradients.setdefault(step.context, [0.0] * 7)
                scale = ratio * advantage / max(1, len(batch))
                for j in range(7):
                    g[j] += scale * ((j == step.action) - p[j])
        for context, grad in gradients.items():
            logits = self.logits.setdefault(context, [0.0] * 7)
            for j in range(7):
                logits[j] += self.learning_rate * grad[j]
        for trajectory, reward in batch:
            k = str(trajectory.task.family)
            self.baselines[k] = 0.95 * self.baselines.get(k, 0.5) + 0.05 * reward
        if batch:
            self.version += 1
        return {"loss": sum(losses)/max(1, len(losses)),
                "max_ratio": max(ratios, default=1.0),
                "clip_fraction": sum(abs(r-1) > self.clip_eps for r in ratios)/max(1, len(ratios))}

    def state(self):
        return {"version": self.version, "logits": self.logits, "baselines": self.baselines}

    def restore(self, state):
        self.version, self.logits, self.baselines = state["version"], state["logits"], state["baselines"]

    def distill(self, trajectories, rate=0.02):
        # Search -> Train imitation uses ONLY trusted successful candidates.
        for t in trajectories:
            for step in t.steps:
                p = softmax(self.logits.get(step.context, [0.0] * 7))
                z = self.logits.setdefault(step.context, [0.0] * 7)
                for j in range(7):
                    z[j] += rate * ((j == step.action) - p[j])


class WeightRegistry:
    """Immutable content-addressed snapshots; in-flight episodes pin a version."""
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.active = None

    def publish(self, snapshot):
        if self.active and snapshot.version <= self.active.version:
            raise ValueError("weight versions must increase")
        raw = {"version": snapshot.version, "logits": snapshot.logits}
        checksum = digest(raw)
        path = self.root / f"v{snapshot.version}-{checksum}.json"
        if path.exists() and path.read_text() != canonical(raw):
            raise ValueError("immutable weight artifact collision")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(canonical(raw))
        tmp.replace(path)
        pointer = self.root / "active.json"
        temp = pointer.with_suffix(".tmp")
        temp.write_text(canonical({"artifact": path.name, "sha256": checksum}))
        temp.replace(pointer)
        self.active = Snapshot(snapshot.version, dict(snapshot.logits))
        return checksum

    def load(self):
        import json
        manifest = json.loads((self.root / "active.json").read_text())
        name = manifest["artifact"]
        if Path(name).name != name:
            raise ValueError("invalid weight artifact path")
        raw = json.loads((self.root / name).read_text())
        if digest(raw) != manifest["sha256"]:
            raise ValueError("weight checksum mismatch")
        self.active = Snapshot(raw["version"], {k: tuple(v) for k, v in raw["logits"].items()})
        return self.active


class AffineToolEnvironment:
    """Executable restricted DSL coding tasks, with no arbitrary code execution.

    Seven actions: inspect, three edits, run public tests, submit, off-by-one edit.
    Overfit edits pass the public x=0 test but fail hidden nonzero inputs.
    """
    def __init__(self, task):
        self.task = task
        self.program = (0, 0)
        self.tested = False
        self.done = False

    def context(self, turn):
        return f"{self.task.family}:{min(turn, 3)}:{int(self.tested)}"

    def step(self, action):
        t = self.task
        if action == 0:
            return t.prompt
        if action == 1:
            self.program = (t.coefficient, t.bias)
        elif action == 2:
            self.program = (0, t.bias)  # Specification gaming.
        elif action == 3:
            self.program = (-t.coefficient, t.bias)
        elif action == 4:
            self.tested = True
            return "public tests: " + str(public_reward(t, self.program))
        elif action == 5:
            self.done = True
            return "submitted"
        elif action == 6:
            self.program = (t.coefficient, t.bias + 1)
        else:
            raise ValueError("unknown tool action")
        self.tested = False
        return f"program: {self.program[0]}*x+{self.program[1]}"


def public_reward(task, program):
    a, b = program
    return sum(a*x+b == task.coefficient*x+task.bias for x in task.public_inputs)/len(task.public_inputs)


def trusted_reward(task, program, completed=True):
    # Evaluator-owned test suite, never included in actor observations.
    xs = (-7, -2, 1, 3, 11)
    a, b = program
    return float(completed and all(a*x+b == task.coefficient*x+task.bias for x in xs))


async def rollout(task, snapshot, seed, trajectory_id, *, max_steps=24,
                  deadline_s=7200, tool_latency_s=0.0, journal=None, checkpoint=None):
    if max_steps <= 0 or deadline_s <= 0:
        raise ValueError("invalid episode budget")
    start = time.perf_counter()
    env, rng, steps = AffineToolEnvironment(task), random.Random(seed), []
    if checkpoint is not None and Path(checkpoint).exists():
        import json
        saved = json.loads(Path(checkpoint).read_text())
        if saved["identity"] != digest({"task": task.task_id, "snapshot": {"version": snapshot.version, "logits": snapshot.logits}, "seed": seed}):
            raise ValueError("episode checkpoint provenance mismatch")
        for raw in saved["steps"]:
            context = env.context(len(steps))
            action = rng.choices(range(7), weights=snapshot.probabilities(context))[0]
            if action != raw["action"]:
                raise ValueError("checkpoint action stream mismatch")
            env.step(action)
            steps.append(Step(**raw))
    # Snapshot is pinned across tool calls even if the learner publishes weights.
    async with asyncio.timeout(deadline_s):
        for turn in range(len(steps), max_steps):
            if env.done:
                break
            context = env.context(turn)
            p = snapshot.probabilities(context)
            action = rng.choices(range(7), weights=p)[0]
            observation = env.step(action)
            steps.append(Step(context, action, math.log(p[action]), observation))
            if journal is not None:
                journal(trajectory_id, turn, steps[-1])
            if checkpoint is not None:
                from dataclasses import asdict
                path = Path(checkpoint)
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(".tmp")
                temp.write_text(canonical({"identity": digest({"task":task.task_id,
                    "snapshot":{"version":snapshot.version,"logits":snapshot.logits},"seed":seed}),
                    "steps":[asdict(s) for s in steps]}))
                temp.replace(path)
            await asyncio.sleep(tool_latency_s)  # Also yields control at zero latency.
            if env.done:
                break
    return Trajectory(trajectory_id, task, snapshot.version, tuple(steps),
                      env.program, env.done, seed, time.perf_counter()-start)
