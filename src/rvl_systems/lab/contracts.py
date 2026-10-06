from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Task:
    task_id: str
    family: int
    coefficient: int
    bias: int
    public_inputs: tuple[int, ...] = (0,)
    split: str = "train"

    @property
    def prompt(self) -> str:
        return f"Implement f(x) = {self.coefficient}*x + {self.bias}. Tools: inspect, edit, test, finish."


@dataclass(frozen=True)
class Step:
    context: str
    action: int
    behavior_logprob: float
    observation: str

    def __post_init__(self):
        if not math.isfinite(self.behavior_logprob) or self.behavior_logprob > 0:
            raise ValueError("invalid behavior logprob")


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    task: Task
    policy_version: int
    steps: tuple[Step, ...]
    program: tuple[int, int]
    completed: bool
    seed: int
    elapsed_s: float = field(default=0.0, compare=False)

    def __post_init__(self):
        if self.policy_version < 0 or self.elapsed_s < 0 or not math.isfinite(self.elapsed_s):
            raise ValueError("invalid trajectory provenance")
        if self.task.split != "train":
            raise ValueError("evaluation tasks must never enter training replay")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw):
        raw = dict(raw)
        task = dict(raw["task"])
        task["public_inputs"] = tuple(task["public_inputs"])
        raw["task"] = Task(**task)
        raw["steps"] = tuple(Step(**s) for s in raw["steps"])
        raw["program"] = tuple(raw["program"])
        return cls(**raw)


@dataclass(frozen=True)
class Verdict:
    reward: float
    verifier_version: int
    scores: tuple[float, ...]
    cell: str
    uncertainty: float
    trusted: bool = False

    def __post_init__(self):
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in
                   (self.reward, self.uncertainty, *self.scores)):
            raise ValueError("verifier values must be finite probabilities")

    @property
    def disagreement(self):
        return max(self.scores) - min(self.scores) if self.scores else 0.0
