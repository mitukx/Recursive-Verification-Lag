from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Generation:
    prompt_id: str
    prompt: str
    response: str
    logprob: float
    token_count: int
    latency_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifiedGeneration:
    generation: Generation
    reward: float
    verifier_latency_s: float
    verifier_version: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainRecord:
    prompt_id: str
    response: str
    reward: float
    advantage: float
    old_logprob: float
