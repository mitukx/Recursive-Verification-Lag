from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean

from .backends import ToyTabularBackend
from .types import TrainRecord, VerifiedGeneration


@dataclass(frozen=True)
class GRPOConfig:
    learning_rate: float = 0.15
    advantage_eps: float = 1e-6
    clip_advantage: float = 5.0


def compute_group_advantages(
    samples: list[VerifiedGeneration],
    *,
    eps: float = 1e-6,
    clip: float = 5.0,
) -> list[TrainRecord]:
    """Normalize rewards per prompt while preserving the original sample order."""
    by_prompt: dict[str, list[float]] = {}
    for sample in samples:
        by_prompt.setdefault(sample.generation.prompt_id, []).append(sample.reward)

    stats: dict[str, tuple[float, float]] = {}
    for prompt_id, rewards in by_prompt.items():
        mean = fmean(rewards)
        variance = fmean((reward - mean) ** 2 for reward in rewards)
        stats[prompt_id] = (mean, math.sqrt(variance + eps))

    records: list[TrainRecord] = []
    for item in samples:
        prompt_id = item.generation.prompt_id
        mean, scale = stats[prompt_id]
        advantage = max(-clip, min(clip, (item.reward - mean) / scale))
        records.append(
            TrainRecord(
                prompt_id=prompt_id,
                response=item.generation.response,
                reward=item.reward,
                advantage=advantage,
                old_logprob=item.generation.logprob,
            )
        )
    return records


def train_toy_policy(
    backend: ToyTabularBackend,
    samples: list[VerifiedGeneration],
    config: GRPOConfig,
) -> dict[str, float]:
    records = compute_group_advantages(
        samples,
        eps=config.advantage_eps,
        clip=config.clip_advantage,
    )
    index = {action: i for i, action in enumerate(backend.actions)}
    updates = 0
    mean_abs_adv = 0.0
    for record in records:
        if record.response not in index:
            continue
        backend.apply_policy_gradient(
            record.prompt_id,
            index[record.response],
            record.advantage,
            config.learning_rate,
        )
        updates += 1
        mean_abs_adv += abs(record.advantage)
    return {
        "updates": float(updates),
        "mean_abs_advantage": mean_abs_adv / max(1, updates),
        "mean_reward": fmean([s.reward for s in samples]) if samples else 0.0,
    }
