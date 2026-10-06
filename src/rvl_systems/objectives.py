from __future__ import annotations

import math
from statistics import fmean


def clipped_grpo_objective(
    new_logprob: float,
    old_logprob: float,
    advantage: float,
    *,
    clip_eps: float = 0.2,
) -> float:
    """Sequence-level clipped surrogate used for diagnostics and unit tests."""
    if not 0.0 < clip_eps < 1.0:
        raise ValueError("clip_eps must be in (0, 1)")
    ratio = math.exp(max(-20.0, min(20.0, new_logprob - old_logprob)))
    clipped = max(1.0 - clip_eps, min(1.0 + clip_eps, ratio))
    return min(ratio * advantage, clipped * advantage)


def mean_clipped_grpo_loss(
    rows: list[tuple[float, float, float]],
    *,
    clip_eps: float = 0.2,
) -> float:
    """Return negative mean surrogate objective for (new, old, advantage) rows."""
    if not rows:
        return 0.0
    return -fmean(
        clipped_grpo_objective(new, old, adv, clip_eps=clip_eps)
        for new, old, adv in rows
    )
