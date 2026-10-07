import torch

def _require_torch():
    return torch


def torch_grpo_surrogate(
    current_logps,
    old_logps,
    advantages,
    *,
    clip_eps: float = 0.2,
    max_abs_log_ratio: float = 20.0,
):
    """Reference tokenwise clipped GRPO/PPO surrogate."""
    t = _require_torch()
    if not 0.0 < clip_eps < 1.0:
        raise ValueError("clip_eps must be in (0, 1)")
    if max_abs_log_ratio <= 0:
        raise ValueError("max_abs_log_ratio must be positive")
    if current_logps.shape != old_logps.shape:
        raise ValueError("current and old log-probabilities must have equal shape")
    if current_logps.shape != advantages.shape:
        raise ValueError("advantages must match token log-probability shape")

    raw_log_ratio = current_logps - old_logps
    log_ratio = t.clamp(
        raw_log_ratio,
        min=-max_abs_log_ratio,
        max=max_abs_log_ratio,
    )
    ratio = t.exp(log_ratio)
    clipped = t.clamp(
        ratio,
        1.0 - clip_eps,
        1.0 + clip_eps,
    )
    surrogate = t.minimum(
        ratio * advantages,
        clipped * advantages,
    )
    clip_mask = (
        (ratio < 1.0 - clip_eps)
        | (ratio > 1.0 + clip_eps)
    ).to(current_logps.dtype)
    abs_log_ratio = log_ratio.abs()
    behavior_kl = (ratio - 1.0) - log_ratio
    return surrogate, clip_mask, abs_log_ratio, behavior_kl
