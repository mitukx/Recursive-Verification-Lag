from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RatioDiagnostics:
    count: int
    clip_fraction: float
    nonfinite_count: int
    mean_ratio: float
    max_abs_log_ratio: float


def safe_importance_ratio(
    new_logprob: float,
    old_logprob: float,
    *,
    max_abs_log_ratio: float = 20.0,
) -> float:
    delta = new_logprob - old_logprob
    if not math.isfinite(delta):
        raise FloatingPointError("non-finite log-probability ratio")
    delta = max(-max_abs_log_ratio, min(max_abs_log_ratio, delta))
    return math.exp(delta)


def summarize_ratios(
    rows: list[tuple[float, float]],
    *,
    clip_eps: float = 0.2,
    max_abs_log_ratio: float = 20.0,
) -> RatioDiagnostics:
    if not 0.0 < clip_eps < 1.0:
        raise ValueError("clip_eps must be in (0, 1)")
    ratios: list[float] = []
    nonfinite = 0
    max_abs = 0.0
    clipped = 0
    for new, old in rows:
        delta = new - old
        if not math.isfinite(delta):
            nonfinite += 1
            continue
        max_abs = max(max_abs, abs(delta))
        ratio = safe_importance_ratio(
            new,
            old,
            max_abs_log_ratio=max_abs_log_ratio,
        )
        ratios.append(ratio)
        if ratio < 1.0 - clip_eps or ratio > 1.0 + clip_eps:
            clipped += 1
    if not ratios:
        return RatioDiagnostics(0, 0.0, nonfinite, 0.0, max_abs)
    return RatioDiagnostics(
        count=len(ratios),
        clip_fraction=clipped / len(ratios),
        nonfinite_count=nonfinite,
        mean_ratio=sum(ratios) / len(ratios),
        max_abs_log_ratio=max_abs,
    )
