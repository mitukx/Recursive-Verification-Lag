from __future__ import annotations

from dataclasses import dataclass

from .rollout import RolloutRequest


@dataclass(frozen=True)
class ServingCost:
    prefill_tokens: int
    decode_tokens: int
    kv_tokens: int


def estimate_serving_cost(request: RolloutRequest) -> ServingCost:
    """Conservative serving-cost estimate used only for scheduling."""
    prefill = request.prompt_tokens_estimate or 0
    decode_per_sample = (
        request.decode_tokens_estimate
        if request.decode_tokens_estimate is not None
        else max(1, request.work_units // request.samples)
    )
    decode = request.samples * decode_per_sample
    kv = request.samples * (prefill + decode_per_sample)
    return ServingCost(
        prefill_tokens=prefill,
        decode_tokens=decode,
        kv_tokens=max(1, kv),
    )
