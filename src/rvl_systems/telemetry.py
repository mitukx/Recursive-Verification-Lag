from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import fmean


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


@dataclass
class Telemetry:
    counters: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    observations: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def increment(self, name: str, value: float = 1.0) -> None:
        self.counters[name] += value

    def observe(self, name: str, value: float) -> None:
        self.observations[name].append(float(value))

    def snapshot(self) -> dict[str, float]:
        out = dict(self.counters)
        for name, values in self.observations.items():
            if values:
                out[f"{name}.mean"] = fmean(values)
                out[f"{name}.max"] = max(values)
                out[f"{name}.p50"] = _percentile(values, 0.50)
                out[f"{name}.p95"] = _percentile(values, 0.95)
        tokens = out.get("rollout.tokens", 0.0)
        latency = sum(self.observations.get("rollout.request_latency_s", []))
        if latency > 0:
            out["rollout.tokens_per_s"] = tokens / latency
        return out
