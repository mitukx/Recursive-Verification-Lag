from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import fmean


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
        tokens = out.get("rollout.tokens", 0.0)
        latency = sum(self.observations.get("rollout.request_latency_s", []))
        if latency > 0:
            out["rollout.tokens_per_s"] = tokens / latency
        return out
