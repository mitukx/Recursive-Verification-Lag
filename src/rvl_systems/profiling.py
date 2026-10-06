from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Span:
    name: str
    start_ns: int
    end_ns: int
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000_000.0


class TraceRecorder:
    """Dependency-free Chrome trace recorder for control-plane profiling."""

    def __init__(self) -> None:
        self.spans: list[Span] = []

    @contextmanager
    def span(self, name: str, **metadata: object) -> Iterator[None]:
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            end = time.perf_counter_ns()
            self.spans.append(Span(name, start, end, dict(metadata)))

    def summary(self) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[float]] = {}
        for span in self.spans:
            grouped.setdefault(span.name, []).append(span.duration_ms)
        out: dict[str, dict[str, float]] = {}
        for name, values in grouped.items():
            ordered = sorted(values)
            p50 = ordered[len(ordered) // 2]
            p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
            out[name] = {
                "count": float(len(values)),
                "mean_ms": sum(values) / len(values),
                "p50_ms": p50,
                "p95_ms": p95,
                "max_ms": max(values),
            }
        return out

    def export_chrome_trace(self, path: str | Path) -> None:
        events = []
        if not self.spans:
            Path(path).write_text(json.dumps({"traceEvents": []}), encoding="utf-8")
            return
        origin = min(span.start_ns for span in self.spans)
        for span in self.spans:
            events.append({
                "name": span.name,
                "cat": "rvl-systems",
                "ph": "X",
                "ts": (span.start_ns - origin) / 1000.0,
                "dur": (span.end_ns - span.start_ns) / 1000.0,
                "pid": 1,
                "tid": 1,
                "args": span.metadata,
            })
        Path(path).write_text(
            json.dumps({"traceEvents": events}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
