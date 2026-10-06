from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkReport:
    name: str
    metrics: dict[str, float | int | str]
    config: dict[str, float | int | str]
    git_sha: str = "unknown"
    model: str = "none"

    def payload(self) -> dict:
        return {
            **asdict(self),
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.payload(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
