from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path


def fetch_text(url: str, timeout_s: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return response.read().decode("utf-8")


def fetch_vllm_metrics(endpoint: str, timeout_s: float = 10.0) -> str:
    return fetch_text(endpoint.rstrip("/") + "/metrics", timeout_s=timeout_s)


def parse_prometheus_samples(text: str) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        metric_with_labels, value_raw = parts
        name = metric_with_labels.split("{", 1)[0]
        try:
            value = float(value_raw)
        except ValueError:
            continue
        samples.setdefault(name, []).append(value)
    return samples


def sum_metric(samples: dict[str, list[float]], name: str) -> float:
    return sum(samples.get(name, []))


def max_metric(samples: dict[str, list[float]], name: str) -> float | None:
    values = samples.get(name, [])
    return max(values) if values else None


def counter_delta(
    before: dict[str, list[float]],
    after: dict[str, list[float]],
    name: str,
) -> float:
    return max(0.0, sum_metric(after, name) - sum_metric(before, name))


def gpu_inventory() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"available": False, "gpus": []}

    gpus = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 5:
            gpus.append(
                {
                    "index": fields[0],
                    "name": fields[1],
                    "driver_version": fields[2],
                    "memory_total_mb": fields[3],
                    "compute_capability": fields[4],
                }
            )
    return {"available": bool(gpus), "gpus": gpus}


def write_json(data: object, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
