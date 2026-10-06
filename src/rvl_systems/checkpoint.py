from __future__ import annotations

import json
from pathlib import Path

from .backends import ToyTabularBackend


def save_toy_checkpoint(path: str | Path, backend: ToyTabularBackend, step: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "step": int(step),
        "actions": list(backend.actions),
        "logits": backend.logits,
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tmp.replace(target)


def load_toy_checkpoint(path: str | Path, backend: ToyTabularBackend) -> int:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    actions = tuple(payload["actions"])
    if actions != backend.actions:
        raise ValueError("checkpoint action space does not match backend")
    raw = payload.get("logits", {})
    backend.logits = {str(k): [float(x) for x in v] for k, v in raw.items()}
    return int(payload["step"])
