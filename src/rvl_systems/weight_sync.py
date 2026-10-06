from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WeightManifest:
    version: int
    artifact: str
    sha256: str
    bytes: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "artifact": self.artifact,
                "sha256": self.sha256,
                "bytes": self.bytes,
            },
            sort_keys=True,
        )


class WeightPublisher:
    """Publishes immutable weight artifacts and tracks worker acknowledgements."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._version = 0
        self._acks: dict[str, int] = {}

    @property
    def version(self) -> int:
        return self._version

    def publish_bytes(self, payload: bytes, *, suffix: str = ".bin") -> WeightManifest:
        self._version += 1
        digest = hashlib.sha256(payload).hexdigest()
        name = f"weights-v{self._version:06d}-{digest[:12]}{suffix}"
        path = self.root / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)
        manifest = WeightManifest(
            version=self._version,
            artifact=name,
            sha256=digest,
            bytes=len(payload),
        )
        manifest_path = self.root / f"weights-v{self._version:06d}.json"
        manifest_path.write_text(manifest.to_json(), encoding="utf-8")
        return manifest

    def verify(self, manifest: WeightManifest) -> None:
        path = self.root / manifest.artifact
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != manifest.bytes:
            raise RuntimeError("weight artifact byte count mismatch")
        if digest != manifest.sha256:
            raise RuntimeError("weight artifact checksum mismatch")

    def acknowledge(self, worker_id: str, version: int) -> None:
        if version > self._version:
            raise ValueError("cannot acknowledge an unpublished weight version")
        previous = self._acks.get(worker_id, 0)
        if version < previous:
            raise ValueError("worker acknowledgement cannot move backwards")
        self._acks[worker_id] = version

    def workers_behind(self, workers: list[str]) -> dict[str, int]:
        return {
            worker: self._version - self._acks.get(worker, 0)
            for worker in workers
            if self._acks.get(worker, 0) < self._version
        }

    def all_acknowledged(self, workers: list[str]) -> bool:
        return not self.workers_behind(workers)
