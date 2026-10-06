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

    @classmethod
    def from_json(cls, payload: str) -> "WeightManifest":
        raw = json.loads(payload)
        return cls(
            version=int(raw["version"]),
            artifact=str(raw["artifact"]),
            sha256=str(raw["sha256"]),
            bytes=int(raw["bytes"]),
        )


class WeightPublisher:
    """Crash-consistent immutable weight publisher with durable worker ACKs.

    Published versions are recovered from immutable manifest files rather than
    an in-memory counter. ACK state is stored atomically. A crash may leave an
    unreferenced artifact, but it cannot cause an already manifested version to
    be reused.
    """

    _STATE_FILE = "weight-publisher-state.json"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._state_path = self.root / self._STATE_FILE
        self._version = self._recover_highest_version()
        self._acks = self._load_acks()
        self._validate_acks()
        self._persist_state()

    @property
    def version(self) -> int:
        return self._version

    def _atomic_write_text(self, path: Path, content: str) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)

    def _manifest_path(self, version: int) -> Path:
        return self.root / f"weights-v{version:06d}.json"

    def _recover_highest_version(self) -> int:
        versions: list[int] = []
        for path in self.root.glob("weights-v*.json"):
            stem = path.stem
            try:
                versions.append(int(stem.removeprefix("weights-v")))
            except ValueError:
                continue
        return max(versions, default=0)

    def _load_acks(self) -> dict[str, int]:
        if not self._state_path.exists():
            return {}
        raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        acks = raw.get("acks", {})
        if not isinstance(acks, dict):
            raise RuntimeError("invalid durable weight ACK state")
        return {str(worker): int(version) for worker, version in acks.items()}

    def _validate_acks(self) -> None:
        for worker, version in self._acks.items():
            if version < 0 or version > self._version:
                raise RuntimeError(
                    f"invalid durable ACK for {worker}: version={version}, "
                    f"published={self._version}"
                )

    def _persist_state(self) -> None:
        self._atomic_write_text(
            self._state_path,
            json.dumps(
                {"published_version": self._version, "acks": self._acks},
                sort_keys=True,
            ),
        )

    def manifest_for_version(self, version: int) -> WeightManifest:
        if version <= 0 or version > self._version:
            raise ValueError(f"unknown weight version: {version}")
        path = self._manifest_path(version)
        if not path.exists():
            raise RuntimeError(f"missing immutable manifest for version {version}")
        manifest = WeightManifest.from_json(path.read_text(encoding="utf-8"))
        if manifest.version != version:
            raise RuntimeError(
                f"manifest version mismatch: expected={version}, "
                f"observed={manifest.version}"
            )
        return manifest

    def publish_bytes(self, payload: bytes, *, suffix: str = ".bin") -> WeightManifest:
        version = self._version + 1
        digest = hashlib.sha256(payload).hexdigest()
        name = f"weights-v{version:06d}-{digest[:12]}{suffix}"
        path = self.root / name
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)

        manifest = WeightManifest(
            version=version,
            artifact=name,
            sha256=digest,
            bytes=len(payload),
        )
        self._atomic_write_text(self._manifest_path(version), manifest.to_json())
        self._version = version
        self._persist_state()
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
        self._persist_state()

    def workers_behind(self, workers: list[str], *, version: int | None = None) -> dict[str, int]:
        target = self._version if version is None else version
        if target < 0 or target > self._version:
            raise ValueError(f"invalid target version: {target}")
        return {
            worker: target - self._acks.get(worker, 0)
            for worker in workers
            if self._acks.get(worker, 0) < target
        }

    def all_acknowledged(self, workers: list[str], *, version: int | None = None) -> bool:
        return not self.workers_behind(workers, version=version)
