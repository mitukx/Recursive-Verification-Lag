"""Crash-consistent two-phase deployment for remote verifier fleets."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .types import VerifiedGeneration


@dataclass(frozen=True)
class VerifierArtifactManifest:
    version: int
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError("verifier artifact version must be nonnegative")
        if not self.path:
            raise ValueError("verifier artifact path is required")
        if len(self.sha256) != 64:
            raise ValueError("verifier artifact sha256 must be a 64-character digest")

    @classmethod
    def from_dict(cls, value: dict) -> "VerifierArtifactManifest":
        return cls(
            version=int(value["version"]),
            path=str(value["path"]),
            sha256=str(value["sha256"]),
        )


class PinnedVersionVerifier:
    """Wrap a verifier with deployment-controlled immutable version identity."""

    def __init__(self, verifier, version: int):
        self._verifier = verifier
        self._version = int(version)

    @property
    def version(self) -> int:
        return self._version

    def refresh(self) -> None:
        raise RuntimeError("deployed verifier version is immutable; deploy a new artifact")

    async def verify(self, generation):
        result = await self._verifier.verify(generation)
        metadata = dict(result.metadata)
        metadata["underlying_verifier_version"] = int(result.verifier_version)
        metadata["deployment_verifier_version"] = self._version
        return VerifiedGeneration(
            generation=result.generation,
            reward=result.reward,
            verifier_latency_s=result.verifier_latency_s,
            verifier_version=self._version,
            metadata=metadata,
        )


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def publish_verifier_artifact(
    root: str | Path,
    *,
    version: int,
    content: bytes,
    suffix: str = ".bin",
) -> VerifierArtifactManifest:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if version < 0:
        raise ValueError("verifier version must be nonnegative")
    path = root / f"verifier-v{version:08d}{suffix}"
    if path.exists():
        raise FileExistsError(f"immutable verifier artifact already exists: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    os.replace(tmp, path)
    return VerifierArtifactManifest(version, str(path.resolve()), sha256_file(path))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


class VerifierDeploymentCoordinator:
    """Durable prepare/ACK/activate coordinator with epoch fencing.

    A newly constructed coordinator increments the durable epoch. Once workers
    observe that epoch, calls from older coordinator instances fail closed.
    """

    def __init__(
        self,
        clients: Iterable,
        *,
        state_path: str | Path,
    ) -> None:
        self.clients = list(clients)
        if not self.clients:
            raise ValueError("at least one verifier worker client is required")
        self.state_path = Path(state_path)
        previous = self._read_state()
        self.epoch = int(previous.get("coordinator_epoch", 0)) + 1
        self.active_version = int(previous.get("active_version", -1))
        self.pending_manifest = previous.get("pending_manifest")
        self.acked_workers = list(previous.get("acked_workers") or [])
        self._persist()

    def _read_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        value = json.loads(self.state_path.read_text())
        if not isinstance(value, dict):
            raise ValueError("verifier deployment state must be a JSON object")
        return value

    def _persist(self) -> None:
        existing = self._read_state()
        existing_epoch = int(existing.get("coordinator_epoch", 0))
        if existing_epoch > self.epoch:
            raise RuntimeError(
                f"stale verifier coordinator epoch {self.epoch}; "
                f"durable epoch is {existing_epoch}"
            )
        _atomic_json(
            self.state_path,
            {
                "schema_version": 1,
                "coordinator_epoch": self.epoch,
                "active_version": self.active_version,
                "pending_manifest": self.pending_manifest,
                "acked_workers": sorted(self.acked_workers),
            },
        )

    async def prepare(self, manifest: VerifierArtifactManifest) -> list[str]:
        if manifest.version < self.active_version:
            raise ValueError("cannot prepare verifier older than active version")
        payload = asdict(manifest)
        results = await asyncio.gather(
            *[
                client.prepare_verifier(payload, coordinator_epoch=self.epoch)
                for client in self.clients
            ],
            return_exceptions=True,
        )
        acks: list[str] = []
        errors: list[str] = []
        for index, result in enumerate(results):
            if isinstance(result, BaseException):
                errors.append(f"worker[{index}]: {type(result).__name__}: {result}")
                continue
            if int(result["verifier_version"]) != manifest.version:
                errors.append(f"worker[{index}]: prepared wrong verifier version")
                continue
            if int(result["coordinator_epoch"]) != self.epoch:
                errors.append(f"worker[{index}]: ACK from wrong coordinator epoch")
                continue
            acks.append(str(result["worker_id"]))
        if len(set(acks)) != len(acks):
            errors.append("duplicate worker identity in verifier prepare ACKs")
        self.pending_manifest = payload
        self.acked_workers = acks
        self._persist()
        if errors or len(acks) != len(self.clients):
            raise RuntimeError("verifier prepare incomplete: " + "; ".join(errors))
        return acks

    async def activate(self) -> int:
        if self.pending_manifest is None:
            raise RuntimeError("no prepared verifier deployment")
        manifest = VerifierArtifactManifest.from_dict(self.pending_manifest)
        if len(self.acked_workers) != len(self.clients):
            raise RuntimeError("cannot activate without all worker prepare ACKs")
        results = await asyncio.gather(
            *[
                client.activate_verifier(
                    manifest.version,
                    coordinator_epoch=self.epoch,
                )
                for client in self.clients
            ],
            return_exceptions=True,
        )
        errors = []
        for index, result in enumerate(results):
            if isinstance(result, BaseException):
                errors.append(f"worker[{index}]: {type(result).__name__}: {result}")
                continue
            if int(result["verifier_version"]) != manifest.version:
                errors.append(f"worker[{index}]: activated wrong verifier version")
        if errors:
            # Retain pending state. Calling deploy(manifest) again under the same
            # epoch is safe and converges workers through idempotent prepare/activate.
            self._persist()
            raise RuntimeError("verifier activation incomplete: " + "; ".join(errors))

        health = await asyncio.gather(*(client.ping() for client in self.clients))
        versions = {int(row["verifier_version"]) for row in health}
        epochs = {int(row.get("coordinator_epoch", -1)) for row in health}
        worker_ids = [str(row["worker_id"]) for row in health]
        if len(set(worker_ids)) != len(worker_ids):
            self._persist()
            raise RuntimeError("post-activation convergence has duplicate worker identity")
        if versions != {manifest.version} or epochs != {self.epoch}:
            self._persist()
            raise RuntimeError(
                f"post-activation convergence failed: versions={versions}, epochs={epochs}"
            )
        self.active_version = manifest.version
        self.pending_manifest = None
        self.acked_workers = []
        self._persist()
        return self.active_version

    async def deploy(self, manifest: VerifierArtifactManifest) -> int:
        await self.prepare(manifest)
        return await self.activate()

    def state(self) -> dict:
        return self._read_state()
