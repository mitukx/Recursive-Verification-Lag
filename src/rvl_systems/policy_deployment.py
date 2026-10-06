from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .weight_sync import WeightManifest, WeightPublisher


@dataclass(frozen=True)
class DeploymentStatus:
    active_version: int
    pending_version: int | None
    workers_behind: dict[str, int]
    coordinator_epoch: int


class PolicyDeploymentCoordinator:
    """Durable two-phase deployment with crash recovery and coordinator fencing.

    Every coordinator incarnation acquires a monotonically increasing durable
    epoch. Mutating operations re-read durable state and fail closed when a
    newer incarnation has taken over. Active/pending versions survive process
    restart, and a recovered pending artifact is checksum-verified before the
    coordinator can proceed.
    """

    _STATE_FILE = "policy-deployment-state.json"

    def __init__(
        self,
        root: str | Path,
        workers: list[str],
    ) -> None:
        if not workers:
            raise ValueError("at least one rollout worker is required")
        if len(set(workers)) != len(workers):
            raise ValueError("worker IDs must be unique")

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.publisher = WeightPublisher(self.root)
        self.workers = tuple(workers)
        self._state_path = self.root / self._STATE_FILE

        state = self._load_state()
        configured_workers = tuple(state.get("workers", self.workers))
        if configured_workers != self.workers:
            raise RuntimeError(
                "rollout worker set changed across coordinator recovery: "
                f"durable={configured_workers}, requested={self.workers}"
            )

        active_version = int(state.get("active_version", 0))
        pending_version = state.get("pending_version")
        if pending_version is not None:
            pending_version = int(pending_version)

        if active_version < 0 or active_version > self.publisher.version:
            raise RuntimeError("durable active policy version is invalid")
        if pending_version is not None:
            if pending_version <= active_version or pending_version > self.publisher.version:
                raise RuntimeError("durable pending policy version is invalid")
            pending = self.publisher.manifest_for_version(pending_version)
            self.publisher.verify(pending)
        else:
            pending = None

        self.active_version = active_version
        self.pending = pending
        self.epoch = int(state.get("coordinator_epoch", 0)) + 1
        self._persist_state()

    def _atomic_write_text(self, path: Path, content: str) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)

    def _load_state(self) -> dict:
        if not self._state_path.exists():
            return {}
        raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise RuntimeError("invalid durable policy deployment state")
        return raw

    def _persist_state(self) -> None:
        self._atomic_write_text(
            self._state_path,
            json.dumps(
                {
                    "active_version": self.active_version,
                    "pending_version": (
                        self.pending.version if self.pending is not None else None
                    ),
                    "coordinator_epoch": self.epoch,
                    "workers": list(self.workers),
                },
                sort_keys=True,
            ),
        )

    def _assert_current_epoch(self) -> None:
        state = self._load_state()
        durable_epoch = int(state.get("coordinator_epoch", 0))
        if durable_epoch != self.epoch:
            raise RuntimeError(
                "stale policy deployment coordinator fenced: "
                f"local_epoch={self.epoch}, durable_epoch={durable_epoch}"
            )

    def publish(self, payload: bytes, *, suffix: str = ".bin") -> WeightManifest:
        self._assert_current_epoch()
        if self.pending is not None:
            raise RuntimeError(
                f"cannot publish while version {self.pending.version} is pending"
            )
        manifest = self.publisher.publish_bytes(payload, suffix=suffix)
        self.publisher.verify(manifest)
        self.pending = manifest
        self._persist_state()
        return manifest

    def acknowledge(self, worker_id: str, version: int) -> None:
        self._assert_current_epoch()
        if worker_id not in self.workers:
            raise KeyError(f"unknown worker: {worker_id}")
        if self.pending is None:
            raise RuntimeError("no pending policy deployment")
        if version != self.pending.version:
            raise ValueError(
                f"worker {worker_id} acknowledged version {version}; "
                f"pending version is {self.pending.version}"
            )
        self.publisher.acknowledge(worker_id, version)

    def can_activate(self) -> bool:
        self._assert_current_epoch()
        if self.pending is None:
            return False
        return self.publisher.all_acknowledged(
            list(self.workers), version=self.pending.version
        )

    def activate(self) -> int:
        self._assert_current_epoch()
        if self.pending is None:
            raise RuntimeError("no pending policy deployment")
        if not self.can_activate():
            lagging = self.publisher.workers_behind(
                list(self.workers), version=self.pending.version
            )
            raise RuntimeError(f"workers have not acknowledged pending policy: {lagging}")
        version = self.pending.version
        if version <= self.active_version:
            raise RuntimeError(
                f"policy activation must advance monotonically: "
                f"active={self.active_version}, pending={version}"
            )
        self.publisher.verify(self.pending)
        self.active_version = version
        self.pending = None
        self._persist_state()
        return version

    def assert_rollout_version(self, version: int) -> None:
        if version != self.active_version:
            raise RuntimeError(
                f"rollout policy version {version} is not active "
                f"(active={self.active_version})"
            )

    def status(self) -> DeploymentStatus:
        self._assert_current_epoch()
        pending_version = self.pending.version if self.pending is not None else None
        return DeploymentStatus(
            active_version=self.active_version,
            pending_version=pending_version,
            workers_behind=(
                self.publisher.workers_behind(
                    list(self.workers), version=pending_version
                )
                if pending_version is not None
                else {}
            ),
            coordinator_epoch=self.epoch,
        )
