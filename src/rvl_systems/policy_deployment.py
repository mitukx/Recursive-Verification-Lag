from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .weight_sync import WeightManifest, WeightPublisher


@dataclass(frozen=True)
class DeploymentStatus:
    active_version: int
    pending_version: int | None
    workers_behind: dict[str, int]


class PolicyDeploymentCoordinator:
    """Two-phase policy deployment for trainer -> rollout worker synchronization.

    A newly published weight artifact is *pending* until every required worker
    acknowledges that exact version. Only then can the version become active
    for new rollout requests. At most one deployment may be pending at once.
    """

    def __init__(
        self,
        root: str | Path,
        workers: list[str],
    ) -> None:
        if not workers:
            raise ValueError("at least one rollout worker is required")
        if len(set(workers)) != len(workers):
            raise ValueError("worker IDs must be unique")
        self.publisher = WeightPublisher(root)
        self.workers = tuple(workers)
        self.active_version = 0
        self.pending: WeightManifest | None = None

    def publish(self, payload: bytes, *, suffix: str = ".bin") -> WeightManifest:
        if self.pending is not None:
            raise RuntimeError(
                f"cannot publish while version {self.pending.version} is pending"
            )
        manifest = self.publisher.publish_bytes(payload, suffix=suffix)
        self.publisher.verify(manifest)
        self.pending = manifest
        return manifest

    def acknowledge(self, worker_id: str, version: int) -> None:
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
        if self.pending is None:
            return False
        return self.publisher.all_acknowledged(list(self.workers))

    def activate(self) -> int:
        if self.pending is None:
            raise RuntimeError("no pending policy deployment")
        if not self.can_activate():
            lagging = self.publisher.workers_behind(list(self.workers))
            raise RuntimeError(f"workers have not acknowledged pending policy: {lagging}")
        version = self.pending.version
        if version != self.active_version + 1:
            raise RuntimeError(
                f"policy activation must advance exactly once: "
                f"active={self.active_version}, pending={version}"
            )
        self.active_version = version
        self.pending = None
        return version

    def assert_rollout_version(self, version: int) -> None:
        if version != self.active_version:
            raise RuntimeError(
                f"rollout policy version {version} is not active "
                f"(active={self.active_version})"
            )

    def status(self) -> DeploymentStatus:
        pending_version = self.pending.version if self.pending is not None else None
        return DeploymentStatus(
            active_version=self.active_version,
            pending_version=pending_version,
            workers_behind=self.publisher.workers_behind(list(self.workers))
            if self.pending is not None
            else {},
        )
