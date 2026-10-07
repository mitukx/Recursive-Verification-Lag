from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .rpc_protocol import read_json_line, write_json_line
from .types import Generation, VerifiedGeneration
from .verifier import Verifier
from .verifier_deployment import (
    PinnedVersionVerifier,
    VerifierArtifactManifest,
    sha256_file,
)


def _generation_from_dict(raw: dict) -> Generation:
    return Generation(
        prompt_id=str(raw["prompt_id"]),
        prompt=str(raw["prompt"]),
        response=str(raw["response"]),
        logprob=float(raw["logprob"]),
        token_count=int(raw["token_count"]),
        latency_s=float(raw["latency_s"]),
        metadata=dict(raw.get("metadata") or {}),
    )


class VerifierWorkerServer:
    """Version-fenced newline-delimited JSON RPC verifier worker."""

    def __init__(
        self,
        verifier: Verifier,
        *,
        worker_id: str,
        capacity: int = 1,
        request_timeout_s: float = 120.0,
        service_time_hint_s: float | None = None,
        deployment_loader=None,
        deployment_state_path: str | Path | None = None,
    ) -> None:
        if not worker_id:
            raise ValueError("worker_id is required")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.verifier = verifier
        self.worker_id = worker_id
        self.capacity = int(capacity)
        self.request_timeout_s = float(request_timeout_s)
        if service_time_hint_s is not None and service_time_hint_s <= 0:
            raise ValueError("service_time_hint_s must be positive")
        self.service_time_hint_s = (
            float(service_time_hint_s) if service_time_hint_s is not None else None
        )
        self._semaphore = asyncio.Semaphore(self.capacity)
        self._server: asyncio.AbstractServer | None = None
        self._inflight = 0
        self._healthy = True
        self.deployment_loader = deployment_loader
        self.deployment_state_path = (
            Path(deployment_state_path) if deployment_state_path is not None else None
        )
        self._coordinator_epoch = 0
        self._active_manifest: VerifierArtifactManifest | None = None
        self._prepared_manifest: VerifierArtifactManifest | None = None
        self._prepared_verifier = None

    @property
    def verifier_version(self) -> int:
        return int(self.verifier.version)

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> tuple[str, int]:
        await self._recover_deployment_state()
        self._server = await asyncio.start_server(self._handle, host, port)
        address = self._server.sockets[0].getsockname()
        return str(address[0]), int(address[1])

    def _deployment_state(self) -> dict:
        return {
            "schema_version": 1,
            "coordinator_epoch": self._coordinator_epoch,
            "active_manifest": (
                asdict(self._active_manifest) if self._active_manifest is not None else None
            ),
            "prepared_manifest": (
                asdict(self._prepared_manifest) if self._prepared_manifest is not None else None
            ),
        }

    def _persist_deployment_state(self) -> None:
        if self.deployment_state_path is None:
            return
        self.deployment_state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.deployment_state_path.with_suffix(self.deployment_state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._deployment_state(), indent=2, sort_keys=True) + "\n")
        os.replace(tmp, self.deployment_state_path)

    async def _load_deployed_verifier(self, manifest: VerifierArtifactManifest):
        if self.deployment_loader is None:
            raise RuntimeError("verifier deployment loader is not configured")
        path = Path(manifest.path)
        if not path.is_file():
            raise FileNotFoundError(f"verifier artifact missing: {path}")
        if sha256_file(path) != manifest.sha256:
            raise RuntimeError("verifier artifact checksum mismatch")
        loaded = self.deployment_loader(manifest)
        if inspect.isawaitable(loaded):
            loaded = await loaded
        if not hasattr(loaded, "verify"):
            raise TypeError("deployment loader did not return a verifier")
        return PinnedVersionVerifier(loaded, manifest.version)

    async def _recover_deployment_state(self) -> None:
        if self.deployment_state_path is None or not self.deployment_state_path.exists():
            return
        if self.deployment_loader is None:
            raise RuntimeError("deployment state exists but no deployment loader is configured")
        raw = json.loads(self.deployment_state_path.read_text())
        self._coordinator_epoch = int(raw.get("coordinator_epoch", 0))
        active = raw.get("active_manifest")
        prepared = raw.get("prepared_manifest")
        if active is not None:
            self._active_manifest = VerifierArtifactManifest.from_dict(active)
            self.verifier = await self._load_deployed_verifier(self._active_manifest)
        if prepared is not None:
            self._prepared_manifest = VerifierArtifactManifest.from_dict(prepared)
            if (
                self._active_manifest is not None
                and self._prepared_manifest == self._active_manifest
            ):
                self._prepared_verifier = self.verifier
            else:
                self._prepared_verifier = await self._load_deployed_verifier(
                    self._prepared_manifest
                )

    def _accept_coordinator_epoch(self, epoch: int) -> None:
        if epoch < self._coordinator_epoch:
            raise RuntimeError(
                f"stale coordinator epoch {epoch}; current epoch is {self._coordinator_epoch}"
            )
        if epoch > self._coordinator_epoch:
            self._coordinator_epoch = epoch
            self._prepared_manifest = None
            self._prepared_verifier = None
            self._persist_deployment_state()

    async def _prepare_verifier(self, request: dict) -> dict:
        epoch = int(request["coordinator_epoch"])
        self._accept_coordinator_epoch(epoch)
        manifest = VerifierArtifactManifest.from_dict(dict(request["manifest"]))
        if manifest.version < self.verifier_version:
            raise RuntimeError("cannot prepare verifier older than active version")
        if manifest.version == self.verifier_version:
            if self._active_manifest is None or manifest != self._active_manifest:
                raise RuntimeError("active verifier version has different artifact identity")
            self._prepared_manifest = manifest
            self._prepared_verifier = self.verifier
        else:
            self._prepared_verifier = await self._load_deployed_verifier(manifest)
            self._prepared_manifest = manifest
        self._persist_deployment_state()
        return {
            "ok": True,
            "worker_id": self.worker_id,
            "verifier_version": manifest.version,
            "coordinator_epoch": self._coordinator_epoch,
            "artifact_sha256": manifest.sha256,
        }

    async def _activate_verifier(self, request: dict) -> dict:
        epoch = int(request["coordinator_epoch"])
        if epoch != self._coordinator_epoch:
            raise RuntimeError(
                f"activation epoch {epoch} does not match worker epoch {self._coordinator_epoch}"
            )
        version = int(request["verifier_version"])
        if version == self.verifier_version:
            if self._active_manifest is None or self._active_manifest.version != version:
                raise RuntimeError("active verifier has no matching artifact identity")
            if (
                self._prepared_manifest is not None
                and self._prepared_manifest.version == version
            ):
                self._prepared_manifest = None
                self._prepared_verifier = None
                self._persist_deployment_state()
            return {
                "ok": True,
                "worker_id": self.worker_id,
                "verifier_version": version,
                "coordinator_epoch": self._coordinator_epoch,
            }
        if self._prepared_manifest is None or self._prepared_verifier is None:
            raise RuntimeError("verifier activation requires a prepared artifact")
        if self._prepared_manifest.version != version:
            raise RuntimeError("prepared verifier version does not match activation")
        self.verifier = self._prepared_verifier
        self._active_manifest = self._prepared_manifest
        self._prepared_manifest = None
        self._prepared_verifier = None
        self._persist_deployment_state()
        return {
            "ok": True,
            "worker_id": self.worker_id,
            "verifier_version": self.verifier_version,
            "coordinator_epoch": self._coordinator_epoch,
        }

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    def set_health(self, healthy: bool) -> None:
        self._healthy = bool(healthy)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await read_json_line(reader)
            op = request.get("op")
            if op == "ping":
                await write_json_line(
                    writer,
                    {
                        "ok": True,
                        "worker_id": self.worker_id,
                        "verifier_version": self.verifier_version,
                        "healthy": self._healthy,
                        "capacity": self.capacity,
                        "inflight": self._inflight,
                        "service_time_hint_s": self.service_time_hint_s,
                        "coordinator_epoch": self._coordinator_epoch,
                        "active_artifact_sha256": (
                            self._active_manifest.sha256
                            if self._active_manifest is not None
                            else None
                        ),
                        "prepared_verifier_version": (
                            self._prepared_manifest.version
                            if self._prepared_manifest is not None
                            else None
                        ),
                    },
                )
                return
            if op == "prepare_verifier":
                await write_json_line(writer, await self._prepare_verifier(request))
                return
            if op == "activate_verifier":
                await write_json_line(writer, await self._activate_verifier(request))
                return
            if op != "verify":
                await write_json_line(writer, {"ok": False, "error": f"unsupported op: {op}"})
                return
            if not self._healthy:
                raise RuntimeError("verifier worker unhealthy")
            request_id = str(request.get("request_id") or "")
            if not request_id:
                raise ValueError("request_id is required")
            expected_version = int(request["expected_verifier_version"])
            if expected_version != self.verifier_version:
                raise RuntimeError(
                    f"verifier version {self.verifier_version} does not match expected "
                    f"{expected_version}"
                )
            generation = _generation_from_dict(dict(request["generation"]))
            started = time.perf_counter()
            async with self._semaphore:
                self._inflight += 1
                try:
                    async with asyncio.timeout(self.request_timeout_s):
                        verified = await self.verifier.verify(generation)
                finally:
                    self._inflight -= 1
            if (
                verified.verifier_version != expected_version
                or self.verifier_version != expected_version
            ):
                raise RuntimeError("verifier changed version while request was in flight")
            metadata = dict(verified.metadata)
            metadata.update(
                {
                    "rpc_worker_id": self.worker_id,
                    "rpc_service_latency_s": time.perf_counter() - started,
                }
            )
            await write_json_line(
                writer,
                {
                    "ok": True,
                    "request_id": request_id,
                    "worker_id": self.worker_id,
                    "verifier_version": expected_version,
                    "verified": {
                        "generation": asdict(verified.generation),
                        "reward": verified.reward,
                        "verifier_latency_s": verified.verifier_latency_s,
                        "verifier_version": verified.verifier_version,
                        "metadata": metadata,
                    },
                },
            )
        except Exception as exc:
            try:
                await write_json_line(
                    writer,
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                )
            except Exception:
                pass
        finally:
            writer.close()
            await writer.wait_closed()


class TCPVerifierClient:
    """Remote verifier client with strict request/version identity checks."""

    def __init__(self, host: str, port: int, *, timeout_s: float = 120.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)

    async def ping(self) -> dict:
        reader, writer = await asyncio.open_connection(self.host, self.port)
        try:
            await write_json_line(writer, {"op": "ping"})
            response = await asyncio.wait_for(read_json_line(reader), timeout=self.timeout_s)
        finally:
            writer.close()
            await writer.wait_closed()
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "verifier ping failed")))
        return response

    async def _management_rpc(self, payload: dict) -> dict:
        reader, writer = await asyncio.open_connection(self.host, self.port)
        try:
            await write_json_line(writer, payload)
            response = await asyncio.wait_for(read_json_line(reader), timeout=self.timeout_s)
        finally:
            writer.close()
            await writer.wait_closed()
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "verifier management RPC failed")))
        return response

    async def prepare_verifier(self, manifest: dict, *, coordinator_epoch: int) -> dict:
        return await self._management_rpc(
            {
                "op": "prepare_verifier",
                "manifest": dict(manifest),
                "coordinator_epoch": int(coordinator_epoch),
            }
        )

    async def activate_verifier(self, version: int, *, coordinator_epoch: int) -> dict:
        return await self._management_rpc(
            {
                "op": "activate_verifier",
                "verifier_version": int(version),
                "coordinator_epoch": int(coordinator_epoch),
            }
        )

    async def verify(self, generation: Generation, *, expected_verifier_version: int) -> VerifiedGeneration:
        request_id = uuid.uuid4().hex
        reader, writer = await asyncio.open_connection(self.host, self.port)
        try:
            await write_json_line(
                writer,
                {
                    "op": "verify",
                    "request_id": request_id,
                    "expected_verifier_version": int(expected_verifier_version),
                    "generation": asdict(generation),
                },
            )
            response = await asyncio.wait_for(read_json_line(reader), timeout=self.timeout_s)
        finally:
            writer.close()
            await writer.wait_closed()
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "remote verification failed")))
        if response.get("request_id") != request_id:
            raise RuntimeError("verifier returned mismatched request_id")
        if int(response["verifier_version"]) != int(expected_verifier_version):
            raise RuntimeError("verifier response version mismatch")
        raw = dict(response["verified"])
        generation_out = _generation_from_dict(dict(raw["generation"]))
        if generation_out != generation:
            raise RuntimeError("verifier mutated immutable generation payload")
        return VerifiedGeneration(
            generation=generation_out,
            reward=float(raw["reward"]),
            verifier_latency_s=float(raw["verifier_latency_s"]),
            verifier_version=int(raw["verifier_version"]),
            metadata=dict(raw.get("metadata") or {}),
        )


class DistributedVerifierFleet:
    """Small failover fleet that fences stale verifier workers by version."""

    def __init__(
        self,
        clients: Iterable[TCPVerifierClient],
        *,
        expected_verifier_version: int,
        failure_threshold: int = 1,
    ) -> None:
        self.clients = list(clients)
        if not self.clients:
            raise ValueError("at least one verifier client is required")
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.expected_verifier_version = int(expected_verifier_version)
        self.failure_threshold = int(failure_threshold)
        self._failures = [0 for _ in self.clients]
        self._quarantined = [False for _ in self.clients]
        self._cursor = 0
        self._round_robin_lock = asyncio.Lock()

    def publish_expected_version(self, version: int) -> None:
        version = int(version)
        if version < self.expected_verifier_version:
            raise ValueError("verifier version cannot move backwards")
        self.expected_verifier_version = version
        self._failures = [0 for _ in self.clients]
        self._quarantined = [False for _ in self.clients]

    async def refresh_health(self) -> list[dict]:
        rows = []
        for idx, client in enumerate(self.clients):
            try:
                info = await client.ping()
                compatible = (
                    bool(info.get("healthy"))
                    and int(info["verifier_version"]) == self.expected_verifier_version
                )
                self._quarantined[idx] = not compatible
                if compatible:
                    self._failures[idx] = 0
                rows.append({**info, "compatible": compatible})
            except Exception as exc:
                self._quarantined[idx] = True
                rows.append({"ok": False, "compatible": False, "error": str(exc)})
        return rows

    async def _reserve_round_robin(self, excluded: set[int]) -> int | None:
        async with self._round_robin_lock:
            for offset in range(len(self.clients)):
                idx = (self._cursor + offset) % len(self.clients)
                if self._quarantined[idx] or idx in excluded:
                    continue
                self._cursor = (idx + 1) % len(self.clients)
                return idx
        return None

    async def verify(self, generation: Generation) -> VerifiedGeneration:
        errors: list[str] = []
        attempted: set[int] = set()
        while len(attempted) < len(self.clients):
            idx = await self._reserve_round_robin(attempted)
            if idx is None:
                break
            attempted.add(idx)
            client = self.clients[idx]
            try:
                result = await client.verify(
                    generation,
                    expected_verifier_version=self.expected_verifier_version,
                )
                self._failures[idx] = 0
                return result
            except Exception as exc:
                self._failures[idx] += 1
                errors.append(f"worker[{idx}]: {exc}")
                if self._failures[idx] >= self.failure_threshold:
                    self._quarantined[idx] = True
        raise RuntimeError("all verifier workers unavailable: " + "; ".join(errors))

    @property
    def quarantined(self) -> tuple[bool, ...]:
        return tuple(self._quarantined)



class AdaptiveVerifierFleet(DistributedVerifierFleet):
    """Capacity/latency-aware verifier routing with bounded failover.

    The routing score approximates predicted completion time:

        ((local_inflight + remote_inflight + 1) / capacity) * service_time

    where service_time is an EWMA of measured end-to-end verifier RPC latency,
    initialized from the worker's optional health hint. The score is only a
    systems scheduling heuristic; it is not a reward-quality signal.
    """

    def __init__(
        self,
        clients: Iterable[TCPVerifierClient],
        *,
        expected_verifier_version: int,
        failure_threshold: int = 1,
        ewma_alpha: float = 0.25,
        default_service_time_s: float = 0.05,
        request_deadline_s: float | None = None,
    ) -> None:
        super().__init__(
            clients,
            expected_verifier_version=expected_verifier_version,
            failure_threshold=failure_threshold,
        )
        if not 0.0 < ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0,1]")
        if default_service_time_s <= 0:
            raise ValueError("default_service_time_s must be positive")
        if request_deadline_s is not None and request_deadline_s <= 0:
            raise ValueError("request_deadline_s must be positive")
        self.ewma_alpha = float(ewma_alpha)
        self.default_service_time_s = float(default_service_time_s)
        self.request_deadline_s = (
            float(request_deadline_s) if request_deadline_s is not None else None
        )
        n = len(self.clients)
        self._local_inflight = [0 for _ in range(n)]
        self._remote_inflight = [0 for _ in range(n)]
        self._capacity = [1 for _ in range(n)]
        self._service_time = [self.default_service_time_s for _ in range(n)]
        self._completed = [0 for _ in range(n)]
        self._routing_lock = asyncio.Lock()

    def publish_expected_version(self, version: int) -> None:
        super().publish_expected_version(version)
        self._local_inflight = [0 for _ in self.clients]
        self._remote_inflight = [0 for _ in self.clients]
        self._completed = [0 for _ in self.clients]

    async def refresh_health(self) -> list[dict]:
        rows = await super().refresh_health()
        for idx, row in enumerate(rows):
            if not row.get("compatible"):
                continue
            self._capacity[idx] = max(1, int(row.get("capacity", 1)))
            self._remote_inflight[idx] = max(0, int(row.get("inflight", 0)))
            hint = row.get("service_time_hint_s")
            if hint is not None and float(hint) > 0 and self._completed[idx] == 0:
                self._service_time[idx] = float(hint)
        return rows

    def _score(self, idx: int) -> float:
        queued = self._local_inflight[idx] + self._remote_inflight[idx] + 1
        return (queued / self._capacity[idx]) * self._service_time[idx]

    async def _reserve_best(self, excluded: set[int]) -> int | None:
        async with self._routing_lock:
            candidates = [
                idx for idx in range(len(self.clients))
                if not self._quarantined[idx] and idx not in excluded
            ]
            if not candidates:
                return None
            idx = min(candidates, key=lambda i: (self._score(i), i))
            self._local_inflight[idx] += 1
            return idx

    async def verify(self, generation: Generation) -> VerifiedGeneration:
        errors: list[str] = []
        attempted: set[int] = set()
        while len(attempted) < len(self.clients):
            idx = await self._reserve_best(attempted)
            if idx is None:
                break
            attempted.add(idx)
            started = time.perf_counter()
            try:
                call = self.clients[idx].verify(
                    generation,
                    expected_verifier_version=self.expected_verifier_version,
                )
                if self.request_deadline_s is None:
                    result = await call
                else:
                    result = await asyncio.wait_for(call, timeout=self.request_deadline_s)
                elapsed = time.perf_counter() - started
                async with self._routing_lock:
                    old = self._service_time[idx]
                    self._service_time[idx] = (
                        self.ewma_alpha * elapsed + (1.0 - self.ewma_alpha) * old
                    )
                    self._completed[idx] += 1
                    self._failures[idx] = 0
                metadata = dict(result.metadata)
                metadata["routing_predicted_service_s"] = self._service_time[idx]
                metadata["routing_worker_index"] = idx
                return VerifiedGeneration(
                    generation=result.generation,
                    reward=result.reward,
                    verifier_latency_s=result.verifier_latency_s,
                    verifier_version=result.verifier_version,
                    metadata=metadata,
                )
            except Exception as exc:
                errors.append(f"worker[{idx}]: {type(exc).__name__}: {exc}")
                async with self._routing_lock:
                    self._failures[idx] += 1
                    if self._failures[idx] >= self.failure_threshold:
                        self._quarantined[idx] = True
            finally:
                async with self._routing_lock:
                    self._local_inflight[idx] = max(0, self._local_inflight[idx] - 1)
        raise RuntimeError("all verifier workers unavailable: " + "; ".join(errors))

    def routing_snapshot(self) -> list[dict]:
        return [
            {
                "worker_index": idx,
                "capacity": self._capacity[idx],
                "local_inflight": self._local_inflight[idx],
                "remote_inflight": self._remote_inflight[idx],
                "service_time_ewma_s": self._service_time[idx],
                "completed": self._completed[idx],
                "quarantined": self._quarantined[idx],
                "predicted_completion_s": self._score(idx),
            }
            for idx in range(len(self.clients))
        ]
