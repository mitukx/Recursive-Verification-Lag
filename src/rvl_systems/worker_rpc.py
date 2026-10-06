from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict
from typing import Any

from .backends import InferenceBackend
from .rpc_protocol import read_json_line, write_json_line
from .types import Generation


class RolloutWorkerServer:
    """Minimal newline-delimited JSON RPC server for remote rollout workers."""

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        policy_version: int = 0,
        request_timeout_s: float = 120.0,
    ) -> None:
        self.backend = backend
        self.policy_version = policy_version
        self.request_timeout_s = request_timeout_s
        self._server: asyncio.AbstractServer | None = None

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> tuple[str, int]:
        self._server = await asyncio.start_server(self._handle, host, port)
        socket = self._server.sockets[0]
        address = socket.getsockname()
        return str(address[0]), int(address[1])

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    def publish_policy_version(self, version: int) -> None:
        if version < self.policy_version:
            raise ValueError("policy version cannot move backwards")
        self.policy_version = version

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            request = await read_json_line(reader)
            op = request.get("op")
            if op == "ping":
                await write_json_line(
                    writer,
                    {"ok": True, "policy_version": self.policy_version},
                )
                return
            if op != "generate":
                await write_json_line(
                    writer,
                    {"ok": False, "error": f"unsupported op: {op}"},
                )
                return
            request_id = str(request.get("request_id") or "")
            if not request_id:
                raise ValueError("request_id is required")
            async with asyncio.timeout(self.request_timeout_s):
                generations = await self.backend.generate(
                    str(request["prompt_id"]),
                    str(request["prompt"]),
                    n=int(request["n"]),
                    temperature=float(request["temperature"]),
                    seed=int(request["seed"]),
                )
            await write_json_line(
                writer,
                {
                    "ok": True,
                    "request_id": request_id,
                    "policy_version": self.policy_version,
                    "generations": [asdict(g) for g in generations],
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


class TCPWorkerBackend:
    """InferenceBackend client for RolloutWorkerServer."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        expected_policy_version: int | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.host = host
        self.port = port
        self.expected_policy_version = expected_policy_version
        self.timeout_s = timeout_s

    async def ping(self) -> int:
        reader, writer = await asyncio.open_connection(self.host, self.port)
        try:
            await write_json_line(writer, {"op": "ping"})
            response = await asyncio.wait_for(read_json_line(reader), timeout=self.timeout_s)
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error", "worker ping failed")))
            return int(response["policy_version"])
        finally:
            writer.close()
            await writer.wait_closed()

    async def generate(
        self,
        prompt_id: str,
        prompt: str,
        *,
        n: int,
        temperature: float,
        seed: int,
    ) -> list[Generation]:
        request_id = uuid.uuid4().hex
        reader, writer = await asyncio.open_connection(self.host, self.port)
        try:
            await write_json_line(
                writer,
                {
                    "op": "generate",
                    "request_id": request_id,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "n": n,
                    "temperature": temperature,
                    "seed": seed,
                },
            )
            response = await asyncio.wait_for(read_json_line(reader), timeout=self.timeout_s)
        finally:
            writer.close()
            await writer.wait_closed()
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "worker generation failed")))
        if response.get("request_id") != request_id:
            raise RuntimeError("worker returned mismatched request_id")
        version = int(response["policy_version"])
        if self.expected_policy_version is not None and version != self.expected_policy_version:
            raise RuntimeError(
                f"worker policy version {version} does not match expected "
                f"{self.expected_policy_version}"
            )
        out: list[Generation] = []
        for raw in response.get("generations", []):
            metadata = dict(raw.get("metadata") or {})
            metadata["policy_version"] = version
            metadata["rpc_worker"] = f"{self.host}:{self.port}"
            out.append(
                Generation(
                    prompt_id=str(raw["prompt_id"]),
                    prompt=str(raw["prompt"]),
                    response=str(raw["response"]),
                    logprob=float(raw["logprob"]),
                    token_count=int(raw["token_count"]),
                    latency_s=float(raw["latency_s"]),
                    metadata=metadata,
                )
            )
        if len(out) != n:
            raise RuntimeError(f"worker returned {len(out)} generations, expected {n}")
        return out
