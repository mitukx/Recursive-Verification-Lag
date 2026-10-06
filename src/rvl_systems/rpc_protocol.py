from __future__ import annotations

import asyncio
import json
from typing import Any

MAX_MESSAGE_BYTES = 2 * 1024 * 1024


async def read_json_line(reader: asyncio.StreamReader) -> dict[str, Any]:
    raw = await reader.readline()
    if not raw:
        raise ConnectionError("peer closed connection")
    if len(raw) > MAX_MESSAGE_BYTES:
        raise ValueError("RPC message exceeds maximum size")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON RPC message") from exc
    if not isinstance(value, dict):
        raise ValueError("RPC message must be a JSON object")
    return value


async def write_json_line(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise ValueError("RPC message exceeds maximum size")
    writer.write(encoded)
    await writer.drain()
