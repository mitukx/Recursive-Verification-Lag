"""Candidate-side numeric RPC. No oracle, expected outputs, rewards or clocks.

This file is mounted separately from the evaluator in Docker. Never deserialize
pickle or import evaluator code into the candidate process.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import struct
import sys
import zipfile

import numpy as np
import torch


MAX_FRAME = 8 * 1024 * 1024


def pack(arrays):
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def unpack(raw):
    if len(raw) > MAX_FRAME:
        raise ValueError("frame too large")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_FRAME:
            raise ValueError("expanded frame too large")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def read_exact(stream, count):
    result = bytearray()
    while len(result) < count:
        chunk = stream.read(count - len(result))
        if not chunk:
            raise EOFError("incomplete frame")
        result.extend(chunk)
    return bytes(result)


def read_frame(stream):
    size = struct.unpack("!Q", read_exact(stream, 8))[0]
    if size > MAX_FRAME:
        raise ValueError("frame too large")
    return unpack(read_exact(stream, size))


def write_frame(stream, arrays):
    raw = pack(arrays)
    if len(raw) > MAX_FRAME:
        raise ValueError("frame too large")
    stream.write(struct.pack("!Q", len(raw)) + raw)
    stream.flush()


def metadata(value):
    return np.frombuffer(json.dumps(value, sort_keys=True).encode(), dtype=np.uint8)


def evaluate(fn, arrays):
    options = json.loads(arrays["meta"].tobytes())
    current = torch.from_numpy(arrays["current"].copy())
    old = torch.from_numpy(arrays["old"].copy())
    advantage = torch.from_numpy(arrays["advantage"].copy())
    if options.get("strided"):
        def strided(value):
            storage = torch.empty(value.numel() * 2, dtype=value.dtype)
            storage[::2] = value.reshape(-1)
            return storage[::2].reshape(value.shape)
        current, old, advantage = map(strided, (current, old, advantage))
    current.requires_grad_(True)
    result = fn(current, old, advantage, clip_eps=options["clip_eps"],
                max_abs_log_ratio=options["max_abs_log_ratio"])
    if not isinstance(result, tuple) or len(result) != 4:
        raise ValueError("surrogate must return four tensors")
    upstream = torch.from_numpy(arrays["upstream"].copy())
    (result[0] * upstream).sum().backward()
    if current.grad is None:
        raise ValueError("surrogate must preserve current-logprob gradients")
    output = {f"output_{i}": row.detach().cpu().numpy() for i, row in enumerate(result)}
    output["gradient"] = current.grad.detach().cpu().numpy()
    output["meta"] = metadata({"ok": True})
    return output


def main():
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("candidate", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    # Text printed by a candidate is diagnostic only, never a reward channel.
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
    fn = getattr(module, "torch_grpo_surrogate")
    stream_in, stream_out = sys.stdin.buffer, sys.stdout.buffer
    write_frame(stream_out, {"meta": metadata({"ready": True})})
    while True:
        try:
            arrays = read_frame(stream_in)
        except EOFError:
            return
        try:
            with contextlib.redirect_stdout(sys.stderr):
                output = evaluate(fn, arrays)
        except Exception as exc:
            output = {"meta": metadata({"ok": False, "error_type": type(exc).__name__})}
        write_frame(stream_out, output)


if __name__ == "__main__":
    main()
