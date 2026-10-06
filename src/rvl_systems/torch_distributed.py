from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    backend: str


def context_from_env(backend: str = "gloo") -> DistributedContext:
    return DistributedContext(
        rank=int(os.environ.get("RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        backend=backend,
    )


def init_process_group(backend: str = "gloo") -> DistributedContext:
    try:
        import torch.distributed as dist
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for distributed execution") from exc
    ctx = context_from_env(backend)
    if ctx.world_size <= 1:
        raise ValueError("distributed execution requires WORLD_SIZE > 1")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return ctx


def destroy_process_group() -> None:
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def broadcast_module_parameters(model: Any, src: int = 0) -> None:
    import torch.distributed as dist
    if not dist.is_initialized():
        raise RuntimeError("process group is not initialized")
    for parameter in model.parameters():
        dist.broadcast(parameter.data, src=src)
    for buffer in model.buffers():
        dist.broadcast(buffer.data, src=src)


def all_reduce_mean(tensor: Any) -> Any:
    import torch.distributed as dist
    if not dist.is_initialized():
        raise RuntimeError("process group is not initialized")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor
