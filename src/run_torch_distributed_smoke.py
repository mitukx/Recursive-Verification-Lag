"""Two-process torch.distributed correctness smoke test.

Run with:
  torchrun --standalone --nproc-per-node=2 -m src.run_torch_distributed_smoke
"""

from __future__ import annotations

import json

import torch
import torch.distributed as dist

from src.rvl_systems.torch_distributed import (
    all_reduce_mean,
    broadcast_module_parameters,
    destroy_process_group,
    init_process_group,
)


def main() -> None:
    ctx = init_process_group("gloo")
    try:
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            if ctx.rank == 0:
                model.weight.fill_(3.0)
            else:
                model.weight.fill_(-7.0)

        broadcast_module_parameters(model, src=0)
        expected = torch.full_like(model.weight, 3.0)
        if not torch.equal(model.weight, expected):
            raise RuntimeError(
                f"rank {ctx.rank}: parameter broadcast failed: {model.weight}"
            )

        value = torch.tensor(float(ctx.rank + 1))
        all_reduce_mean(value)
        expected_mean = (ctx.world_size + 1) / 2.0
        if abs(float(value) - expected_mean) > 1e-6:
            raise RuntimeError(
                f"rank {ctx.rank}: all-reduce mean failed: {float(value)}"
            )

        dist.barrier()
        print(json.dumps({
            "rank": ctx.rank,
            "world_size": ctx.world_size,
            "backend": ctx.backend,
            "broadcast_weight": float(model.weight[0, 0]),
            "all_reduce_mean": float(value),
            "ok": True,
        }, sort_keys=True))
    finally:
        destroy_process_group()


if __name__ == "__main__":
    main()
