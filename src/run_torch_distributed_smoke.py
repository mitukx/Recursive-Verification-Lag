"""Two-process torch.distributed + DDP correctness smoke test.

Run with:
  torchrun --standalone --nproc-per-node=2 -m src.run_torch_distributed_smoke
"""

from __future__ import annotations

import json

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.rvl_systems.torch_distributed import (
    all_reduce_mean,
    broadcast_module_parameters,
    destroy_process_group,
    init_process_group,
)


def main() -> None:
    ctx = init_process_group("gloo")
    try:
        # Explicit parameter broadcast correctness.
        base_model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            if ctx.rank == 0:
                base_model.weight.fill_(3.0)
            else:
                base_model.weight.fill_(-7.0)
        broadcast_module_parameters(base_model, src=0)
        expected = torch.full_like(base_model.weight, 3.0)
        if not torch.equal(base_model.weight, expected):
            raise RuntimeError(
                f"rank {ctx.rank}: parameter broadcast failed: {base_model.weight}"
            )

        # Collective correctness.
        value = torch.tensor(float(ctx.rank + 1))
        all_reduce_mean(value)
        expected_mean = (ctx.world_size + 1) / 2.0
        if abs(float(value) - expected_mean) > 1e-6:
            raise RuntimeError(
                f"rank {ctx.rank}: all-reduce mean failed: {float(value)}"
            )

        # Real DDP gradient synchronization and optimizer step.
        ddp_model = DDP(base_model)
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
        x = torch.tensor([[float(ctx.rank + 1), 1.0]])
        target = torch.tensor([[float(2 * ctx.rank + 1)]])
        optimizer.zero_grad(set_to_none=True)
        prediction = ddp_model(x)
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()

        weight = ddp_model.module.weight.detach().clone()
        gathered = [torch.empty_like(weight) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, weight)
        for other in gathered[1:]:
            if not torch.allclose(gathered[0], other, atol=1e-7, rtol=0.0):
                raise RuntimeError(
                    f"rank {ctx.rank}: DDP parameters diverged across ranks: {gathered}"
                )

        dist.barrier()
        print(json.dumps({
            "rank": ctx.rank,
            "world_size": ctx.world_size,
            "backend": ctx.backend,
            "broadcast_weight": 3.0,
            "all_reduce_mean": float(value),
            "ddp_loss": float(loss.detach()),
            "ddp_weight": [float(x) for x in weight.flatten()],
            "ok": True,
        }, sort_keys=True))
    finally:
        destroy_process_group()


if __name__ == "__main__":
    main()
