"""torchrun replay learner: CPU/Gloo DDP or CUDA/NCCL FSDP."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ..grpo import compute_group_advantages
from ..hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
from ..types import Generation, VerifiedGeneration


def initialize(model,mode="ddp"):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    rank,world,local = (int(os.environ.get(k,v)) for k,v in
                        (("RANK","0"),("WORLD_SIZE","1"),("LOCAL_RANK","0")))
    cuda = torch.cuda.is_available()
    if mode == "fsdp" and not cuda:
        raise RuntimeError("FSDP acceptance requires CUDA GPUs")
    if cuda:
        torch.cuda.set_device(local)
        device = torch.device("cuda",local)
    else:
        device = torch.device("cpu")
    if not dist.is_initialized():
        dist.init_process_group("nccl" if cuda else "gloo")
    model.to(device)
    precision_name = "fp32"
    if mode == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        from functools import partial
        low_precision = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
        precision_name = (
            "bf16" if low_precision == torch.bfloat16 else "fp16"
        )
        model = FullyShardedDataParallel(
            model,
            device_id=device,
            use_orig_params=True,
            auto_wrap_policy=partial(
                size_based_auto_wrap_policy,
                min_num_params=1_000_000,
            ),
            mixed_precision=MixedPrecision(
                param_dtype=low_precision,
                reduce_dtype=torch.float32,
                buffer_dtype=low_precision,
            ),
        )
    elif mode == "ddp":
        model = DistributedDataParallel(model,device_ids=[local] if cuda else None)
    else:
        raise ValueError("unknown distributed mode")
    return model,rank,world,device,precision_name


def distributed_step(
    model,
    samples,
    output,
    *,
    mode="ddp",
    learning_rate=1e-5,
    resume_from=None,
):
    import torch
    import torch.distributed as dist
    model,rank,world,device,precision_name = initialize(model,mode)
    try:
        if not samples or len(samples)%world:
            raise ValueError("global replay batch must be nonempty and divisible by world size")
        # Global grouping precedes sharding; per-rank normalization is incorrect.
        advantages = [r.advantage for r in compute_group_advantages(samples)]
        trainer = HFCausalLMGRPOTrainer(
            model,
            config=HFTTrainerConfig(learning_rate=learning_rate),
        )
        resumed = False
        if resume_from:
            if mode != "fsdp":
                raise ValueError("resume_from is currently supported for fsdp mode")
            from torch.distributed.checkpoint import load
            from torch.distributed.checkpoint.state_dict import (
                get_state_dict,
                set_state_dict,
            )
            model_state, optim_state = get_state_dict(
                model,
                trainer.optimizer,
            )
            state = {
                "model": model_state,
                "optimizer": optim_state,
            }
            load(
                state,
                checkpoint_id=str(resume_from),
            )
            set_state_dict(
                model,
                trainer.optimizer,
                model_state_dict=state["model"],
                optim_state_dict=state["optimizer"],
            )
            resumed = True
            dist.barrier()
        indices = list(range(rank,len(samples),world))
        local = [samples[i] for i in indices]
        start = time.perf_counter()
        metrics = trainer.train_step(local,advantages=[advantages[i] for i in indices])
        keys = ("loss","mean_reward","clip_fraction","behavior_kl_estimate")
        aggregate = torch.tensor([metrics[k] for k in keys],device=device)
        dist.all_reduce(aggregate)
        aggregate /= world
        for key,value in zip(keys,aggregate.tolist()):
            metrics[key] = value
        ratio = torch.tensor(metrics["max_abs_log_ratio"],device=device)
        dist.all_reduce(ratio,op=dist.ReduceOp.MAX)
        metrics["max_abs_log_ratio"] = float(ratio)
        metrics["samples"] = len(samples)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        total_tokens = sum(s.generation.token_count for s in samples)
        duration = torch.tensor(elapsed,device=device)
        dist.all_reduce(duration,op=dist.ReduceOp.MAX)
        if mode == "ddp":
            for parameter in model.parameters():
                reference = parameter.detach().clone()
                dist.broadcast(reference,src=0)
                if not torch.allclose(parameter.detach(),reference,atol=1e-6,rtol=1e-5):
                    raise AssertionError("DDP ranks diverged")
            state = model.module.state_dict()
            if rank == 0:
                Path(output).mkdir(parents=True,exist_ok=True)
                torch.save({"model":state,"optimizer":trainer.optimizer.state_dict()},Path(output)/"checkpoint.pt")
        else:
            from torch.distributed.checkpoint import save
            from torch.distributed.checkpoint.state_dict import get_state_dict
            model_state,optim_state = get_state_dict(model,trainer.optimizer)
            save({"model":model_state,"optimizer":optim_state},checkpoint_id=str(Path(output)/"sharded-checkpoint"))
        report = {
            "mode": mode,
            "backend": dist.get_backend(),
            "world_size": world,
            "tokens_per_s": total_tokens / float(duration),
            "elapsed_s": float(duration),
            "metrics": metrics,
            "gpu_peak_memory_bytes": (
                torch.cuda.max_memory_allocated()
                if device.type == "cuda"
                else None
            ),
            "mixed_precision": precision_name,
            "resumed_from_checkpoint": resumed,
            "mfu": None,
            "note": (
                "MFU requires measured model FLOPs and device peak throughput"
            ),
        }
        if rank == 0:
            Path(output).mkdir(parents=True,exist_ok=True)
            (Path(output)/"distributed-report.json").write_text(json.dumps(report,indent=2,sort_keys=True))
            print(json.dumps(report,sort_keys=True))
        dist.barrier()
        return report
    finally:
        dist.destroy_process_group()


def read_samples(path):
    out = []
    for line in Path(path).read_text().splitlines():
        raw = json.loads(line)
        raw["generation"] = Generation(**raw["generation"])
        out.append(VerifiedGeneration(**raw))
    return out
