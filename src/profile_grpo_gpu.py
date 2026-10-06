from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path


def _event_device_us(event) -> float:
    for name in ("self_device_time_total", "self_cuda_time_total", "cuda_time_total"):
        value = getattr(event, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0


def _event_cpu_us(event) -> float:
    return float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile a real-GPU GRPO step and export stage-aware Chrome trace evidence."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--replay", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--precision", choices=["auto","fp32","bf16","fp16"], default="auto")
    parser.add_argument("--objective-backend", choices=["torch","triton"], default="torch")
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace", required=True)
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")

    import torch
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    from src.rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
    from src.rvl_systems.lab.distributed_learner import read_samples

    if args.precision == "auto":
        precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    else:
        precision = args.precision
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requested but unsupported by this CUDA device")
    dtype = {
        "fp32":torch.float32,
        "bf16":torch.bfloat16,
        "fp16":torch.float16,
    }[precision]

    replay = read_samples(args.replay)
    if len(replay) < args.samples:
        raise ValueError("replay contains fewer samples than requested")
    replay = replay[:args.samples]
    total_tokens = sum(sample.generation.token_count for sample in replay)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model,dtype=dtype)
    model.config.use_cache = False
    model.to("cuda")
    trainer = HFCausalLMGRPOTrainer(
        model,
        config=HFTTrainerConfig(
            learning_rate=0.0,
            objective_backend=args.objective_backend,
        ),
    )

    # Warm up allocator, kernels, optimizer state, and optional Triton compilation.
    trainer.train_step(replay)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        metrics = trainer.train_step(replay)
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - start

    trace = Path(args.trace)
    trace.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace))

    rows = []
    stages = []
    for event in prof.key_averages():
        row = {
            "name":str(event.key),
            "count":int(event.count),
            "self_cpu_time_us":_event_cpu_us(event),
            "self_device_time_us":_event_device_us(event),
            "cpu_memory_usage":int(getattr(event,"cpu_memory_usage",0) or 0),
            "device_memory_usage":int(
                getattr(event,"device_memory_usage",getattr(event,"cuda_memory_usage",0)) or 0
            ),
        }
        if not all(
            math.isfinite(float(row[key]))
            for key in ("self_cpu_time_us","self_device_time_us")
        ):
            raise FloatingPointError("profiler produced non-finite timing")
        rows.append(row)
        if row["name"].startswith("rvl.grpo."):
            stages.append(row)

    top_device = sorted(rows,key=lambda row:row["self_device_time_us"],reverse=True)[:25]
    top_cpu = sorted(rows,key=lambda row:row["self_cpu_time_us"],reverse=True)[:25]
    peak = int(torch.cuda.max_memory_allocated())
    payload = {
        "schema":1,
        "git_sha":os.environ.get("GITHUB_SHA","unknown"),
        "model":args.model,
        "precision":precision,
        "objective_backend":args.objective_backend,
        "device_name":torch.cuda.get_device_name(0),
        "samples":len(replay),
        "response_tokens":total_tokens,
        "wall_s":wall_s,
        "tokens_per_s":total_tokens/max(wall_s,1e-12),
        "gpu_peak_memory_bytes":peak,
        "train_metrics":metrics,
        "named_stages":stages,
        "top_self_device_time":top_device,
        "top_self_cpu_time":top_cpu,
        "trace_path":str(trace),
        "trace_bytes":trace.stat().st_size,
        "claim_boundary":"Single profiled GRPO step after warmup; bottleneck evidence is hardware/model scoped.",
    }
    if not stages:
        raise RuntimeError("named GRPO profiler stages missing")
    if payload["trace_bytes"] <= 0 or payload["tokens_per_s"] <= 0:
        raise RuntimeError("invalid profiler evidence")
    out = Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(payload,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
