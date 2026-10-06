from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path


def _finite_metrics(metrics: dict[str, float]) -> bool:
    return all(
        math.isfinite(float(value))
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and key != "samples"
    )


def run_precision(model_name: str, samples, precision: str, seed: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    from src.rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig

    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        return {"precision":precision,"success":False,"error":"bf16_not_supported"}

    dtype = {
        "fp32":torch.float32,
        "bf16":torch.bfloat16,
        "fp16":torch.float16,
    }[precision]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(model_name,dtype=dtype)
    model.config.use_cache = False
    model.to("cuda")
    trainer = HFCausalLMGRPOTrainer(
        model,
        config=HFTTrainerConfig(learning_rate=0.0),
    )
    total_tokens = sum(sample.generation.token_count for sample in samples)
    start = time.perf_counter()
    try:
        metrics = trainer.train_step(samples)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        report = {
            "precision":precision,
            "success":_finite_metrics(metrics),
            "wall_s":elapsed,
            "tokens_per_s":total_tokens/max(elapsed,1e-12),
            "gpu_peak_memory_bytes":torch.cuda.max_memory_allocated(),
            "metrics":metrics,
        }
        if not report["success"]:
            report["error"] = "nonfinite_metrics"
        return report
    except BaseException as exc:
        torch.cuda.synchronize()
        return {
            "precision":precision,
            "success":False,
            "error":f"{type(exc).__name__}: {exc}",
            "gpu_peak_memory_bytes":torch.cuda.max_memory_allocated(),
        }
    finally:
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-GPU FP32/BF16/FP16 GRPO numerical and throughput comparison."
    )
    parser.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--replay",required=True)
    parser.add_argument("--samples",type=int,default=8)
    parser.add_argument("--seed",type=int,default=20261006)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    from src.rvl_systems.lab.distributed_learner import read_samples
    replay = read_samples(args.replay)
    if len(replay) < args.samples:
        raise ValueError("replay contains fewer samples than requested")
    replay = replay[:args.samples]

    runs = {
        precision:run_precision(args.model,replay,precision,args.seed)
        for precision in ("fp32","bf16","fp16")
    }
    reference = runs["fp32"]
    comparisons = {}
    if reference.get("success"):
        ref_metrics = reference["metrics"]
        for precision in ("bf16","fp16"):
            row = runs[precision]
            if not row.get("success"):
                comparisons[precision] = {"success":False}
                continue
            metrics = row["metrics"]
            loss_ref = float(ref_metrics["loss"])
            grad_ref = float(ref_metrics["grad_norm"])
            comparisons[precision] = {
                "success":True,
                "relative_loss_error":abs(float(metrics["loss"])-loss_ref)/max(abs(loss_ref),1e-8),
                "relative_grad_norm_error":abs(float(metrics["grad_norm"])-grad_ref)/max(abs(grad_ref),1e-8),
                "behavior_kl_abs_error":abs(float(metrics["behavior_kl_estimate"])-float(ref_metrics["behavior_kl_estimate"])),
                "max_abs_log_ratio_error":abs(float(metrics["max_abs_log_ratio"])-float(ref_metrics["max_abs_log_ratio"])),
                "throughput_ratio_vs_fp32":float(row["tokens_per_s"])/max(float(reference["tokens_per_s"]),1e-12),
                "peak_memory_ratio_vs_fp32":float(row["gpu_peak_memory_bytes"])/max(float(reference["gpu_peak_memory_bytes"]),1),
            }

    payload = {
        "schema":1,
        "git_sha":os.environ.get("GITHUB_SHA","unknown"),
        "model":args.model,
        "replay":str(Path(args.replay)),
        "samples":len(replay),
        "device_name":torch.cuda.get_device_name(0),
        "runs":runs,
        "comparisons":comparisons,
        "all_precisions_finite":all(bool(runs[p].get("success")) for p in ("fp32","bf16","fp16")),
        "claim_boundary":"Single-step low-precision GRPO parity/performance measurement; not a production-scale throughput claim.",
    }
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(payload,indent=2,sort_keys=True))
    if not payload["all_precisions_finite"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
