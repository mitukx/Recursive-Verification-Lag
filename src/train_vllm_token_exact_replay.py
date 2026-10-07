"""Consume remote token-exact replay in the local GRPO learner and measure parity."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from src.rvl_systems.hf_backend import HFLocalBackend
from src.rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
from src.rvl_systems.lab.distributed_learner import read_samples


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_probe(model, *, tensors: int = 24, values_per_tensor: int = 64):
    rows = []
    for name, parameter in model.named_parameters():
        if len(rows) >= tensors:
            break
        flat = parameter.detach().reshape(-1)
        if not flat.numel():
            continue
        count = min(values_per_tensor, flat.numel())
        if count == 1:
            indices = [0]
        else:
            indices = [
                int(round(i * (flat.numel() - 1) / (count - 1)))
                for i in range(count)
            ]
        values = flat[indices].float().cpu().tolist()
        rows.append((name, values))
    if not rows:
        raise ValueError("model exposes no trainable parameter probe")
    return rows


def probe_delta(before, after) -> dict:
    if [x[0] for x in before] != [x[0] for x in after]:
        raise ValueError("parameter probe identity changed")
    diffs = [
        abs(float(a) - float(b))
        for (_, left), (_, right) in zip(before, after)
        for a, b in zip(left, right)
    ]
    return {
        "sampled_parameter_values": len(diffs),
        "parameter_probe_l1_change": sum(diffs),
        "parameter_probe_max_abs_change": max(diffs or [0.0]),
    }


def reward_diagnostics(samples) -> dict:
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.generation.prompt_id].append(float(sample.reward))
    informative = sum(len(set(values)) > 1 for values in groups.values())
    return {
        "groups": len(groups),
        "samples": len(samples),
        "informative_reward_groups": informative,
        "reward_mean": sum(float(x.reward) for x in samples) / len(samples),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Exact local snapshot served by vLLM")
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--precision", choices=["auto","fp32","fp16","bf16"], default="fp32")
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--max-preupdate-log-ratio", type=float, default=0.20)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.max_preupdate_log_ratio <= 0:
        raise ValueError("max pre-update log-ratio threshold must be positive")

    samples = read_samples(args.replay)
    if not samples:
        raise ValueError("token-exact replay is empty")
    diagnostics = reward_diagnostics(samples)

    backend = HFLocalBackend(
        str(args.model),
        max_new_tokens=1,
        precision=args.precision,
    )
    model = backend.model
    trainer = HFCausalLMGRPOTrainer(
        model,
        config=HFTTrainerConfig(learning_rate=args.learning_rate),
    )
    torch = trainer.torch
    model.eval()
    pre_ratios = []
    pre_kls = []
    with torch.no_grad():
        for sample in samples:
            _, _, ratio, kl = trainer._sample_objective(sample, 1.0)
            value = float(ratio.detach().cpu())
            kl_value = float(kl.detach().cpu())
            if not math.isfinite(value) or not math.isfinite(kl_value):
                raise FloatingPointError("non-finite pre-update serving/learner parity")
            pre_ratios.append(value)
            pre_kls.append(kl_value)
    max_ratio = max(pre_ratios)
    if max_ratio > args.max_preupdate_log_ratio:
        raise RuntimeError(
            f"serving/learner pre-update log-ratio {max_ratio:.6f} exceeds "
            f"declared threshold {args.max_preupdate_log_ratio:.6f}"
        )

    before = parameter_probe(model)
    metrics = trainer.train_step(samples)
    after = parameter_probe(model)
    delta = probe_delta(before, after)

    report = {
        "status": "completed",
        "bridge": "remote_token_exact_serving_to_local_grpo",
        "model_snapshot": str(args.model),
        "device": backend.resolved_device,
        "precision": backend.resolved_precision,
        "learning_rate": args.learning_rate,
        "replay_sha256": sha256_file(args.replay),
        **diagnostics,
        "preupdate_parity": {
            "max_abs_log_ratio": max_ratio,
            "mean_behavior_kl_estimate": sum(pre_kls) / len(pre_kls),
            "threshold": args.max_preupdate_log_ratio,
            "passed": True,
        },
        "train_metrics": metrics,
        **delta,
        "optimizer_had_nonzero_gradient": float(metrics.get("grad_norm", 0.0)) > 0.0,
        "claim_boundary": (
            "A real remote serving replay was accepted by the same-model GRPO learner "
            "without client retokenization. Parameter-change metrics depend on reward "
            "variance and are not a capability-improvement claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
