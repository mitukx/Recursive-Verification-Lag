"""One-device real-model RLVR/GRPO smoke run for Mac, CUDA, or CPU."""

from __future__ import annotations

import argparse
import asyncio
import json
from statistics import fmean

from src.rvl_systems.hf_backend import HFLocalBackend
from src.rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
from src.rvl_systems.rollout import AsyncRolloutEngine, RolloutRequest
from src.rvl_systems.telemetry import Telemetry
from src.rvl_systems.verifier import ExactMatchVerifier


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    args = parser.parse_args()

    prompts = {
        "two_plus_two": "Answer with only the integer. What is 2 + 2?",
        "one_plus_two": "Answer with only the integer. What is 1 + 2?",
    }
    answers = {"two_plus_two": "4", "one_plus_two": "3"}

    backend = HFLocalBackend(
        args.model,
        max_new_tokens=args.max_new_tokens,
        precision=args.precision,
    )
    telemetry = Telemetry()
    rollouts = AsyncRolloutEngine(backend, max_concurrency=1, telemetry=telemetry)
    verifier = ExactMatchVerifier(answers, telemetry=telemetry)
    trainer = HFCausalLMGRPOTrainer(
        backend.model,
        config=HFTTrainerConfig(learning_rate=args.learning_rate),
    )

    history = []
    for step in range(args.steps):
        generations = await rollouts.run([
            RolloutRequest(
                prompt_id=prompt_id,
                prompt=prompt,
                samples=args.samples,
                temperature=1.0,
                seed=20261006 + step * 100 + i,
            )
            for i, (prompt_id, prompt) in enumerate(prompts.items())
        ])
        verified = list(await asyncio.gather(*(verifier.verify(g) for g in generations)))
        train_metrics = trainer.train_step(verified)
        history.append({
            "step": step,
            "reward": fmean(x.reward for x in verified),
            **train_metrics,
        })

    print(json.dumps({
        "model": args.model,
        "device": backend.resolved_device,
        "precision": backend.resolved_precision,
        "history": history,
        "telemetry": telemetry.snapshot(),
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
