"""CPU/Mac runnable end-to-end RLVR systems demo."""

from __future__ import annotations

import argparse
import asyncio
import json

from src.rvl_systems.backends import ToyTabularBackend
from src.rvl_systems.grpo import GRPOConfig
from src.rvl_systems.pipeline import PipelineConfig, RLVRPipeline
from src.rvl_systems.refresh import AdaptiveRefreshController
from src.rvl_systems.telemetry import Telemetry
from src.rvl_systems.verifier import ExactMatchVerifier


async def main(rounds: int, samples: int) -> None:
    prompts = {
        "two_plus_two": "Return only the integer: 2 + 2",
        "one_plus_two": "Return only the integer: 1 + 2",
        "one_plus_one": "Return only the integer: 1 + 1",
    }
    answers = {"two_plus_two": "4", "one_plus_two": "3", "one_plus_one": "2"}
    telemetry = Telemetry()
    backend = ToyTabularBackend(actions=("0", "1", "2", "3", "4"))
    verifier = ExactMatchVerifier(answers, telemetry=telemetry)
    pipeline = RLVRPipeline(
        backend=backend,
        verifier=verifier,
        refresh=AdaptiveRefreshController(movement_budget=0.35, max_stale_steps=3),
        grpo=GRPOConfig(learning_rate=0.25),
        telemetry=telemetry,
        max_concurrency=8,
    )
    history = await pipeline.run(
        prompts,
        PipelineConfig(rounds=rounds, samples_per_prompt=samples, base_seed=20261006),
    )
    print(json.dumps({"history": history, "telemetry": telemetry.snapshot()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    asyncio.run(main(args.rounds, args.samples))
