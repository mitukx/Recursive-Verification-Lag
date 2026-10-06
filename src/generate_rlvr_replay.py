from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from src.rvl_systems.hf_backend import HFLocalBackend
from src.rvl_systems.rlvr_benchmark import load_gsm8k_tasks, response_reward
from src.rvl_systems.verifier import FunctionalVerifier


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate immutable token-exact verified replay for distributed RL acceptance."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--precision", choices=["auto","fp32","fp16","bf16"], default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.tasks <= 0 or args.samples <= 1:
        raise ValueError("tasks must be positive and samples must be > 1")
    tasks, _ = load_gsm8k_tasks(
        train_limit=args.tasks,
        eval_limit=1,
        seed=args.seed,
    )
    backend = HFLocalBackend(
        args.model,
        max_new_tokens=args.max_new_tokens,
        precision=args.precision,
    )
    answers = {task.task_id: task.answer for task in tasks}
    verifier = FunctionalVerifier(
        lambda generation: response_reward(
            generation.response,
            answers[generation.prompt_id],
        )
    )

    rows = []
    for index, task in enumerate(tasks):
        generations = await backend.generate(
            task.task_id,
            task.prompt,
            n=args.samples,
            temperature=args.temperature,
            seed=args.seed + index,
        )
        rows.extend(await asyncio.gather(*(verifier.verify(g) for g in generations)))

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "model": args.model,
        "tasks": len(tasks),
        "samples_per_task": args.samples,
        "verified_generations": len(rows),
        "reward_mean": sum(row.reward for row in rows) / len(rows),
        "precision": backend.resolved_precision,
        "device": backend.resolved_device,
        "seed": args.seed,
        "output": str(path),
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
