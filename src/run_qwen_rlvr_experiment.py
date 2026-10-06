from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path
from statistics import fmean

from src.rvl_systems.benchmark_report import BenchmarkReport
from src.rvl_systems.hf_backend import HFLocalBackend
from src.rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
from src.rvl_systems.lab.promotion import EvalSample, PromotionLedger, PromotionPolicy, evaluate_promotion
from src.rvl_systems.rlvr_benchmark import (
    RLVRTask,
    load_gsm8k_tasks,
    load_jsonl_tasks,
    response_reward,
)
from src.rvl_systems.rollout import AsyncRolloutEngine, RolloutRequest
from src.rvl_systems.telemetry import Telemetry
from src.rvl_systems.verifier import FunctionalVerifier


async def evaluate(
    engine: AsyncRolloutEngine,
    tasks: list[RLVRTask],
    *,
    seed: int,
) -> tuple[float, list[dict[str, object]]]:
    predictions: list[dict[str, object]] = []
    correct = 0
    for index, task in enumerate(tasks):
        generation = (
            await engine.run(
                [
                    RolloutRequest(
                        prompt_id=task.task_id,
                        prompt=task.prompt,
                        samples=1,
                        temperature=0.0,
                        seed=seed + index,
                    )
                ]
            )
        )[0]
        reward = response_reward(generation.response, task.answer)
        correct += int(reward)
        predictions.append(
            {
                "task_id": task.task_id,
                "expected": task.answer,
                "response": generation.response,
                "correct": bool(reward),
                "token_count": generation.token_count,
                "latency_s": generation.latency_s,
            }
        )
    return correct / max(1, len(tasks)), predictions


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Held-out RLVR experiment for Qwen/Hugging Face causal LMs. "
            "Evaluates greedily before and after GRPO-style verified updates."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", choices=["gsm8k", "jsonl"], default="gsm8k")
    parser.add_argument("--tasks-jsonl")
    parser.add_argument("--train-limit", type=int, default=32)
    parser.add_argument("--eval-limit", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--train-temperature", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument(
        "--precision",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="auto",
    )
    parser.add_argument(
        "--objective-backend",
        choices=["torch", "triton"],
        default="torch",
    )
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--output-dir", default="artifacts/qwen-rlvr")
    parser.add_argument("--save-model-dir")
    parser.add_argument("--transactional-promotion", action="store_true")
    parser.add_argument("--promotion-min-delta", type=float, default=-0.02)
    parser.add_argument("--promotion-max-regression", type=float, default=0.10)
    parser.add_argument("--promotion-max-new-failures", type=int, default=2)
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.prompts_per_step <= 0 or args.samples_per_prompt <= 1:
        raise ValueError(
            "prompts-per-step must be positive and samples-per-prompt > 1"
        )
    if args.train_temperature <= 0:
        raise ValueError("train-temperature must be positive")

    if args.dataset == "gsm8k":
        train_tasks, eval_tasks = load_gsm8k_tasks(
            train_limit=args.train_limit,
            eval_limit=args.eval_limit,
            seed=args.seed,
        )
    else:
        if not args.tasks_jsonl:
            parser.error("--tasks-jsonl is required for --dataset jsonl")
        tasks = load_jsonl_tasks(args.tasks_jsonl)
        train_tasks = [task for task in tasks if task.split == "train"]
        eval_tasks = [
            task for task in tasks if task.split in {"eval", "test", "validation"}
        ]
        if not train_tasks or not eval_tasks:
            raise ValueError("JSONL must contain both train and eval/test tasks")
        train_tasks = train_tasks[: args.train_limit]
        eval_tasks = eval_tasks[: args.eval_limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    telemetry = Telemetry()
    backend = HFLocalBackend(
        args.model,
        max_new_tokens=args.max_new_tokens,
        precision=args.precision,
    )
    engine = AsyncRolloutEngine(
        backend,
        max_concurrency=1,
        telemetry=telemetry,
    )
    answers = {task.task_id: task.answer for task in train_tasks}
    verifier = FunctionalVerifier(
        lambda generation: response_reward(
            generation.response,
            answers[generation.prompt_id],
        ),
        telemetry=telemetry,
    )
    trainer = HFCausalLMGRPOTrainer(
        backend.model,
        config=HFTTrainerConfig(
            learning_rate=args.learning_rate,
            objective_backend=args.objective_backend,
        ),
    )

    promotion_policy = PromotionPolicy(
        min_samples=len(eval_tasks),
        min_mean_delta=args.promotion_min_delta,
        max_family_regression=args.promotion_max_regression,
        max_uncompensated_new_failures=args.promotion_max_new_failures,
    )
    promotion_ledger = PromotionLedger(output_dir / "promotion-ledger.jsonl")

    started = time.perf_counter()
    before_accuracy, before_predictions = await evaluate(
        engine,
        eval_tasks,
        seed=args.seed + 100_000,
    )
    write_jsonl(
        output_dir / "predictions-before.jsonl",
        before_predictions,
    )

    rng = random.Random(args.seed)
    shuffled = list(train_tasks)
    rng.shuffle(shuffled)
    cursor = 0
    history: list[dict[str, float | int]] = []

    for step in range(args.steps):
        batch: list[RLVRTask] = []
        for _ in range(args.prompts_per_step):
            if cursor >= len(shuffled):
                rng.shuffle(shuffled)
                cursor = 0
            batch.append(shuffled[cursor])
            cursor += 1

        generations = await engine.run(
            [
                RolloutRequest(
                    prompt_id=task.task_id,
                    prompt=task.prompt,
                    samples=args.samples_per_prompt,
                    temperature=args.train_temperature,
                    seed=args.seed + step * 10_000 + index,
                )
                for index, task in enumerate(batch)
            ]
        )
        verified = list(
            await asyncio.gather(
                *(verifier.verify(generation) for generation in generations)
            )
        )

        incumbent_state = trainer.snapshot_training_state() if args.transactional_promotion else None
        incumbent_accuracy = None
        incumbent_predictions = None
        if args.transactional_promotion:
            incumbent_accuracy, incumbent_predictions = await evaluate(
                engine,
                eval_tasks,
                seed=args.seed + 300_000 + step,
            )

        train_metrics = trainer.train_step(verified)

        promotion = None
        if args.transactional_promotion:
            candidate_accuracy, candidate_predictions = await evaluate(
                engine,
                eval_tasks,
                seed=args.seed + 300_000 + step,
            )
            eval_samples = [
                EvalSample(
                    task_id=before_row["task_id"],
                    family=0,
                    incumbent_reward=float(before_row["correct"]),
                    candidate_reward=float(after_row["correct"]),
                )
                for before_row, after_row in zip(incumbent_predictions, candidate_predictions)
                if before_row["task_id"] == after_row["task_id"]
            ]
            if len(eval_samples) != len(eval_tasks):
                raise RuntimeError("held-out evaluation identity mismatch")
            promotion = evaluate_promotion(
                eval_samples,
                incumbent_version=step,
                candidate_version=step + 1,
                policy=promotion_policy,
            )
            promotion_ledger.append(promotion)
            if not promotion.accepted:
                trainer.restore_training_state(incumbent_state)
                # HFLocalBackend serves trainer.model in this experiment, so
                # restoring the trainer model also restores rollout weights.
            train_metrics.update({
                "promotion_accepted": int(promotion.accepted),
                "promotion_incumbent_accuracy": incumbent_accuracy,
                "promotion_candidate_accuracy": candidate_accuracy,
                "promotion_delta": promotion.mean_delta,
                "promotion_evidence_sha256": promotion.evidence_sha256,
                "promotion_reasons": list(promotion.reasons),
            })

        group_rewards: dict[str, list[float]] = {}
        for item in verified:
            group_rewards.setdefault(
                item.generation.prompt_id,
                [],
            ).append(item.reward)
        nonconstant_groups = sum(
            len(set(rewards)) > 1
            for rewards in group_rewards.values()
        )
        history.append(
            {
                "step": step,
                "rollout_reward": fmean(item.reward for item in verified),
                "nonconstant_reward_groups": nonconstant_groups,
                **train_metrics,
            }
        )

    after_accuracy, after_predictions = await evaluate(
        engine,
        eval_tasks,
        seed=args.seed + 200_000,
    )
    write_jsonl(
        output_dir / "predictions-after.jsonl",
        after_predictions,
    )
    wall_s = time.perf_counter() - started

    if args.save_model_dir:
        model_dir = Path(args.save_model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        backend.model.save_pretrained(model_dir)
        backend.tokenizer.save_pretrained(model_dir)

    metrics: dict[str, float | int | str] = {
        "before_accuracy": before_accuracy,
        "after_accuracy": after_accuracy,
        "accuracy_delta": after_accuracy - before_accuracy,
        "train_examples": len(train_tasks),
        "eval_examples": len(eval_tasks),
        "steps": args.steps,
        "samples_per_prompt": args.samples_per_prompt,
        "prompts_per_step": args.prompts_per_step,
        "wall_s": wall_s,
        "last_train_reward": float(history[-1]["rollout_reward"]),
        "nonconstant_reward_groups_total": int(
            sum(int(row["nonconstant_reward_groups"]) for row in history)
        ),
        "device": backend.resolved_device,
        "precision": backend.resolved_precision,
        "objective_backend": args.objective_backend,
        "transactional_promotion": int(args.transactional_promotion),
        "promotion_records": promotion_ledger.seq,
        "promotion_head_sha256": promotion_ledger.head,
    }
    report = BenchmarkReport(
        name="heldout-qwen-rlvr",
        metrics=metrics,
        config={
            "dataset": args.dataset,
            "model": args.model,
            "train_limit": len(train_tasks),
            "eval_limit": len(eval_tasks),
            "max_new_tokens": args.max_new_tokens,
            "learning_rate": args.learning_rate,
            "train_temperature": args.train_temperature,
            "seed": args.seed,
            "transactional_promotion": args.transactional_promotion,
            "promotion_min_delta": args.promotion_min_delta,
            "promotion_max_regression": args.promotion_max_regression,
            "promotion_max_new_failures": args.promotion_max_new_failures,
        },
        git_sha=os.environ.get("GITHUB_SHA", "unknown"),
        model=args.model,
    )
    report.write_json(output_dir / "benchmark.json")
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "telemetry.json").write_text(
        json.dumps(telemetry.snapshot(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report.payload(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
