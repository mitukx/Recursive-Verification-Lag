"""Generate verified token-exact replay from a remote vLLM/SGLang completion server."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from src.rvl_systems.lab.token_serving import TokenServingBackend
from src.rvl_systems.rlvr_benchmark import load_gsm8k_tasks, response_reward
from src.rvl_systems.verifier import FunctionalVerifier


DEFAULT_GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_diagnostics(rows) -> dict:
    by_prompt = defaultdict(list)
    total_tokens = 0
    for sample in rows:
        generation = sample.generation
        meta = generation.metadata
        prompt_ids = meta.get("prompt_token_ids")
        response_ids = meta.get("response_token_ids")
        logps = meta.get("response_token_logprobs")
        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise ValueError("missing exact prompt token IDs")
        if not isinstance(response_ids, list) or not response_ids:
            raise ValueError("missing exact response token IDs")
        if not isinstance(logps, list) or len(logps) != len(response_ids):
            raise ValueError("response token/logprob alignment mismatch")
        if generation.token_count != len(response_ids):
            raise ValueError("generation token count disagrees with response IDs")
        if abs(generation.logprob - sum(float(x) for x in logps)) > 1e-6:
            raise ValueError("sequence logprob disagrees with token logprobs")
        by_prompt[generation.prompt_id].append(float(sample.reward))
        total_tokens += generation.token_count
    informative = sum(len(set(values)) > 1 for values in by_prompt.values())
    return {
        "groups": len(by_prompt),
        "samples": len(rows),
        "generated_tokens": total_tokens,
        "informative_reward_groups": informative,
        "reward_mean": (
            sum(float(row.reward) for row in rows) / len(rows) if rows else 0.0
        ),
    }


async def generate(args) -> dict:
    if args.tasks <= 0 or args.samples <= 1:
        raise ValueError("tasks must be positive and samples must be > 1")
    tasks, _ = load_gsm8k_tasks(
        train_limit=args.tasks,
        eval_limit=1,
        seed=args.seed,
        revision=args.dataset_revision,
    )
    backend = TokenServingBackend(
        args.endpoint,
        args.model,
        max_new_tokens=args.max_new_tokens,
        timeout_s=args.timeout_s,
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

    diagnostics = replay_diagnostics(rows)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix(path.suffix + ".meta.json").exists():
        raise FileExistsError("refuse to overwrite token-exact replay evidence")
    path.write_text(
        "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "status": "remote_token_exact_verified_replay",
        "backend": "TokenServingBackend",
        "model": args.model,
        "dataset": "openai/gsm8k",
        "dataset_revision": args.dataset_revision,
        "tasks": len(tasks),
        "samples_per_task": args.samples,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "replay_sha256": sha256_file(path),
        **diagnostics,
        "claim_boundary": (
            "This establishes server-owned token/logprob replay provenance over the "
            "OpenAI-compatible completion path. It is not a model-quality claim."
        ),
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset-revision", default=DEFAULT_GSM8K_REVISION)
    p.add_argument("--tasks", type=int, default=8)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=20261007)
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = asyncio.run(generate(args))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
