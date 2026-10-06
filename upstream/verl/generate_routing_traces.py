"""Generate reproducible homogeneous and heterogeneous routing traces."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


def make_prompt(tokenizer, target_tokens: int, index: int) -> tuple[str, int]:
    seed_text = (
        f"Request {index}. Continue this neutral workload context. "
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. "
    )
    raw = (seed_text * ((target_tokens // 12) + 8)).strip()
    ids = tokenizer.encode(raw, add_special_tokens=False)[:target_tokens]
    prompt = tokenizer.decode(ids, skip_special_tokens=True)
    exact = len(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt, exact


def write_trace(path: Path, tokenizer, *, requests: int, seed: int, mode: str):
    rng = random.Random(seed)
    rows = []
    for i in range(requests):
        if mode == "homogeneous":
            prompt_target, decode = 512, 256
        elif mode == "heterogeneous":
            if rng.random() < 0.20:
                prompt_target = rng.choice([1536, 2048, 2560])
                decode = rng.choice([384, 512, 640])
            else:
                prompt_target = rng.choice([128, 256, 384, 512, 768])
                decode = rng.choice([64, 96, 128, 192, 256])
        else:
            raise ValueError(mode)
        prompt, prompt_tokens = make_prompt(tokenizer, prompt_target, i)
        rows.append({
            "request_id": f"{mode}-{i:05d}",
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "max_tokens": decode,
            "seed": seed + i,
            "session_id": None,
            "ignore_eos": True,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--requests", type=int, default=96)
    p.add_argument("--seed", type=int, default=20261007)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=False,
    )
    for mode in ("homogeneous", "heterogeneous"):
        write_trace(
            args.output_dir / f"{mode}.jsonl",
            tokenizer,
            requests=args.requests,
            seed=args.seed,
            mode=mode,
        )


if __name__ == "__main__":
    main()
