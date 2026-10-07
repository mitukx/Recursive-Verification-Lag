"""Corrected, pre-update-gated single-GPU RL pilot with a separate evaluation split.

The original v1 runner and evidence remain unchanged. No model-generated code,
paid service, Drive mount or GitHub credential is used.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import datetime
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs/colab_rlvr_pilot_v2.json"


def save(root, name, value):
    (root / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def append(root, name, value):
    with (root / name).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def parameter_digest(model):
    """Bound host memory to one parameter; hash all named parameter bytes."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        array = parameter.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(memoryview(array).cast("B"))
        del array
    return digest.hexdigest()


def score_fresh_actions(trainer, generations):
    """Retain independently recomputable score differences before any update."""
    torch = trainer.torch
    model = trainer.model
    device = next(model.parameters()).device
    rows = []
    use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        with torch.no_grad():
            for generation in generations:
                meta = generation.metadata
                temperature = float(meta["sampling_temperature"])
                if (not math.isfinite(temperature) or temperature <= 0 or
                        meta["logprob_distribution"] != "neutral_temperature_scaled_policy"):
                    raise ValueError("neutral stochastic policy scores required")
                prompt, response, old_values = (meta[k] for k in
                    ("prompt_token_ids", "response_token_ids", "response_token_logprobs"))
                if not prompt or not response or len(response) != len(old_values):
                    raise ValueError("complete action and behavior scores required")
                sequence = torch.tensor([prompt + response], device=device, dtype=torch.long)
                logits = model(input_ids=sequence).logits[0, len(prompt)-1:len(prompt)-1+len(response)].float()
                target = torch.tensor(response, device=device, dtype=torch.long)
                current = torch.log_softmax(logits / temperature, -1).gather(1, target[:, None]).squeeze(1)
                old = torch.tensor(old_values, device=device, dtype=torch.float32)
                if not torch.isfinite(current).all() or not torch.isfinite(old).all():
                    raise FloatingPointError("nonfinite fresh behavior/learner scores")
                gap = current - old
                ratio = gap.exp()
                clipped = (ratio < 1-trainer.config.clip_eps) | (ratio > 1+trainer.config.clip_eps)
                rows.append({"learner_token_logprobs": current.cpu().tolist(),
                             "max_abs_log_ratio": float(gap.abs().max()),
                             "clip_fraction": float(clipped.float().mean())})
    finally:
        model.config.use_cache = use_cache
    return rows


def require_fresh_parity(rows, gate):
    if not rows or any(not math.isfinite(float(row[key])) for row in rows
                       for key in ("max_abs_log_ratio", "clip_fraction")):
        raise ValueError("finite fresh-policy diagnostics required")
    if (max(r["max_abs_log_ratio"] for r in rows) > gate["max_abs_log_ratio"] or
            max(r["clip_fraction"] for r in rows) > gate["max_clip_fraction"]):
        raise RuntimeError("fresh-policy gate failed before optimizer; retain evidence")


async def run(lock, systems, output):
    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from src.rvl_systems.hf_backend import HFLocalBackend
    from src.rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
    from src.rvl_systems.rlvr_benchmark import build_math_prompt, gsm8k_reference_answer, response_reward
    from src.rvl_systems.types import VerifiedGeneration

    actual = {name: importlib.metadata.version(name) for name in lock["dependencies"]}
    if actual != lock["dependencies"]:
        raise RuntimeError("locked experiment dependency mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU substitution prohibited")
    (output / "nvidia-smi.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    save(output, "environment.json", {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
         "device": torch.cuda.get_device_name(0), "dependencies": actual, "paid_compute_started": False,
         "systems_source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=systems, text=True).strip(),
         "research_source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
         "systems_source_files": {name: hashlib.sha256((systems/name).read_bytes()).hexdigest() for name in
              ("src/rvl_systems/hf_backend.py", "src/rvl_systems/hf_trainer.py", "src/rvl_systems/triton_grpo.py")}})
    model_path = snapshot_download(lock["model"], revision=lock["model_revision"])
    data = load_dataset(lock["dataset"], "main", revision=lock["dataset_revision"])
    def tasks(split, indices):
        return [(f"{split}-{i}", build_math_prompt(data[split][i]["question"]),
                 gsm8k_reference_answer(data[split][i]["answer"])) for i in indices]
    train, evaluation = tasks("train", lock["train_indices"]), tasks("test", lock["evaluation_indices"])
    save(output, "task_manifest.json", {"selection": lock["split_selection"], "train": train, "evaluation": evaluation,
         "model_revision": lock["model_revision"], "dataset_revision": lock["dataset_revision"]})
    def backend():
        result = HFLocalBackend(model_path, max_new_tokens=lock["max_new_tokens"], precision=lock["precision"])
        result.ensure_loaded()
        return result
    def timed_digest(model):
        torch.cuda.synchronize(); start = time.perf_counter()
        value = parameter_digest(model)
        torch.cuda.synchronize()
        return value, time.perf_counter()-start
    async def evaluate(b, filename):
        rows = []
        start = time.perf_counter()
        for index, (task, prompt, answer) in enumerate(evaluation):
            generation = (await b.generate(task, prompt, n=1, temperature=0., seed=100000+index))[0]
            row = {"task_id": task, "expected": answer, "response": generation.response,
                   "correct": response_reward(generation.response, answer),
                   "token_count": generation.token_count, "latency_s": generation.latency_s}
            rows.append(row); save(output, filename, rows)
            print("EVAL", filename, index+1, len(evaluation), flush=True)
        return rows, time.perf_counter()-start

    b = backend()
    initial_hash, hash_wall = timed_digest(b.model)
    save(output, "initial_model.json", {"parameter_sha256": initial_hash, "hash_wall_s": hash_wall,
         "generation_config": b.model.generation_config.to_dict()})
    baseline, baseline_wall = await evaluate(b, "baseline.json")
    repeat, repeat_wall = await evaluate(b, "zero_update_repeat.json")
    identical = [(r["response"], r["correct"]) for r in baseline] == [(r["response"], r["correct"]) for r in repeat]
    save(output, "zero_update_control.json", {"identical": identical, "tasks": len(baseline),
         "baseline_wall_s": baseline_wall, "repeat_wall_s": repeat_wall})
    if not identical:
        raise RuntimeError("zero-update greedy reproducibility failed")
    del b; gc.collect(); torch.cuda.empty_cache()
    before = {row["task_id"]: row["correct"] for row in baseline}
    summaries = []
    for seed in lock["seeds"]:
        for arm in lock["arms"]:
            print("START", seed, arm, flush=True)
            b = backend()
            before_hash, before_hash_wall = timed_digest(b.model)
            if before_hash != initial_hash:
                raise RuntimeError("condition does not start from identical full model parameters")
            trainer = HFCausalLMGRPOTrainer(b.model, config=HFTTrainerConfig(learning_rate=lock["learning_rate"]))
            permutation = list(train); random.Random(seed).shuffle(permutation)
            history = []; start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            for step in range(lock["steps"]):
                task, prompt, answer = permutation[step % len(permutation)]
                torch.cuda.synchronize(); tick = time.perf_counter()
                generations = await b.generate(task, prompt, n=lock["samples_per_prompt"],
                    temperature=lock["train_temperature"], seed=seed+step*10000)
                torch.cuda.synchronize(); generation_wall = time.perf_counter()-tick
                tick = time.perf_counter()
                true = [response_reward(g.response, answer) for g in generations]
                supplied = list(true)
                if arm == "within_prompt_shuffled_reward":
                    random.Random(seed+900000+step).shuffle(supplied)
                verified = [VerifiedGeneration(g, reward, 0., 0,
                    metadata={"true_reward_evaluation_only": original, "arm": arm})
                    for g, reward, original in zip(generations, supplied, true)]
                reward_wall = time.perf_counter()-tick
                # Persist actions first, so a numerical gate failure retains them.
                for sample in verified:
                    append(output, f"{seed}_{arm}_rollouts.jsonl", asdict(sample))
                torch.cuda.synchronize(); tick = time.perf_counter()
                fresh = score_fresh_actions(trainer, generations)
                torch.cuda.synchronize(); fresh_wall = time.perf_counter()-tick
                append(output, f"{seed}_{arm}_fresh_scores.jsonl", {"step": step, "scores": fresh})
                require_fresh_parity(fresh, lock["fresh_policy_gate"])
                append(output, "events.jsonl", {"seed": seed, "arm": arm, "step": step,
                    "event": "fresh_gate_passed_before_optimizer", "optimization_steps_completed": len(history)})
                torch.cuda.synchronize(); tick = time.perf_counter()
                use_cache = b.model.config.use_cache
                b.model.config.use_cache = False
                try:
                    metrics = trainer.train_step(verified)
                finally:
                    b.model.config.use_cache = use_cache
                torch.cuda.synchronize(); learn_wall = time.perf_counter()-tick
                row = {"step": step, "seed": seed, "arm": arm,
                       "true_reward_mean": sum(true)/len(true), "nonconstant_reward_group": len(set(true)) > 1,
                       "generation_wall_s": generation_wall, "fresh_score_wall_s": fresh_wall,
                       "reward_wall_s": reward_wall, "learn_wall_s": learn_wall,
                       "response_tokens": sum(g.token_count for g in generations), **metrics}
                history.append(row); save(output, f"{seed}_{arm}_history.json", history)
                append(output, "events.jsonl", {"seed": seed, "arm": arm, "step": step,
                    "event": "optimizer_completed", "optimization_steps_completed": len(history)})
                print("STEP", seed, arm, step, row, flush=True)
            after_hash, after_hash_wall = timed_digest(b.model)
            terminal, eval_wall = await evaluate(b, f"{seed}_{arm}_terminal.json")
            summary = {"seed": seed, "arm": arm, "baseline_accuracy": sum(before.values())/len(before),
                "terminal_accuracy": sum(r["correct"] for r in terminal)/len(terminal),
                "accuracy_delta": sum(r["correct"]-before[r["task_id"]] for r in terminal)/len(terminal),
                "nonconstant_groups": sum(h["nonconstant_reward_group"] for h in history),
                "initial_parameter_sha256": before_hash, "terminal_parameter_sha256": after_hash,
                "parameter_hash_wall_s": before_hash_wall+after_hash_wall,
                "terminal_evaluation_wall_s": eval_wall, "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
                "condition_wall_s": time.perf_counter()-start}
            summaries.append(summary); save(output, "summary.json", summaries)
            print("RESULT", summary, flush=True)
            del trainer, b; gc.collect(); torch.cuda.empty_cache()
    save(output, "completion.json", {"status": "completed", "conditions": len(summaries),
         "optimizer_steps": len(summaries)*lock["steps"], "claim": lock["scope"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock = json.loads(LOCK.read_text())
    save(output, "protocol.json", lock)
    (output / "source_protocol.json").write_bytes(LOCK.read_bytes())
    (output / "source_runner.py").write_bytes(Path(__file__).read_bytes())
    save(output, "attempt.json", {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 "paid_compute_started": False})
    systems = ROOT.parent / ("rvl-v2-systems-" + lock["systems_source_sha"][:12])
    try:
        if not systems.exists():
            subprocess.run(["git", "clone", "https://github.com/mitukx/Recursive-Verification-Lag.git", str(systems)], check=True)
            subprocess.run(["git", "checkout", "--detach", lock["systems_source_sha"]], cwd=systems, check=True)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=systems, text=True).strip()
        if head != lock["systems_source_sha"] or subprocess.check_output(["git", "status", "--porcelain"], cwd=systems):
            raise RuntimeError("systems checkout source mismatch or local modifications")
        sys.path.insert(0, str(systems))
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        asyncio.run(run(lock, systems, output))
    except BaseException:
        save(output, "failure.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise
    finally:
        files = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in output.rglob("*") if p.is_file() and p.name != "manifest.json"}
        save(output, "manifest.json", {"files": files, "protocol_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
             "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__ == "__main__":
    main()
