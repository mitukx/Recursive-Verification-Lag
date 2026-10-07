"""Exploratory GPU numerical diagnosis, kept separate from the original pilot.

Run as python -m scripts.run_hf_behavior_parity_diagnostic. No optimizer update,
evaluation task selection, generated code execution or paid service is used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

from scripts.summarize_colab_rlvr_pilot import verify_manifest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/hf_behavior_parity_diagnostic_v1.json"


def save(root, name, obj):
    (root / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def run(pilot, output, config):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import RepetitionPenaltyLogitsProcessor
    from src.rvl_systems.hf_backend import HFLocalBackend
    from src.rvl_systems.hf_trainer import HFCausalLMGRPOTrainer
    from src.rvl_systems.types import VerifiedGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU substitution")
    evidence_manifest = verify_manifest(pilot)
    environment = json.loads((pilot / "environment.json").read_text())
    if environment["systems_source_sha"] != config["legacy_systems_source_sha"]:
        raise ValueError("original pilot systems source mismatch")
    legacy = [json.loads(line) for line in (pilot / "17_trusted_reward_rollouts.jsonl").read_text().splitlines()][:4]
    if len(legacy) != 4 or len({v["generation"]["prompt"] for v in legacy}) != 1:
        raise ValueError("complete first legacy group required")
    (output / "nvidia-smi.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    save(output, "environment.json", {"python": sys.version, "torch": torch.__version__,
         "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
         "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
         "paid_compute_started": False, "legacy_protocol_sha256": evidence_manifest["protocol_sha256"]})
    model_path = snapshot_download(config["model"], revision=config["model_revision"])
    backend = HFLocalBackend(model_path, max_new_tokens=config["max_new_tokens"], precision=config["precision"])
    backend.ensure_loaded()
    model = backend.model
    save(output, "model_generation_config.json", model.generation_config.to_dict())
    penalty = float(model.generation_config.repetition_penalty)
    processor = RepetitionPenaltyLogitsProcessor(penalty)
    replay = []
    model.eval()
    with torch.no_grad():
        for index, sample in enumerate(legacy):
            metadata = sample["generation"]["metadata"]
            prompt, response = metadata["prompt_token_ids"], metadata["response_token_ids"]
            sequence = torch.tensor([prompt + response], dtype=torch.long, device="cuda")
            logits = model(input_ids=sequence, use_cache=False).logits[0, len(prompt)-1:len(prompt)-1+len(response)].float()
            targets = torch.tensor(response, dtype=torch.long, device="cuda")
            raw = torch.log_softmax(logits, -1).gather(1, targets[:, None]).squeeze(1)
            penalized = []
            for token in range(len(response)):
                adjusted = processor(sequence[:, :len(prompt)+token], logits[token:token+1].clone())
                penalized.append(torch.log_softmax(adjusted, -1)[0, targets[token]])
            penalized = torch.stack(penalized)
            recorded = torch.tensor(metadata["response_token_logprobs"], dtype=torch.float32, device="cuda")
            row = {"sample": index, "prompt_token_ids": prompt, "response_token_ids": response,
                   "legacy_recorded_logprobs": recorded.cpu().tolist(), "raw_policy_logprobs": raw.cpu().tolist(),
                   "replayed_repetition_penalty_logprobs": penalized.cpu().tolist(),
                   "raw_policy_max_abs_log_ratio": float((raw-recorded).abs().max()),
                   "penalty_replay_max_abs_error": float((penalized-recorded).abs().max())}
            replay.append(row)
            save(output, "legacy_replay.json", replay)
            del logits, raw, penalized, recorded, sequence
    trainer = HFCausalLMGRPOTrainer(model)
    corrected = []
    prompt = legacy[0]["generation"]["prompt"]
    for temperature in config["corrected_temperatures"]:
        generations = backend._generate_sync("legacy-first-training-prompt", prompt,
                      config["samples_per_temperature"], temperature, config["seed"])
        rows = []
        with torch.no_grad():
            for generation in generations:
                # A unit advantage measures the probability ratio only. This
                # is not a task reward and no backward/optimizer step occurs.
                objective = trainer._sample_objective(VerifiedGeneration(generation, 0., 0., 0), 1.)
                rows.append({"generation": asdict(generation), "max_abs_log_ratio": float(objective[2]),
                             "clip_fraction": float(objective[1]), "behavior_kl_estimate": float(objective[3])})
        corrected.append({"temperature": temperature, "samples": rows})
        save(output, "corrected_samples.json", corrected)
        print("CORRECTED", temperature, [r["max_abs_log_ratio"] for r in rows], flush=True)
    legacy_max = max(row["raw_policy_max_abs_log_ratio"] for row in replay)
    penalty_error = max(row["penalty_replay_max_abs_error"] for row in replay)
    corrected_max = max(row["max_abs_log_ratio"] for group in corrected for row in group["samples"])
    result = {"status": "completed", "inherited_repetition_penalty": penalty,
              "legacy_raw_policy_max_abs_log_ratio": legacy_max, "legacy_penalty_replay_max_abs_error": penalty_error,
              "corrected_max_abs_log_ratio": corrected_max, "optimization_steps": 0,
              "penalty_reconstruction_passed": penalty_error <= config["legacy_repetition_replay_tolerance"],
              "corrected_parity_passed": corrected_max <= config["same_weights_max_abs_log_ratio_tolerance"],
              "capability_gain_claim": False, "scope": config["scope"]}
    save(output, "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if not result["penalty_reconstruction_passed"] or not result["corrected_parity_passed"]:
        raise RuntimeError("declared numerical diagnostic failed; retain results")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.pilot_evidence.resolve()):
        parser.error("diagnostic must not alter original evidence")
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(CONFIG.read_text())
    save(args.output, "protocol.json", config)
    try:
        run(args.pilot_evidence, args.output, config)
    except BaseException:
        save(args.output, "failure.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        hashes = {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in args.output.rglob("*") if p.is_file() and p.name != "manifest.json"}
        save(args.output, "manifest.json", {"files": hashes,
             "source_files": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
                  "src/rvl_systems/hf_backend.py", "src/rvl_systems/hf_trainer.py",
                  "scripts/run_hf_behavior_parity_diagnostic.py", "configs/hf_behavior_parity_diagnostic_v1.json")}})


if __name__ == "__main__":
    main()
