"""Synthetic integrity fixtures and offline tiny-model gates, not GPU results."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from scripts.run_colab_rlvr_pilot_v2 import parameter_digest, require_fresh_parity, score_fresh_actions
from scripts.summarize_colab_rlvr_pilot_v2 import LOCK, summarize

try:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    TORCH = True
except ImportError:
    TORCH = False


class CorrectedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lock = json.loads(LOCK.read_text())
        self.write("protocol.json", self.lock)
        (self.root / "source_protocol.json").write_bytes(LOCK.read_bytes())
        (self.root / "source_runner.py").write_bytes((LOCK.parent.parent / "scripts/run_colab_rlvr_pilot_v2.py").read_bytes())
        self.write("environment.json", {"systems_source_sha": self.lock["systems_source_sha"],
                   "cuda": "SYNTHETIC FIXTURE", "dependencies": self.lock["dependencies"], "paid_compute_started": False})
        train = [[f"train-{i}", "synthetic prompt", "1"] for i in self.lock["train_indices"]]
        evaluation = [[f"test-{i}", "synthetic prompt", "1"] for i in self.lock["evaluation_indices"]]
        self.write("task_manifest.json", {"selection": self.lock["split_selection"], "train": train, "evaluation": evaluation,
                   "model_revision": self.lock["model_revision"], "dataset_revision": self.lock["dataset_revision"]})
        baseline = [{"task_id": row[0], "expected": "1", "response": f"#### {int(i%2==0)}",
                     "correct": float(i%2==0)} for i,row in enumerate(evaluation)]
        self.write("baseline.json", baseline); self.write("zero_update_repeat.json", baseline)
        self.write("zero_update_control.json", {"identical": True, "tasks": len(baseline)})
        self.write("initial_model.json", {"parameter_sha256": "0"*64})
        summaries, events = [], []
        for seed in self.lock["seeds"]:
            for arm in self.lock["arms"]:
                prefix = f"{seed}_{arm}"
                permutation = list(train); random.Random(seed).shuffle(permutation)
                history, rollouts, fresh = [], [], []
                for step in range(self.lock["steps"]):
                    task = permutation[step][0]
                    for _ in range(4):
                        rollouts.append({"generation": {"prompt_id": task, "prompt": "synthetic prompt", "response": "#### 0",
                            "logprob": -3., "token_count": 2, "metadata": {"sampling_temperature": 1.,
                            "logprob_distribution": "neutral_temperature_scaled_policy", "response_token_ids": [3,4],
                            "response_token_logprobs": [-1.,-2.]}}, "reward": 0.,
                            "metadata": {"arm": arm, "true_reward_evaluation_only": 0.}})
                    fresh.append({"step": step, "scores": [{"learner_token_logprobs": [-1.,-2.],
                                 "max_abs_log_ratio": 0., "clip_fraction": 0.} for _ in range(4)]})
                    history.append({"step": step, "seed": seed, "arm": arm, "true_reward_mean": 0.,
                         "nonconstant_reward_group": False, "response_tokens": 8, "loss": 0., "grad_norm": 0.,
                         "max_abs_log_ratio": 0., "clip_fraction": 0., "generation_wall_s": 1.,
                         "fresh_score_wall_s": 1., "reward_wall_s": 1., "learn_wall_s": 1.})
                    events.extend([{"seed": seed, "arm": arm, "step": step, "event": name,
                                    "optimization_steps_completed": step+int(name=="optimizer_completed")}
                                   for name in ("fresh_gate_passed_before_optimizer", "optimizer_completed")])
                self.lines(prefix+"_rollouts.jsonl", rollouts); self.lines(prefix+"_fresh_scores.jsonl", fresh)
                self.write(prefix+"_history.json", history); self.write(prefix+"_terminal.json", baseline)
                summaries.append({"seed": seed, "arm": arm, "baseline_accuracy": .5, "terminal_accuracy": .5,
                    "accuracy_delta": 0., "initial_parameter_sha256": "0"*64, "terminal_parameter_sha256": "0"*64,
                    "nonconstant_groups": 0, "parameter_hash_wall_s": 1., "terminal_evaluation_wall_s": 1.,
                    "condition_wall_s": 1., "peak_allocated_gpu_bytes": 1})
        self.write("summary.json", summaries); self.lines("events.jsonl", events)
        self.write("completion.json", {"status": "completed", "conditions": 6, "optimizer_steps": 24})
        self.manifest()

    def write(self, name, value):
        (self.root/name).write_text(json.dumps(value))

    def lines(self, name, values):
        (self.root/name).write_text("".join(json.dumps(v)+"\n" for v in values))

    def manifest(self):
        self.write("manifest.json", {"files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in self.root.iterdir() if p.name != "manifest.json"},
            "protocol_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256((self.root/"source_runner.py").read_bytes()).hexdigest()})

    def test_complete_null_result_stays_null(self):
        result = summarize(self.root)
        self.assertEqual(result["optimizer_steps"], 24)
        self.assertEqual(result["nonzero_gradient_updates"], 0)
        self.assertEqual(result["trusted_minus_shuffled"]["mean"], 0)
        self.assertFalse(result["frontier_capability_claim"])

    def test_rehashed_bad_scores_cannot_be_promoted(self):
        name = "17_trusted_reward_fresh_scores.jsonl"
        rows = [json.loads(v) for v in (self.root/name).read_text().splitlines()]
        row = rows[0]["scores"][0]; row["learner_token_logprobs"] = [-.9,-1.9]
        row["max_abs_log_ratio"] = float(np.max(np.abs(np.array(row["learner_token_logprobs"],dtype=np.float32)-[-1.,-2.])))
        # Preserve the actual float32 subtraction the learner uses.
        row["max_abs_log_ratio"] = float(np.max(np.abs(np.array(row["learner_token_logprobs"],dtype=np.float32)-np.array([-1.,-2.],dtype=np.float32))))
        self.lines(name, rows); self.manifest()
        with self.assertRaisesRegex(ValueError, "failed fresh-policy gate"):
            summarize(self.root)

    def test_rehashed_optimizer_before_gate_is_rejected(self):
        rows = [json.loads(v) for v in (self.root/"events.jsonl").read_text().splitlines()]
        rows[0], rows[1] = rows[1], rows[0]
        self.lines("events.jsonl", rows); self.manifest()
        with self.assertRaisesRegex(ValueError, "optimizer must follow"):
            summarize(self.root)

    def test_failed_attempt_retains_failure_without_accuracy(self):
        self.write("failure.json", {"status": "failed", "traceback": "synthetic gate failure"}); self.manifest()
        result = summarize(self.root)
        self.assertEqual(result["status"], "incomplete")
        self.assertNotIn("baseline_accuracy", result)
        self.assertIn("synthetic", result["retained_failure"]["traceback"])

    def test_rehashed_forged_prediction_flag_rejected(self):
        rows = json.loads((self.root/"baseline.json").read_text()); rows[0]["correct"] = 0.
        self.write("baseline.json", rows); self.manifest()
        with self.assertRaisesRegex(ValueError, "retained prediction"):
            summarize(self.root)


@unittest.skipUnless(TORCH, "optional real PyTorch/Transformers dependencies")
class FreshPolicyGateTests(unittest.TestCase):
    def test_real_model_scores_gate_before_update_and_reject_mismatched_behavior(self):
        from src.rvl_systems.hf_trainer import HFCausalLMGRPOTrainer
        from src.rvl_systems.types import Generation
        torch.manual_seed(17)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=32,n_positions=32,n_embd=16,n_layer=1,n_head=2,
            resid_pdrop=0,embd_pdrop=0,attn_pdrop=0))
        trainer = HFCausalLMGRPOTrainer(model)
        p,r = [1,2],[3,4]
        with torch.no_grad():
            logits = model(torch.tensor([p+r])).logits[0,1:3]
            old = torch.log_softmax(logits.float()/.7,-1).gather(1,torch.tensor(r)[:,None]).squeeze(1).tolist()
        generation = Generation("p","prompt","response",sum(old),2,0., {"prompt_token_ids": p,"response_token_ids": r,
            "response_token_logprobs": old,"sampling_temperature": .7,"logprob_distribution":"neutral_temperature_scaled_policy"})
        before = parameter_digest(model); cache = model.config.use_cache
        good = score_fresh_actions(trainer,[generation])
        require_fresh_parity(good, {"max_abs_log_ratio": 1e-5,"max_clip_fraction":0.})
        bad = copy.deepcopy(generation); bad.metadata["response_token_logprobs"][0] += .1
        with self.assertRaisesRegex(RuntimeError,"before optimizer"):
            require_fresh_parity(score_fresh_actions(trainer,[bad]), {"max_abs_log_ratio":.002,"max_clip_fraction":0.})
        self.assertEqual(parameter_digest(model),before)
        self.assertEqual(model.config.use_cache,cache)
        self.assertFalse(trainer.optimizer.state)
        # Hashing must cover a parameter beyond the old first-layer probe.
        with torch.no_grad():list(model.parameters())[-1].flatten()[0].add_(.1)
        self.assertNotEqual(parameter_digest(model),before)


if __name__ == "__main__":
    unittest.main()
