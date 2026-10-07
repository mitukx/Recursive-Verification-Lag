"""Synthetic contract fixtures only; these are never published as GPU evidence."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_colab_rlvr_pilot import LOCK, RESEARCH_SHA, RUNNER, summarize


class ColabPilotSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lock = json.loads(LOCK.read_text())
        # Match the real runner: source lock hash and reformatted protocol copy
        # intentionally differ even though their decoded JSON is identical.
        (self.root / "protocol.json").write_text(json.dumps(self.lock, indent=2, sort_keys=True) + "\n")
        self.write("environment.json", {"systems_source_sha": self.lock["systems_source_sha"],
                   "research_source_sha": RESEARCH_SHA, "cuda": "test-fixture"})
        (self.root / "nvidia-smi.txt").write_text("SYNTHETIC UNIT TEST, NOT HARDWARE EVIDENCE")
        self.write("task_manifest.json", {"model_revision": self.lock["model_revision"],
                   "dataset_revision": self.lock["dataset_revision"],
                   "selection": self.lock["split_selection"],
                   "train": [[f"train-{i}", "synthetic prompt", "1"] for i in range(16)],
                   "evaluation": [[f"test-{i}", "synthetic prompt", "1"] for i in range(16)]})
        baseline = self.predictions(8)
        self.write("baseline.json", baseline)
        self.write("zero_update_repeat.json", baseline)
        self.write("zero_update_control.json", {"identical": True, "tasks": 16})
        summaries = []
        for seed in self.lock["seeds"]:
            for arm in self.lock["arms"]:
                self.write(f"{seed}_{arm}_terminal.json", self.predictions(8))
                self.write(f"{seed}_{arm}_history.json", [
                    {"step": i, "seed": seed, "arm": arm, "loss": 0, "grad_norm": 0,
                     "train_wall_s": 1, "parameter_probe_max_abs_change": 0,
                     "max_abs_log_ratio": 0, "clip_fraction": 0,
                     "nonconstant_reward_group": False, "true_reward_mean": 0, "response_tokens": 4}
                    for i in range(4)])
                rollouts = [{"generation": {"prompt_id": f"train-{step}", "response": "#### 0", "token_count": 1},
                             "reward": 0, "metadata": {"arm": arm, "true_reward_evaluation_only": 0}}
                            for step in range(4) for _ in range(4)]
                (self.root / f"{seed}_{arm}_rollouts.jsonl").write_text("".join(json.dumps(v) + "\n" for v in rollouts))
                summaries.append({"seed": seed, "arm": arm, "baseline_accuracy": .5,
                                  "terminal_accuracy": .5, "accuracy_delta": 0, "nonconstant_groups": 0,
                                  "max_parameter_probe_change": 0, "peak_allocated_gpu_bytes": 1, "wall_s": 4})
        self.write("summary.json", summaries)
        self.write("completion.json", {"status": "completed", "all_seed_arm_runs": 6})
        self.manifest()

    @staticmethod
    def predictions(correct_count):
        return [{"task_id": f"test-{i}", "expected": "1", "response": f"#### {int(i < correct_count)}",
                 "correct": float(i < correct_count)} for i in range(16)]

    def write(self, name, obj):
        (self.root / name).write_text(json.dumps(obj))

    def manifest(self):
        self.write("manifest.json", {
            "files": {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in self.root.rglob("*") if p.is_file() and p.name != "manifest.json"},
            "protocol_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest()})

    def test_null_result_stays_null_even_when_completed(self):
        result = summarize(self.root)
        self.assertEqual(result["status"], "completed_small_gpu_pilot")
        self.assertFalse(result["frontier_or_hiring_readiness_claim"])
        self.assertEqual(result["trusted_minus_shuffled"]["mean"], 0)
        self.assertTrue(all(r["nonzero_gradient_updates"] == 0 for r in result["runs"]))

    def test_source_hash_is_distinct_from_reformatted_copy(self):
        self.assertNotEqual(hashlib.sha256(LOCK.read_bytes()).hexdigest(),
                            hashlib.sha256((self.root / "protocol.json").read_bytes()).hexdigest())
        self.assertEqual(summarize(self.root)["status"], "completed_small_gpu_pilot")

    def test_rehashed_protocol_change_cannot_pass_source_lock(self):
        changed = dict(self.lock, steps=8)
        self.write("protocol.json", changed)
        self.manifest()
        with self.assertRaisesRegex(ValueError, "source protocol"):
            summarize(self.root)

    def test_incomplete_failure_retained_without_positive_metrics(self):
        (self.root / "43_trusted_reward_terminal.json").unlink()
        self.write("failure.json", {"status": "failed", "traceback": "synthetic OOM"})
        self.manifest()
        result = summarize(self.root)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["retained_failure"]["status"], "failed")
        self.assertNotIn("arm_outcomes", result)

    def test_changed_file_rejected(self):
        self.write("baseline.json", self.predictions(16))
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            summarize(self.root)

    def test_rehashed_forged_correctness_still_rejected(self):
        rows = self.predictions(8)
        rows[0]["correct"] = 0
        self.write("baseline.json", rows)
        self.manifest()
        with self.assertRaisesRegex(ValueError, "retained prediction"):
            summarize(self.root)

    def test_rehashed_forged_accuracy_still_rejected(self):
        rows = json.loads((self.root / "summary.json").read_text())
        rows[0]["accuracy_delta"] = .5
        self.write("summary.json", rows)
        self.manifest()
        with self.assertRaisesRegex(ValueError, "raw predictions"):
            summarize(self.root)

    def test_rehashed_forged_rollout_reward_still_rejected(self):
        path = self.root / "17_trusted_reward_rollouts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["reward"] = 1
        path.write_text("".join(json.dumps(v) + "\n" for v in rows))
        self.manifest()
        with self.assertRaisesRegex(ValueError, "rollout rewards"):
            summarize(self.root)

    def test_manifest_cannot_escape_evidence_directory(self):
        metadata = json.loads((self.root / "manifest.json").read_text())
        metadata["files"]["../outside.json"] = "0" * 64
        self.write("manifest.json", metadata)
        with self.assertRaisesRegex(ValueError, "unsafe evidence path"):
            summarize(self.root)


if __name__ == "__main__":
    unittest.main()
