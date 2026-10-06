import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.validate_qwen_alignment_bridge_evidence import (
    sha256,
    validate_evidence,
)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def drift_row(prompt_id, candidate_index, arm, old, new, proxy, trusted):
    old = np.asarray(old, float)
    new = np.asarray(new, float)
    raw = new - old
    clipped = np.clip(raw, -20.0, 20.0)
    k3 = np.exp(clipped) - 1.0 - clipped
    return {
        "candidate_index": candidate_index,
        "prompt_id": prompt_id,
        "response": f"response-{candidate_index}",
        "response_token_ids": list(range(10, 10 + len(old))),
        "proxy_reward": proxy,
        "trusted_reward": trusted,
        "arm": arm,
        "old_token_logprobs": old.tolist(),
        "new_token_logprobs": new.tolist(),
        "raw_log_ratio": raw.tolist(),
        "clipped_log_ratio": clipped.tolist(),
        "token_k3": k3.tolist(),
        "mean_k3": float(k3.mean()),
        "max_abs_log_ratio": float(np.abs(clipped).max()),
        "mean_abs_log_ratio": float(np.abs(clipped).mean()),
        "tokens": len(old),
    }


def drift_summary(rows):
    k3 = [x for row in rows for x in row["token_k3"]]
    ratios = [abs(x) for row in rows for x in row["clipped_log_ratio"]]
    return {
        "post_update_k3": float(np.mean(k3)),
        "post_update_max_abs_log_ratio": float(np.max(ratios)),
        "post_update_mean_abs_log_ratio": float(np.mean(ratios)),
        "post_update_tokens": len(k3),
    }


class QwenBridgeEvidenceValidatorTest(unittest.TestCase):
    def make_completed_fixture(self, base: Path):
        protocol = base / "protocol.json"
        lock = {
            "systems_source_sha": "systems-sha",
            "seeds": [17],
        }
        write_json(protocol, lock)
        root = base / "evidence"
        seed = root / "seed-17"
        seed.mkdir(parents=True)

        write_json(
            root / "environment.json",
            {
                "systems_source_sha": "systems-sha",
                "research_source_sha": "research-sha",
                "device": "UnitTest GPU",
                "device_count": 1,
                "dependencies": {
                    "torch": "x",
                    "transformers": "x",
                    "datasets": "x",
                    "accelerate": "x",
                    "huggingface_hub": "x",
                    "numpy": "x",
                },
            },
        )
        (root / "nvidia-smi.txt").write_text("GPU")
        (root / "pip-freeze.txt").write_text("numpy==x\n")
        write_json(root / "task_manifest.json", {"unit": True})
        write_json(seed / "training_geometry.json", {"unit": True})

        calibration_rows = {}
        effect_rows = {}
        for arm, new in (("harmful", [-0.9]), ("benign", [-1.1])):
            cal = [drift_row("calibration-0", 0, arm, [-1.0], new, 0.25, 1.0)]
            eff = [drift_row("effect-0", 0, arm, [-1.0], new, 0.25, 1.0)]
            cal_file = f"calibration_token_drift/{arm}-lr-1e-06.jsonl"
            eff_file = f"{arm}_effect_token_drift.jsonl"
            write_jsonl(seed / cal_file, cal)
            write_jsonl(seed / eff_file, eff)
            calibration_rows[arm] = (cal_file, drift_summary(cal))
            effect_rows[arm] = (eff_file, drift_summary(eff))

        grid = {}
        selected = {}
        for arm in ("harmful", "benign"):
            cal_file, summary = calibration_rows[arm]
            row = {
                "lr": 1e-6,
                "wall_s": 0.1,
                "token_drift_file": cal_file,
                "loss": 0.0,
                "mean_reward": 0.0,
                "samples": 1.0,
                "grad_norm": 1.0,
                "clip_fraction": 0.0,
                "max_abs_log_ratio": 0.0,
                "behavior_kl_estimate": 0.0,
                **summary,
            }
            grid[arm] = [row]
            selected[arm] = row
        write_json(
            seed / "drift_calibration.json",
            {"grid": grid, "target_k3": min(x[0]["post_update_k3"] for x in grid.values()), "selected": selected},
        )

        generations = [
            {
                "prompt_id": "evaluation-0",
                "prompt": "p",
                "response": "correct",
                "logprob": -1.0,
                "token_count": 2,
                "latency_s": 0.0,
                "metadata": {
                    "response_token_ids": [1, 3],
                    "response_token_logprobs": [-0.4, -0.6],
                },
            },
            {
                "prompt_id": "evaluation-0",
                "prompt": "p",
                "response": "wrong",
                "logprob": -2.0,
                "token_count": 1,
                "latency_s": 0.0,
                "metadata": {
                    "response_token_ids": [2],
                    "response_token_logprobs": [-2.0],
                },
            },
        ]
        write_jsonl(
            seed / "evaluation_candidates_unscored.jsonl",
            [{"task_id": "evaluation-0", "generations": generations}],
        )
        write_jsonl(
            seed / "evaluation_candidate_trusted_labels.jsonl",
            [
                {"task_id": "evaluation-0", "candidate_index": 0, "candidate_id": "evaluation-0:candidate-0", "trusted_reward": 1.0},
                {"task_id": "evaluation-0", "candidate_index": 1, "candidate_id": "evaluation-0:candidate-1", "trusted_reward": 0.0},
            ],
        )

        eval_posts = {
            "harmful": [[-0.7, -0.7], [-1.8]],
            "benign": [[-0.3, -0.3], [-2.2]],
        }
        for arm, posts in eval_posts.items():
            rows = []
            for i, (g, post) in enumerate(zip(generations, posts)):
                baseline = g["metadata"]["response_token_logprobs"]
                rows.append(
                    {
                        "task_id": "evaluation-0",
                        "candidate_index": i,
                        "candidate_id": f"evaluation-0:candidate-{i}",
                        "arm": arm,
                        "response": g["response"],
                        "response_token_ids": g["metadata"]["response_token_ids"],
                        "baseline_token_logprobs": baseline,
                        "post_update_token_logprobs": post,
                        "baseline_sequence_logprob": float(np.sum(baseline)),
                        "post_update_sequence_logprob": float(np.sum(post)),
                        "baseline_mean_token_logprob": float(np.mean(baseline)),
                        "post_update_mean_token_logprob": float(np.mean(post)),
                    }
                )
            write_jsonl(seed / f"{arm}_evaluation_candidate_token_logprobs.jsonl", rows)
            write_json(seed / f"{arm}_terminal_greedy_unscored.json", [])

        write_jsonl(
            seed / "evaluation_preference.jsonl",
            [
                {
                    "task_id": "evaluation-0",
                    "arm": "harmful",
                    "trusted_rewards": [1.0, 0.0],
                    "informative": True,
                    "baseline_margin": 1.0,
                    "post_margin": 0.4,
                    "preference_shift": -0.6,
                },
                {
                    "task_id": "evaluation-0",
                    "arm": "benign",
                    "trusted_rewards": [1.0, 0.0],
                    "informative": True,
                    "baseline_margin": 1.0,
                    "post_margin": 1.6,
                    "preference_shift": 0.6,
                },
            ],
        )
        seed_summary = {
            "seed": 17,
            "selected_lrs": {"harmful": 1e-6, "benign": 1e-6},
            "target_calibration_k3": 0.0,
            "arms": {
                arm: {
                    "lr": 1e-6,
                    "train_metrics": {"grad_norm": 1.0},
                    "effect_drift": effect_rows[arm][1],
                    "effect_token_drift_file": effect_rows[arm][0],
                    "evaluation_candidate_logprob_file": f"{arm}_evaluation_candidate_token_logprobs.jsonl",
                }
                for arm in ("harmful", "benign")
            },
            "effect_k3_ratio": 1.0,
            "informative_evaluation_prompts": 1,
            "mean_preference_shift": {"harmful": -0.6, "benign": 0.6},
            "benign_minus_harmful_preference_shift": 1.2,
            "eligible_primary_seed": True,
            "greedy": {},
            "evaluation_access_before_lr_selection": 0,
        }
        write_json(seed / "seed_summary.json", seed_summary)
        write_json(
            root / "summary.json",
            {
                "eligible_seeds": 1,
                "required_eligible_seeds": 1,
                "primary_evidence_sufficient": True,
                "primary_mean_benign_minus_harmful_preference_shift": 1.2,
                "primary_direction_passed": True,
                "seed_results": [seed_summary],
                "claim_scope": "unit",
            },
        )

        files = {
            str(path.relative_to(root)): sha256(path)
            for path in root.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        write_json(
            root / "manifest.json",
            {
                "status": "completed",
                "protocol_sha256": sha256(protocol),
                "runner_sha256": "runner",
                "files": files,
            },
        )
        return root, protocol

    def test_completed_artifact_recomputes_raw_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, protocol = self.make_completed_fixture(Path(tmp))
            result = validate_evidence(root, protocol, expected_research_sha="research-sha")
            self.assertTrue(result["valid"])
            self.assertEqual(result["execution_status"], "completed")
            self.assertEqual(result["scientific_result"], "direction_passed")
            self.assertEqual(result["eligible_seeds"], 1)

    def test_rehashed_per_prompt_preference_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, protocol = self.make_completed_fixture(Path(tmp))
            path = root / "seed-17" / "evaluation_preference.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            rows[0]["preference_shift"] = -0.1
            write_jsonl(path, rows)
            files = {
                str(file.relative_to(root)): sha256(file)
                for file in root.rglob("*")
                if file.is_file() and file.name != "manifest.json"
            }
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["files"] = files
            write_json(root / "manifest.json", manifest)
            with self.assertRaises(AssertionError):
                validate_evidence(root, protocol, expected_research_sha="research-sha")

    def test_manifest_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, protocol = self.make_completed_fixture(Path(tmp))
            (root / "pip-freeze.txt").write_text("tampered\n")
            with self.assertRaises(AssertionError):
                validate_evidence(root, protocol, expected_research_sha="research-sha")

    def test_failed_execution_is_valid_retained_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            protocol = base / "protocol.json"
            write_json(protocol, {"systems_source_sha": "systems-sha", "seeds": [17]})
            root = base / "failed"
            root.mkdir()
            write_json(root / "failure.json", {"status": "failed", "traceback": "boom"})
            files = {"failure.json": sha256(root / "failure.json")}
            write_json(
                root / "manifest.json",
                {
                    "status": "failed",
                    "protocol_sha256": sha256(protocol),
                    "runner_sha256": "runner",
                    "files": files,
                },
            )
            result = validate_evidence(root, protocol)
            self.assertEqual(result["execution_status"], "failed")
            self.assertTrue(result["failure_retained"])


if __name__ == "__main__":
    unittest.main()
