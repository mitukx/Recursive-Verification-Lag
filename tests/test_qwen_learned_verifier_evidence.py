import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.validate_qwen_alignment_bridge_evidence import sha256
from scripts.validate_qwen_learned_verifier_evidence import (
    ARMS,
    recompute_effect_geometry,
    spearman,
    validate_evidence,
)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def drift_row(prompt_id, candidate_index, arm, old, new):
    old = np.asarray(old, float)
    new = np.asarray(new, float)
    raw = new - old
    clipped = np.clip(raw, -20.0, 20.0)
    k3 = np.exp(clipped) - 1.0 - clipped
    return {
        "candidate_index": candidate_index,
        "prompt_id": prompt_id,
        "response": f"{arm}-response-{candidate_index}",
        "response_token_ids": list(range(10, 10 + len(old))),
        "proxy_reward": 0.5,
        "trusted_reward": 1.0,
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


class LearnedVerifierEvidenceValidatorTest(unittest.TestCase):
    def test_geometry_exposes_pooled_positive_with_negative_grpo_alignment(self):
        rows = []
        groups = [
            ("high", [1.0, 1.0, 0.0, 1.0], [0.9, 0.8, 1.0, 0.7]),
            ("low", [0.0, 0.0, 1.0, 0.0], [0.2, 0.3, 0.1, 0.4]),
        ]
        for task_id, labels, fresh in groups:
            for index, (y, v) in enumerate(zip(labels, fresh)):
                rows.append(
                    {
                        "task_id": task_id,
                        "candidate_index": index,
                        "candidate_id": f"{task_id}:candidate-{index}",
                        "trusted_reward": y,
                        "oracle_score": y,
                        "fresh_score": v,
                        "stale_score": v,
                        "shuffled_score": v,
                    }
                )
        geometry = recompute_effect_geometry(
            rows, advantage_eps=1e-6, clip_advantage=5.0
        )["fresh"]
        self.assertGreater(geometry["cov_y_v"], 0.0)
        self.assertLess(
            geometry["grpo_aligned"]["within_prompt_cov_y_v_occurrence_weighted"],
            0.0,
        )
        self.assertLess(
            geometry["grpo_aligned"]["mean_prompt_cov_y_grpo_advantage"],
            0.0,
        )
        self.assertAlmostEqual(
            geometry["grpo_aligned"]["pooled_cov_reconstruction_error"],
            0.0,
            places=12,
        )

    def make_completed_fixture(self, base: Path):
        protocol = base / "protocol.json"
        lock = {
            "systems_source_sha": "systems-sha",
            "seeds": [17],
            "drift_calibration": {"learning_rate_grid": [1e-6]},
            "terminal_evaluation": {"minimum_informative_prompts": 1},
            "scope": "unit",
        }
        write_json(protocol, lock)
        root = base / "learned"
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
        (root / "nvidia-smi.txt").write_text("GPU\n")
        (root / "pip-freeze.txt").write_text("numpy==x\n")
        write_json(root / "task_manifest.json", {"unit": True})

        score_rows = []
        score_defs = {
            "effect-0": {
                "y": [0.0, 1.0],
                "oracle": [0.0, 1.0],
                "fresh": [0.2, 0.8],
                "stale": [0.4, 0.6],
                "shuffled": [0.8, 0.2],
            },
            "effect-1": {
                "y": [0.0, 1.0],
                "oracle": [0.0, 1.0],
                "fresh": [0.2, 0.8],
                "stale": [0.4, 0.6],
                "shuffled": [0.8, 0.2],
            },
        }
        for task_id, values in score_defs.items():
            for index, y in enumerate(values["y"]):
                score_rows.append(
                    {
                        "task_id": task_id,
                        "candidate_index": index,
                        "candidate_id": f"{task_id}:candidate-{index}",
                        "trusted_reward": y,
                        **{
                            f"{arm}_score": values[arm][index]
                            for arm in ARMS
                        },
                    }
                )
        write_jsonl(seed / "effect_verifier_scores.jsonl", score_rows)
        geometry = recompute_effect_geometry(
            score_rows, advantage_eps=1e-6, clip_advantage=5.0
        )
        geometry["fresh_vs_stale_ranking"] = {
            "eligible_pairs": 1,
            "reversals": 0,
            "reversal_rate": 0.0,
            "tie_excluded_pairs": 0,
        }
        write_json(seed / "pre_update_verifier_geometry.json", geometry)

        policy_rows = []
        for i, row in enumerate(score_rows):
            current = np.asarray([-1.0], float)
            pre = np.asarray([-1.1], float)
            raw = current - pre
            reverse = np.clip(pre - current, -20.0, 20.0)
            k3 = np.exp(reverse) - 1.0 - reverse
            policy_rows.append(
                {
                    "task_id": row["task_id"],
                    "candidate_index": row["candidate_index"],
                    "candidate_id": row["candidate_id"],
                    "current_behavior_token_logprobs": current.tolist(),
                    "pre_shift_token_logprobs": pre.tolist(),
                    "log_ratio_current_over_pre_shift": raw.tolist(),
                    "token_k3_current_vs_pre_shift": k3.tolist(),
                }
            )
        write_jsonl(seed / "pre_update_policy_shift_tokens.jsonl", policy_rows)
        policy_summary = {
            "sampled_current_vs_pre_shift_kl": 0.1,
            "sampled_current_vs_pre_shift_k3": float(
                np.exp(-0.1) - 1.0 + 0.1
            ),
            "max_abs_log_ratio": 0.1,
            "mean_abs_log_ratio": 0.1,
            "tokens": len(policy_rows),
        }
        write_json(seed / "pre_update_policy_shift.json", policy_summary)

        calibration_grid = {}
        selected = {}
        arm_records = {}
        effect_k3 = {}
        for arm in ARMS:
            cal_rows = [drift_row("calibration-0", 0, arm, [-1.0], [-0.9])]
            cal_file = f"calibration_token_drift/{arm}-lr-1e-06.jsonl"
            write_jsonl(seed / cal_file, cal_rows)
            cal_summary = drift_summary(cal_rows)
            cal_record = {
                "lr": 1e-6,
                "token_drift_file": cal_file,
                **cal_summary,
            }
            calibration_grid[arm] = [cal_record]
            selected[arm] = cal_record

            eff_rows = [drift_row("effect-0", 0, arm, [-1.0], [-0.9])]
            eff_file = f"{arm}_effect_token_drift.jsonl"
            write_jsonl(seed / eff_file, eff_rows)
            eff_summary = drift_summary(eff_rows)
            effect_k3[arm] = eff_summary["post_update_k3"]
            arm_records[arm] = {
                "lr": 1e-6,
                "train_metrics": {"grad_norm": 1.0},
                "effect_drift": eff_summary,
                "effect_token_drift_file": eff_file,
                "evaluation_candidate_logprob_file": f"{arm}_evaluation_candidate_token_logprobs.jsonl",
            }
            write_json(seed / f"{arm}_terminal_greedy_unscored.json", [])
        write_json(
            seed / "drift_calibration.json",
            {
                "grid": calibration_grid,
                "target_k3": effect_k3["oracle"],
                "selected": selected,
            },
        )

        generations = [
            {
                "prompt_id": "evaluation-0",
                "prompt": "p",
                "response": "correct",
                "logprob": -1.0,
                "token_count": 1,
                "latency_s": 0.0,
                "metadata": {
                    "response_token_ids": [1],
                    "response_token_logprobs": [-1.0],
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
        write_json(seed / "baseline_greedy_unscored.json", [])
        write_jsonl(
            seed / "evaluation_candidate_trusted_labels.jsonl",
            [
                {
                    "task_id": "evaluation-0",
                    "candidate_index": 0,
                    "candidate_id": "evaluation-0:candidate-0",
                    "trusted_reward": 1.0,
                },
                {
                    "task_id": "evaluation-0",
                    "candidate_index": 1,
                    "candidate_id": "evaluation-0:candidate-1",
                    "trusted_reward": 0.0,
                },
            ],
        )
        post_by_arm = {
            "oracle": [-0.8, -2.2],
            "fresh": [-0.9, -2.2],
            "stale": [-1.1, -1.9],
            "shuffled": [-1.2, -1.8],
        }
        shifts = {}
        pref_rows = []
        for arm, posts in post_by_arm.items():
            rows = []
            for index, (generation, post) in enumerate(zip(generations, posts)):
                baseline = generation["metadata"]["response_token_logprobs"]
                rows.append(
                    {
                        "task_id": "evaluation-0",
                        "candidate_index": index,
                        "candidate_id": f"evaluation-0:candidate-{index}",
                        "arm": arm,
                        "response": generation["response"],
                        "response_token_ids": generation["metadata"]["response_token_ids"],
                        "baseline_token_logprobs": baseline,
                        "post_update_token_logprobs": [post],
                        "baseline_sequence_logprob": float(np.sum(baseline)),
                        "post_update_sequence_logprob": float(post),
                        "baseline_mean_token_logprob": float(np.mean(baseline)),
                        "post_update_mean_token_logprob": float(post),
                    }
                )
            write_jsonl(
                seed / f"{arm}_evaluation_candidate_token_logprobs.jsonl",
                rows,
            )
            base_margin = 1.0
            post_margin = posts[0] - posts[1]
            shift = post_margin - base_margin
            shifts[arm] = shift
            pref_rows.append(
                {
                    "task_id": "evaluation-0",
                    "arm": arm,
                    "trusted_rewards": [1.0, 0.0],
                    "informative": True,
                    "baseline_margin": base_margin,
                    "post_margin": post_margin,
                    "preference_shift": shift,
                }
            )
        write_jsonl(seed / "evaluation_preference.jsonl", pref_rows)

        covariances = [geometry[arm]["cov_y_v"] for arm in ARMS]
        shift_values = [shifts[arm] for arm in ARMS]
        k3_values = [effect_k3[arm] for arm in ARMS]
        geometry_rho = spearman(covariances, shift_values)
        k3_rho = spearman(k3_values, shift_values)
        seed_summary = {
            "seed": 17,
            "stale_fit": {"eligible": True},
            "fresh_fit": {"eligible": True},
            "shift_train_metrics": {},
            "shift_drift": {},
            "pre_update_policy_shift": policy_summary,
            "pre_update_geometry": geometry,
            "selected_lrs": {arm: 1e-6 for arm in ARMS},
            "arms": arm_records,
            "effect_k3_ratio": 1.0,
            "informative_evaluation_prompts": 1,
            "mean_preference_shift": shifts,
            "geometry_spearman": geometry_rho,
            "effect_k3_spearman": k3_rho,
            "eligible_primary_seed": True,
            "greedy": {},
            "evaluation_access_before_arm_outputs_fixed": 0,
        }
        write_json(seed / "seed_summary.json", seed_summary)
        write_json(
            root / "summary.json",
            {
                "eligible_seeds": 1,
                "required_eligible_seeds": 2,
                "primary_evidence_sufficient": False,
                "mean_geometry_spearman": None,
                "mean_effect_k3_spearman": None,
                "primary_direction_passed": None,
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

    def test_completed_underpowered_artifact_recomputes_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, protocol = self.make_completed_fixture(Path(tmp))
            result = validate_evidence(
                root, protocol, expected_research_sha="research-sha"
            )
            self.assertTrue(result["valid"])
            self.assertEqual(result["execution_status"], "completed")
            self.assertEqual(result["scientific_result"], "underpowered")
            self.assertEqual(result["eligible_seeds"], 1)
            self.assertTrue(result["seed_details"][0]["validated_primary"])

    def test_rehashed_effect_score_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, protocol = self.make_completed_fixture(Path(tmp))
            path = root / "seed-17" / "effect_verifier_scores.jsonl"
            rows = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
            rows[0]["fresh_score"] = 0.95
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
                validate_evidence(
                    root, protocol, expected_research_sha="research-sha"
                )

    def test_rehashed_top_level_primary_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, protocol = self.make_completed_fixture(Path(tmp))
            path = root / "summary.json"
            summary = json.loads(path.read_text())
            summary["seed_results"][0]["geometry_spearman"] = -1.0
            write_json(path, summary)
            files = {
                str(file.relative_to(root)): sha256(file)
                for file in root.rglob("*")
                if file.is_file() and file.name != "manifest.json"
            }
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["files"] = files
            write_json(root / "manifest.json", manifest)
            with self.assertRaises(AssertionError):
                validate_evidence(
                    root, protocol, expected_research_sha="research-sha"
                )

    def test_failed_execution_is_valid_retained_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            protocol = base / "protocol.json"
            write_json(protocol, {"systems_source_sha": "systems-sha", "seeds": [17]})
            root = base / "failed"
            root.mkdir()
            write_json(
                root / "failure.json",
                {"status": "failed", "traceback": "boom"},
            )
            write_json(
                root / "manifest.json",
                {
                    "status": "failed",
                    "protocol_sha256": sha256(protocol),
                    "runner_sha256": "runner",
                    "files": {"failure.json": sha256(root / "failure.json")},
                },
            )
            result = validate_evidence(root, protocol)
            self.assertEqual(result["execution_status"], "failed")
            self.assertTrue(result["failure_retained"])


if __name__ == "__main__":
    unittest.main()
