import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_posttraining_campaign import build_scorecard


class PostTrainingCampaignScorecardTests(unittest.TestCase):
    def test_combines_real_capability_and_reward_model_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rlvr = root / "benchmark.json"
            learned = root / "summary.json"
            rlvr.write_text(json.dumps({
                "metrics": {
                    "before_accuracy": 0.40,
                    "after_accuracy": 0.50,
                    "accuracy_delta": 0.10,
                    "eval_examples": 100,
                    "steps": 8
                },
                "config": {"model": "tiny"}
            }))
            learned.write_text(json.dumps({
                "required_eligible_seeds": 2,
                "primary_evidence_sufficient": True,
                "primary_direction_passed": True,
                "mean_geometry_spearman": 0.8,
                "mean_effect_k3_spearman": 0.2,
                "seed_results": [
                    {
                        "eligible_primary_seed": True,
                        "mean_preference_shift": {
                            "oracle": 0.30, "fresh": 0.24, "stale": 0.05, "shuffled": -0.02
                        },
                        "pre_update_geometry": {
                            arm: {"cov_y_v": value}
                            for arm, value in {
                                "oracle": 0.4, "fresh": 0.3, "stale": 0.1, "shuffled": 0.0
                            }.items()
                        }
                    },
                    {
                        "eligible_primary_seed": True,
                        "mean_preference_shift": {
                            "oracle": 0.20, "fresh": 0.16, "stale": 0.04, "shuffled": 0.00
                        },
                        "pre_update_geometry": {
                            arm: {"cov_y_v": value}
                            for arm, value in {
                                "oracle": 0.35, "fresh": 0.25, "stale": 0.08, "shuffled": 0.01
                            }.items()
                        }
                    }
                ]
            }))
            result = build_scorecard(
                rlvr_benchmark=rlvr,
                learned_verifier_summary=learned,
            )
            self.assertTrue(result["claims"]["real_model_capability_gain_observed"])
            self.assertTrue(result["claims"]["learned_verifier_evidence_sufficient"])
            self.assertTrue(result["claims"]["fresh_beats_stale_on_trusted_shift"])
            self.assertTrue(result["claims"]["fresh_beats_shuffled_on_trusted_shift"])
            self.assertAlmostEqual(
                result["learned_reward_model"]["comparisons"]["fresh_minus_stale_trusted_shift"],
                0.155,
            )

    def test_missing_evidence_stays_explicit(self):
        result = build_scorecard()
        self.assertEqual(result["oracle_rlvr"]["status"], "missing")
        self.assertEqual(result["learned_reward_model"]["status"], "missing")
        self.assertFalse(result["claims"]["real_model_capability_gain_observed"])


if __name__ == "__main__":
    unittest.main()
