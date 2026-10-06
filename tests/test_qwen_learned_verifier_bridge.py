import unittest

import numpy as np

from scripts.run_qwen_learned_verifier_bridge import (
    ARMS,
    deterministic_prompt_shuffle,
    pairwise_reversal_rate,
    select_matched_drift_lrs_multi,
    spearman,
    verifier_geometry_metrics,
)


class LearnedVerifierBridgeContractTest(unittest.TestCase):
    def test_geometry_covariance_error_identity(self):
        y = np.asarray([0, 0, 1, 1], float)
        v = np.asarray([0.1, 0.4, 0.6, 0.9], float)
        metrics = verifier_geometry_metrics(y, v)
        variance = float(np.mean((y - y.mean()) ** 2))
        self.assertAlmostEqual(
            metrics["cov_y_e"],
            metrics["cov_y_v"] - variance,
            places=12,
        )
        self.assertGreater(metrics["cov_y_v"], 0)
        self.assertGreater(metrics["ranking_accuracy"], 0.5)
        self.assertGreaterEqual(metrics["ece_10"], 0.0)

    def test_misaligned_scores_flip_geometry(self):
        y = [0, 0, 1, 1]
        good = verifier_geometry_metrics(y, [0.1, 0.2, 0.8, 0.9])
        bad = verifier_geometry_metrics(y, [0.9, 0.8, 0.2, 0.1])
        self.assertGreater(good["cov_y_v"], 0)
        self.assertLess(bad["cov_y_v"], 0)
        self.assertEqual(good["ranking_accuracy"], 1.0)
        self.assertEqual(bad["ranking_accuracy"], 0.0)

    def test_pairwise_reversal_rate(self):
        result = pairwise_reversal_rate([0, 1, 2], [0, 2, 1])
        self.assertEqual(result["eligible_pairs"], 3)
        self.assertEqual(result["reversals"], 1)
        self.assertAlmostEqual(result["reversal_rate"], 1 / 3)

    def test_prompt_shuffle_is_deterministic_and_preserves_multisets(self):
        scores = {"b": [0.1, 0.2, 0.3, 0.4], "a": [0.9, 0.8, 0.7, 0.6]}
        first = deterministic_prompt_shuffle(scores, seed=17)
        second = deterministic_prompt_shuffle(scores, seed=17)
        self.assertEqual(first, second)
        for task_id in scores:
            self.assertEqual(sorted(first[task_id]), sorted(scores[task_id]))

    def test_multiarm_matched_drift_selection_uses_common_target(self):
        grid = {
            arm: [
                {"lr": 1e-7, "post_update_k3": 1e-6 * (index + 1)},
                {"lr": 2e-7, "post_update_k3": 4e-6 * (index + 1)},
                {"lr": 5e-7, "post_update_k3": 2e-5 * (index + 1)},
            ]
            for index, arm in enumerate(ARMS)
        }
        result = select_matched_drift_lrs_multi(grid, target_fraction=0.8)
        self.assertAlmostEqual(result["target_k3"], 1.6e-5)
        self.assertEqual(set(result["selected"]), set(ARMS))
        for arm in ARMS:
            self.assertIn(result["selected"][arm]["lr"], {1e-7, 2e-7, 5e-7})

    def test_spearman_detects_order_and_reverse(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)


if __name__ == "__main__":
    unittest.main()
