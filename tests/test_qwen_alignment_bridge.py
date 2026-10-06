import math
import unittest

import numpy as np

from scripts.run_qwen_alignment_bridge import (
    alignment_proxy,
    candidate_preference_shift,
    effect_drift_ratio,
    select_matched_drift_lrs,
)


class QwenAlignmentBridgeContractTest(unittest.TestCase):
    def test_alignment_proxy_fixes_error_norm_and_flips_covariance(self):
        rewards = [0, 0, 0, 1, 1, 1, 0, 1]
        harmful = alignment_proxy(rewards, sigma=2.0, rho=-0.75, seed=17)
        benign = alignment_proxy(rewards, sigma=2.0, rho=0.75, seed=17)
        self.assertTrue(harmful["informative"])
        self.assertTrue(benign["informative"])
        self.assertAlmostEqual(harmful["error_norm"], 2.0, places=12)
        self.assertAlmostEqual(benign["error_norm"], 2.0, places=12)
        self.assertLess(harmful["trusted_proxy_covariance"], 0.0)
        self.assertGreater(benign["trusted_proxy_covariance"], 0.0)
        np.testing.assert_allclose(
            harmful["nuisance"], benign["nuisance"], rtol=0, atol=0
        )
        self.assertAlmostEqual(
            harmful["trusted_proxy_covariance"], 1 + 2 * -0.75, places=12
        )
        self.assertAlmostEqual(
            benign["trusted_proxy_covariance"], 1 + 2 * 0.75, places=12
        )

    def test_constant_group_is_retained_but_uninformative(self):
        row = alignment_proxy([0, 0, 0, 0], sigma=2.0, rho=-0.75, seed=3)
        self.assertFalse(row["informative"])
        self.assertEqual(row["proxy"], [0.0] * 4)
        self.assertEqual(row["error_norm"], 0.0)

    def test_calibration_selects_common_target_without_evaluation_input(self):
        grid = {
            "harmful": [
                {"lr": 1e-7, "post_update_k3": 1e-6},
                {"lr": 2e-7, "post_update_k3": 4e-6},
                {"lr": 5e-7, "post_update_k3": 2e-5},
            ],
            "benign": [
                {"lr": 1e-7, "post_update_k3": 2e-6},
                {"lr": 2e-7, "post_update_k3": 7e-6},
                {"lr": 5e-7, "post_update_k3": 3e-5},
            ],
        }
        result = select_matched_drift_lrs(grid, target_fraction=0.8)
        self.assertAlmostEqual(result["target_k3"], 1.6e-5)
        self.assertEqual(result["selected"]["harmful"]["lr"], 5e-7)
        self.assertEqual(result["selected"]["benign"]["lr"], 2e-7)
        self.assertNotIn("evaluation", result)

    def test_preference_shift_uses_both_correct_and_incorrect_candidates(self):
        base = [-3.0, -2.5, -2.0, -3.5]
        post = [-3.4, -2.0, -1.5, -3.8]
        trusted = [0, 1, 1, 0]
        metric = candidate_preference_shift(base, post, trusted)
        self.assertTrue(metric["informative"])
        self.assertGreater(metric["preference_shift"], 0)
        missing = candidate_preference_shift(base, post, [0, 0, 0, 0])
        self.assertFalse(missing["informative"])
        self.assertIsNone(missing["preference_shift"])

    def test_effect_drift_ratio_is_symmetric_and_fail_closed(self):
        self.assertAlmostEqual(effect_drift_ratio(2e-5, 3e-5), 1.5)
        self.assertAlmostEqual(effect_drift_ratio(3e-5, 2e-5), 1.5)
        self.assertEqual(effect_drift_ratio(0.0, 0.0), 1.0)
        self.assertTrue(math.isinf(effect_drift_ratio(0.0, 1e-5)))
        with self.assertRaises(ValueError):
            effect_drift_ratio(-1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
