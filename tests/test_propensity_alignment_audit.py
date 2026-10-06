import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.propensity_alignment_audit import (
    allocate_propensities,
    propensity_covariance_audit,
    run,
)


class PropensityAlignmentAuditTest(unittest.TestCase):
    def test_propensity_allocations_respect_budget_and_bounds(self):
        p = np.array([0.05, 0.15, 0.30, 0.50])
        v = np.array([-2.0, -0.5, 0.5, 2.0])
        uniform = allocate_propensities(
            p, v, expected_budget=2.0, method="uniform", minimum_propensity=0.05
        )
        leverage = allocate_propensities(
            p, v, expected_budget=2.0, method="proxy_leverage", minimum_propensity=0.05
        )
        self.assertAlmostEqual(uniform.sum(), 2.0, places=10)
        self.assertAlmostEqual(leverage.sum(), 2.0, places=8)
        self.assertTrue((leverage >= 0.05 - 1e-10).all())
        self.assertTrue((leverage <= 1.0 + 1e-10).all())
        self.assertFalse(np.allclose(uniform, leverage))

        mu = float(p @ v)
        uniform_bound = np.sum((p * np.abs(v - mu) / uniform) ** 2)
        leverage_bound = np.sum((p * np.abs(v - mu) / leverage) ** 2)
        self.assertLess(leverage_bound, uniform_bound)

    def test_strong_alignment_decisions_are_fail_closed(self):
        n = 64
        p = np.full(n, 1.0 / n)
        y = np.array([0.0] * (n // 2) + [1.0] * (n // 2))
        helpful = np.array([-1.0] * (n // 2) + [1.0] * (n // 2))
        harmful = -helpful
        pi = np.full(n, 0.75)
        good = propensity_covariance_audit(
            p, y, helpful, pi, delta=0.05, seed=1
        )
        bad = propensity_covariance_audit(
            p, y, harmful, pi, delta=0.05, seed=2
        )
        self.assertGreater(good["exact_covariance"], 0.0)
        self.assertLess(bad["exact_covariance"], 0.0)
        self.assertEqual(good["wrong_sign_decision"], 0)
        self.assertEqual(bad["wrong_sign_decision"], 0)
        self.assertIn(good["decision"], ("allow", "inconclusive"))
        self.assertIn(bad["decision"], ("block", "inconclusive"))

    def test_small_locked_run_preserves_full_propensity_evidence(self):
        root = Path(__file__).resolve().parents[1]
        lock = json.loads(
            (root / "configs/propensity_alignment_audit_v1.json").read_text()
        )
        lock["task_seeds"] = {"start": 40000, "count": 48}
        lock["expected_label_budgets"] = [8, 16]
        with tempfile.TemporaryDirectory() as temporary:
            protocol = Path(temporary) / "protocol.json"
            protocol.write_text(json.dumps(lock))
            out = Path(temporary) / "out"
            manifest = run(protocol, out)
            self.assertTrue(manifest["primary_radius_better_both_arms"])
            self.assertFalse(
                manifest["trusted_reward_used_for_propensity_allocation"]
            )
            summary = pd.read_csv(out / "summary.csv")
            primary = summary[summary.expected_labels == 16]
            radii = primary.pivot(
                index="arm", columns="method", values="mean_radius"
            )
            self.assertTrue(
                (radii["proxy_leverage"] < radii["uniform"]).all()
            )
            self.assertTrue((out / "selection_records.jsonl.gz").exists())
            self.assertGreater((out / "selection_records.jsonl.gz").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
