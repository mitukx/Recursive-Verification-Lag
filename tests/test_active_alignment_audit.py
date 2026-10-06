import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.active_alignment_audit import (
    active_proposal,
    covariance_audit,
    estimator_range,
    run,
)


class ActiveAlignmentAuditTest(unittest.TestCase):
    def test_active_proposal_is_normalized_and_equalizes_coefficients(self):
        p = np.array([0.1, 0.2, 0.3, 0.4])
        v = np.array([-3.0, -1.0, 2.0, 4.0])
        result = active_proposal(p, v)
        r = result["proposal"]
        self.assertAlmostEqual(float(r.sum()), 1.0)
        self.assertTrue((r >= 0).all())
        centered = result["centered_verifier"]
        support = np.abs(centered) > 1e-15
        coeff = np.abs(p[support] * centered[support] / r[support])
        np.testing.assert_allclose(coeff, coeff[0], atol=1e-12, rtol=0)

    def test_active_range_no_worse_than_passive_on_nonconstant_score(self):
        rng = np.random.default_rng(7)
        for _ in range(100):
            p = rng.dirichlet(np.ones(12))
            v = rng.normal(size=12)
            a = active_proposal(p, v)
            active = estimator_range(p, a["centered_verifier"], a["proposal"])
            passive = estimator_range(p, a["centered_verifier"], p)
            self.assertLessEqual(active["max_abs"], passive["max_abs"] + 1e-12)

    def test_covariance_estimator_targets_exact_quantity(self):
        p = np.array([0.2, 0.3, 0.5])
        y = np.array([0.0, 1.0, 1.0])
        v = np.array([-1.0, 0.5, 2.0])
        exact = float(p @ (y * (v - p @ v)))
        estimates = []
        for seed in range(200):
            row = covariance_audit(
                p, y, v, n=2000, delta=0.05, seed=seed, mode="active"
            )
            estimates.append(row["estimate"])
            self.assertAlmostEqual(row["exact_covariance"], exact)
        self.assertLess(abs(float(np.mean(estimates)) - exact), 0.01)

    def test_constant_verifier_is_inconclusive(self):
        p = np.array([0.4, 0.6])
        y = np.array([0.0, 1.0])
        v = np.array([3.0, 3.0])
        row = covariance_audit(p, y, v, n=32, delta=0.05, seed=1, mode="active")
        self.assertEqual(row["decision"], "inconclusive")
        self.assertEqual(row["radius"], 0.0)
        self.assertEqual(row["exact_covariance"], 0.0)

    def test_locked_protocol_executes_and_hashes_outputs(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "audit"
            manifest = run(root / "configs/active_alignment_audit_v1.json", out)
            self.assertEqual(manifest["evaluation_task_count"], 1024)
            self.assertEqual(manifest["primary_budget"], 256)
            self.assertTrue((out / "rows.csv.gz").exists())
            self.assertTrue((out / "summary.csv").exists())
            self.assertIn("primary_prediction_passed", manifest)


if __name__ == "__main__":
    unittest.main()
