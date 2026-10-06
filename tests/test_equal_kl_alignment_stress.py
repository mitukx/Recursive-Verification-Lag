import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.equal_kl_alignment_stress import (
    make_task,
    matched_kl_tilt,
    run,
    weighted_inner,
)


class EqualKLAlignmentStressTest(unittest.TestCase):
    def test_constructed_error_basis_is_weighted_orthogonal(self):
        for seed in range(8):
            p, y, y_std, u = make_task(seed, 24)
            self.assertAlmostEqual(float(p @ y_std), 0.0, places=12)
            self.assertAlmostEqual(weighted_inner(p, y_std, y_std), 1.0, places=12)
            self.assertAlmostEqual(float(p @ u), 0.0, places=12)
            self.assertAlmostEqual(weighted_inner(p, u, u), 1.0, places=12)
            self.assertAlmostEqual(weighted_inner(p, u, y_std), 0.0, places=12)
            self.assertGreater(y.max(), y.min())

    def test_matched_kl_and_constant_score_failure(self):
        p = np.array([0.1, 0.2, 0.3, 0.4])
        score = np.array([-1.0, 0.0, 1.0, 2.0])
        for target in (0.005, 0.05, 0.2):
            out = matched_kl_tilt(p, score, target)
            self.assertTrue(out["attainable"])
            self.assertAlmostEqual(out["policy"].sum(), 1.0, places=12)
            self.assertAlmostEqual(out["kl"], target, places=10)
        self.assertFalse(matched_kl_tilt(p, np.zeros(4), 0.01)["attainable"])

    def test_small_prospective_phase_recovers_local_geometry(self):
        root = Path(__file__).resolve().parents[1]
        lock = json.loads((root / "configs/equal_kl_alignment_stress_v1.json").read_text())
        lock["task_seeds"] = {"start": 0, "count": 48}
        lock["error_scales"] = [0.5, 1.5, 2.0]
        lock["alignment_rhos"] = [-1.0, -0.75, -0.25, 0.0, 0.5, 1.0]
        lock["matched_kl_budgets"] = [0.005, 0.1]
        with tempfile.TemporaryDirectory() as temporary:
            protocol = Path(temporary) / "lock.json"
            protocol.write_text(json.dumps(lock))
            output = Path(temporary) / "out"
            manifest = run(protocol, output)
            self.assertTrue(manifest["primary_prediction_passed"])
            self.assertGreaterEqual(manifest["primary_phase_accuracy"], 0.9)
            rows = pd.read_csv(output / "rows.csv.gz")
            available = rows[rows.attainable == 1]
            self.assertTrue((available.proxy_progress >= -1e-10).all())
            cells = pd.read_csv(output / "cells.csv")
            primary = cells[(cells.requested_kl == 0.005) & (cells.predicted_sign != 0)]
            harmful = primary[primary.predicted_sign < 0]
            benign = primary[primary.predicted_sign > 0]
            self.assertGreater(harmful.false_progress_rate.mean(), 0.9)
            self.assertLess(benign.false_progress_rate.mean(), 0.1)


if __name__ == "__main__":
    unittest.main()
