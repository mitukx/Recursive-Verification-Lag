import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analyze_transfer_geometry import (
    analyze,
    canonical_score_geometry,
    policy_geometry,
)


class TransferGeometryAuditTest(unittest.TestCase):
    def test_positive_affine_invariance_and_policy_metrics(self):
        base = [0.2, 0.3, 0.5]
        same = canonical_score_geometry(base, [0.0, 1.0, 2.0], [7.0, 10.0, 13.0])
        self.assertAlmostEqual(same["canonical_weighted_rms"], 0.0, places=14)
        self.assertAlmostEqual(same["canonical_max_abs"], 0.0, places=14)
        changed = canonical_score_geometry(base, [0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
        self.assertGreater(changed["canonical_weighted_rms"], 0.0)
        p = policy_geometry([0.2, 0.3, 0.5], [0.1, 0.4, 0.5])
        self.assertAlmostEqual(p["policy_total_variation"], 0.1)
        self.assertGreater(p["policy_jensen_shannon"], 0.0)

    def test_complete_archived_geometry_audit(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "geometry"
            analyze(
                root / "results/transfer_calibration_v1/policies.json",
                root / "results/transfer_calibration_v1/ranking_changes.csv",
                root / "data/mbppplus_qwen15b_development_v2_scored.jsonl",
                root / "results/transfer_calibration_v1/manifest.json",
                out,
            )
            summary = pd.read_csv(out / "summary.csv")
            self.assertTrue((summary.tasks == 16).all())
            self.assertTrue((summary.tasks_with_strict_reversal == 3).all())
            self.assertTrue((summary.tasks_with_policy_change == 14).all())
            self.assertTrue((summary.tasks_without_reversal_but_policy_change == 12).all())

            corr = pd.read_csv(out / "correlations.csv")
            gp = corr[
                (corr.x == "canonical_weighted_rms")
                & (corr.y == "policy_total_variation")
            ].iloc[0]
            rp = corr[
                (corr.x == "strict_reversals")
                & (corr.y == "policy_total_variation")
            ].iloc[0]
            self.assertGreater(gp.spearman, 0.7)
            self.assertLess(rp.spearman, 0.4)

            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["tasks_with_strict_reversal"], 3)
            self.assertEqual(manifest["strict_reversals_total_unique_pairs"], 20)
            self.assertLess(manifest["public_control_max_policy_total_variation"], 1e-10)
            self.assertEqual(manifest["new_policy_decisions"], 0)
            self.assertEqual(manifest["new_physical_candidate_executions"], 0)


if __name__ == "__main__":
    unittest.main()
