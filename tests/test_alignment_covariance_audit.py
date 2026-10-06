import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.alignment_covariance_audit import covariance_audit, run
from src.equal_kl_alignment_stress import make_task


class AlignmentCovarianceAuditTest(unittest.TestCase):
    def test_harmful_and_benign_signs_are_distinguishable(self):
        p, y, y_std, u = make_task(123, 32)
        harmful = -y_std
        benign = 3.0 * y_std
        h = covariance_audit(p, y, harmful, n=512, delta=0.05, seed=1)
        b = covariance_audit(p, y, benign, n=512, delta=0.05, seed=2)
        self.assertLess(h["exact_covariance"], 0)
        self.assertGreater(b["exact_covariance"], 0)
        self.assertEqual(h["decision"], "block")
        self.assertEqual(b["decision"], "allow")
        self.assertEqual(h["wrong_sign_decision"], 0)
        self.assertEqual(b["wrong_sign_decision"], 0)

    def test_complete_small_locked_run(self):
        root = Path(__file__).resolve().parents[1]
        lock = json.loads(
            (root / "configs/alignment_covariance_audit_v1.json").read_text()
        )
        lock["task_seeds"] = {"start": 20000, "count": 96}
        lock["trusted_label_budgets"] = [32, 128]
        with tempfile.TemporaryDirectory() as temporary:
            protocol = Path(temporary) / "protocol.json"
            protocol.write_text(json.dumps(lock))
            out = Path(temporary) / "out"
            manifest = run(protocol, out)
            self.assertTrue(manifest["primary_prediction_passed"])
            summary = pd.read_csv(out / "summary.csv")
            primary = summary[summary.labels == 128].set_index("arm")
            self.assertGreaterEqual(primary.loc["harmful", "block_rate"], 0.8)
            self.assertGreaterEqual(primary.loc["benign", "allow_rate"], 0.8)
            self.assertLessEqual(
                primary.loc["harmful", "wrong_sign_decision_rate"], 0.08
            )
            self.assertLessEqual(
                primary.loc["benign", "wrong_sign_decision_rate"], 0.08
            )


if __name__ == "__main__":
    unittest.main()
