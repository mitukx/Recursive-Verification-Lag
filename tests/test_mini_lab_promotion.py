import json
import tempfile
import unittest
from pathlib import Path

from src.rvl_systems.lab.promotion import (
    EvalSample,
    PromotionLedger,
    PromotionPolicy,
    evaluate_promotion,
)


def suite(candidate, incumbent=None, n=48):
    incumbent = [0.0] * n if incumbent is None else incumbent
    return [
        EvalSample(f"eval-{i}", i % 3, incumbent[i], candidate[i])
        for i in range(n)
    ]


class PromotionGateTests(unittest.TestCase):
    def test_accepts_newer_non_regressing_candidate(self):
        samples = suite([1.0] * 48)
        d = evaluate_promotion(
            samples,
            incumbent_version=3,
            candidate_version=4,
            policy=PromotionPolicy(),
        )
        self.assertTrue(d.accepted)
        self.assertEqual(d.wins, 48)
        self.assertEqual(d.losses, 0)
        self.assertEqual(len(d.evidence_sha256), 64)

    def test_rejects_mean_and_slice_regression(self):
        incumbent = [1.0] * 48
        candidate = [1.0] * 48
        for i in range(0, 48, 3):
            candidate[i] = 0.0
        d = evaluate_promotion(
            suite(candidate, incumbent),
            incumbent_version=8,
            candidate_version=9,
            policy=PromotionPolicy(max_family_regression=0.05),
        )
        self.assertFalse(d.accepted)
        self.assertIn("mean_reward_regression", d.reasons)
        self.assertIn("family_slice_regression", d.reasons)

    def test_rejects_non_monotonic_version(self):
        d = evaluate_promotion(
            suite([0.0] * 48),
            incumbent_version=5,
            candidate_version=5,
            policy=PromotionPolicy(),
        )
        self.assertFalse(d.accepted)
        self.assertIn("candidate_version_not_newer", d.reasons)

    def test_duplicate_eval_identity_fails_closed(self):
        samples = suite([0.0] * 48)
        samples[-1] = EvalSample(samples[0].task_id, 2, 0.0, 0.0)
        with self.assertRaises(ValueError):
            evaluate_promotion(
                samples,
                incumbent_version=1,
                candidate_version=2,
                policy=PromotionPolicy(),
            )

    def test_hash_chained_ledger_detects_tampering(self):
        decision = evaluate_promotion(
            suite([1.0] * 48),
            incumbent_version=0,
            candidate_version=1,
            policy=PromotionPolicy(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "promotion-ledger.jsonl"
            ledger = PromotionLedger(path)
            first = ledger.append(decision)
            second = ledger.append(decision)
            self.assertNotEqual(first, second)
            self.assertEqual(PromotionLedger(path).head, second)

            rows = path.read_text().splitlines()
            record = json.loads(rows[0])
            record["decision"]["candidate_mean"] = 0.0
            rows[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
            path.write_text("\n".join(rows) + "\n")
            with self.assertRaises(ValueError):
                PromotionLedger(path)


if __name__ == "__main__":
    unittest.main()
