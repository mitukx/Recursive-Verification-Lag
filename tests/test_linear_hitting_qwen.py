import tempfile
import unittest
from pathlib import Path

from scripts.run_linear_hitting_qwen import run
from scripts.validate_linear_hitting_qwen import validate


class LinearHittingQwenIntegrationTest(unittest.TestCase):
    def test_real_qwen_bank_followup_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "result"
            protocol = Path("configs/linear_hitting_qwen_v1.json")
            bank = Path("data/recovered_qwen05b_bank.jsonl")
            result = run(protocol, bank, root)
            report = validate(root, protocol, bank)
            self.assertTrue(report["valid"])
            self.assertEqual(result["proposal_cells"], 36)
            for meta in result["representations"].values():
                self.assertEqual(meta["probe_count"], meta["feature_rank"])
                self.assertGreater(meta["probe_count"], 0)


if __name__ == "__main__":
    unittest.main()
