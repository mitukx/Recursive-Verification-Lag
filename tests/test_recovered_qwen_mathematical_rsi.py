import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_recovered_qwen_mathematical_rsi import run
from scripts.validate_recovered_qwen_mathematical_rsi import validate


class RecoveredQwenMathematicalRSITest(unittest.TestCase):
    bank = Path("data/recovered_qwen05b_bank.jsonl")
    locked_protocol = Path("configs/recovered_qwen_mathematical_rsi_v1.json")

    def small_protocol(self, base: Path) -> Path:
        raw = json.loads(self.locked_protocol.read_text())
        raw["audit_replicates"] = 4
        path = base / "protocol.json"
        path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
        return path

    def test_real_bank_replay_and_independent_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            protocol = self.small_protocol(base)
            out = base / "result"
            result = run(protocol, self.bank, out)
            report = validate(out, protocol, self.bank)
            self.assertTrue(report["valid"])
            self.assertEqual(result["counts"]["tasks"], 12)
            self.assertEqual(result["counts"]["proposal_cells"], 36)
            self.assertEqual(report["information_budget_required"], 28)

    def test_fixed_probe_covers_all_tasks_and_underbudget_info_refreshes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            protocol = self.small_protocol(base)
            out = base / "result"
            run(protocol, self.bank, out)
            rows = [
                json.loads(line)
                for line in (out / "trials.jsonl").read_text().splitlines()
                if line.strip()
            ]
            fixed = [
                row for row in rows
                if row["method"] in {"coded_fixed_probe", "coded_fixed_probe_info"}
            ]
            self.assertTrue(fixed)
            self.assertTrue(all(row["task_coverage"] == 1.0 for row in fixed))
            low = [
                row for row in rows
                if row["method"] == "coded_fixed_probe_info" and row["budget"] == 24
            ]
            high = [
                row for row in rows
                if row["method"] == "coded_fixed_probe_info" and row["budget"] == 36
            ]
            self.assertTrue(low and high)
            self.assertTrue(all(not row["information_budget_passed"] for row in low))
            self.assertTrue(all(row["action"] == "refresh" for row in low))
            self.assertTrue(all(row["information_budget_passed"] for row in high))

    def test_nonproxy_allow_has_exact_finite_bank_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            protocol = self.small_protocol(base)
            out = base / "result"
            run(protocol, self.bank, out)
            rows = [
                json.loads(line)
                for line in (out / "trials.jsonl").read_text().splitlines()
                if line.strip()
            ]
            allowed = [
                row for row in rows
                if row["method"] != "proxy_only" and row["action"] == "allow"
            ]
            self.assertTrue(all(row["lower"] >= 0 for row in allowed))
            self.assertTrue(all(not row["harmful"] for row in allowed))


if __name__ == "__main__":
    unittest.main()
