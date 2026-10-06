import json
import tempfile
import unittest
from pathlib import Path

from src.rvl_systems.regression_gate import check_reports, evaluate_report


class RegressionGateTest(unittest.TestCase):
    def test_metric_and_relative_rules(self):
        report = {
            "metrics": {
                "success": 1.0,
                "baseline_ms": 20.0,
                "candidate_ms": 7.0,
            }
        }
        failures = evaluate_report(
            report,
            [
                {"metric": "success", "op": "ge", "value": 1.0},
                {
                    "metric": "candidate_ms",
                    "op": "le_metric",
                    "other": "baseline_ms",
                    "factor": 0.5,
                },
            ],
        )
        self.assertEqual(failures, [])

    def test_missing_required_report_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = Path(tmp) / "policy.json"
            policy.write_text(
                json.dumps({"required": []}),
                encoding="utf-8",
            )
            failures = check_reports([], policy)
        self.assertEqual(failures, ["missing required report: required"])


if __name__ == "__main__":
    unittest.main()
