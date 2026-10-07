import json
import sys
import tempfile
import unittest
from pathlib import Path

from src.run_verification_aware_gpu_phase import execute_cell
from src.validate_verification_aware_gpu_evidence import validate_cell
from src.verification_aware_phase_plan import build_plan, load_protocol, sha256_file


class VerificationAwarePhasePlanTests(unittest.TestCase):
    def setUp(self):
        self.protocol_path = Path("configs/verification_aware_async_gpu_v1.json")
        self.protocol = load_protocol(self.protocol_path)
        self.plan = build_plan(
            self.protocol,
            protocol_sha256=sha256_file(self.protocol_path),
            source_sha="deadbeef",
        )

    def test_locked_plan_expands_to_expected_unique_cells(self):
        self.assertEqual(self.plan["cell_count"], 78)
        ids = [c["cell_id"] for c in self.plan["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        phases = {c["phase"] for c in self.plan["cells"]}
        self.assertEqual(
            phases,
            {
                "baseline_three_arm",
                "verifier_latency",
                "verifier_capacity",
                "rollout_pressure",
                "freshness_bounds",
            },
        )

    def test_freshness_pairs_are_not_cartesian(self):
        rows = [
            c for c in self.plan["cells"]
            if c["phase"] == "freshness_bounds" and c["seed"] == 17
        ]
        pairs = {
            (
                c["effective_system"]["max_policy_lag"],
                c["effective_system"]["max_verifier_lag"],
            )
            for c in rows
        }
        self.assertEqual(pairs, {(2, 0), (8, 1), (16, 2)})

    def _write_driver(self, root: Path, *, fail: bool = False) -> Path:
        script = root / "driver.py"
        if fail:
            script.write_text("raise RuntimeError('injected driver failure')\n")
            return script

        required_ts = self.plan["required_timeseries"]
        terminal = self.plan["required_terminal_metrics"]
        invariants = self.plan["hard_invariants"]
        script.write_text(
            "import json, os\n"
            "from pathlib import Path\n"
            "root=Path(os.environ['RVL_EVIDENCE_DIR'])\n"
            f"ts={json.dumps(required_ts)!r}\n"
            f"terminal={json.dumps(terminal)!r}\n"
            f"invariants={json.dumps(invariants)!r}\n"
            "ts=json.loads(ts); terminal=json.loads(terminal); invariants=json.loads(invariants)\n"
            "rows=[]\n"
            "for i in range(2):\n"
            "  row={'t_s':float(i)}\n"
            "  row.update({k:0.0 for k in ts})\n"
            "  rows.append(row)\n"
            "(root/'timeseries.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\\n' for r in rows))\n"
            "summary={'terminal_metrics':{k:0 for k in terminal},"
            "'hard_invariants':[{'statement':s,'passed':True} for s in invariants]}\n"
            "(root/'summary.json').write_text(json.dumps(summary,sort_keys=True))\n"
        )
        return script

    def test_successful_cell_is_hash_manifested_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = self._write_driver(root)
            cell = self.plan["cells"][0]
            record = execute_cell(
                self.plan,
                cell,
                root / "evidence",
                [sys.executable, str(script)],
                require_gpu=False,
            )
            self.assertEqual(record["status"], "completed")
            report = validate_cell(self.plan, cell, root / "evidence")
            self.assertTrue(report["valid"])
            self.assertEqual(report["status"], "completed")

    def test_failed_cell_is_retained_as_valid_negative_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = self._write_driver(root, fail=True)
            cell = self.plan["cells"][0]
            record = execute_cell(
                self.plan,
                cell,
                root / "evidence",
                [sys.executable, str(script)],
                require_gpu=False,
            )
            self.assertEqual(record["status"], "failed")
            cell_root = root / "evidence" / cell["cell_id"]
            self.assertTrue((cell_root / "failure.json").exists())
            self.assertTrue((cell_root / "driver.stderr.log").exists())
            report = validate_cell(self.plan, cell, root / "evidence")
            self.assertTrue(report["valid"])
            self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
