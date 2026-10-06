import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_real_llm_active_audit import ARMS, sha256
from scripts.run_real_llm_mathematical_rsi import METHODS, run
from scripts.validate_real_llm_mathematical_rsi_evidence import validate_evidence


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


class RealLLMMathematicalRSITest(unittest.TestCase):
    def make_input(self, base: Path):
        root = base / "learned"
        seed = root / "seed-17"
        seed.mkdir(parents=True)
        y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], float)
        scores = {
            "oracle": [0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95],
            "fresh": [0.10, 0.20, 0.25, 0.30, 0.70, 0.75, 0.80, 0.90],
            "stale": [0.20, 0.65, 0.30, 0.60, 0.40, 0.70, 0.45, 0.75],
            "shuffled": [0.90, 0.80, 0.70, 0.60, 0.40, 0.30, 0.20, 0.10],
        }
        rows = []
        for i in range(len(y)):
            rows.append({
                "task_id": f"effect-{i // 4}",
                "candidate_index": i % 4,
                "candidate_id": f"effect-{i // 4}:candidate-{i % 4}",
                "trusted_reward": float(y[i]),
                **{f"{arm}_score": scores[arm][i] for arm in ARMS},
            })
        write_jsonl(seed / "effect_verifier_scores.jsonl", rows)
        p = np.full(len(y), 1.0 / len(y))
        geometry = {}
        for arm in ARMS:
            v = np.asarray(scores[arm], float)
            geometry[arm] = {"cov_y_v": float(p @ (y * (v - float(p @ v))))}
        write_json(seed / "seed_summary.json", {
            "seed": 17,
            "arms": {arm: {} for arm in ARMS},
            "mean_preference_shift": {
                "oracle": 0.4,
                "fresh": 0.25,
                "stale": -0.15,
                "shuffled": -0.4,
            },
            "pre_update_geometry": geometry,
        })
        write_json(root / "environment.json", {"research_source_sha": "source-sha"})
        files = {
            str(path.relative_to(root)): sha256(path)
            for path in root.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        write_json(root / "manifest.json", {
            "status": "completed",
            "protocol_sha256": "input-protocol",
            "files": files,
        })
        return root

    def make_protocol(self, base: Path):
        path = base / "protocol.json"
        write_json(path, {
            "status": "locked retrospective real-LLM mathematical-RSI replay; no downstream outcome tuning",
            "input_requirements": {
                "manifest_status": "completed",
                "expected_input_protocol_sha256": "input-protocol",
                "expected_input_research_source_sha": "source-sha",
                "expected_seed_values": [17],
                "expected_candidate_count_per_seed": 8,
            },
            "trusted_label_budgets": [2, 4],
            "audit_replicates": 4,
            "confidence_delta": 0.05,
            "seed_offset": 9000,
            "coded_verification": {
                "probe_strata": 2,
                "max_corrupt_fraction": 0.5,
            },
            "information_budget": {
                "target_error": 0.1,
                "multiplier": 0.5,
            },
            "reconstruction_diagnostic": {
                "branching_factors": [2, 4],
            },
            "primary_budget": 4,
            "interpretation_rule": "unit fixture",
            "limitations": ["unit fixture"],
        })
        return path

    def test_end_to_end_real_model_controller_replay_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            input_root = self.make_input(base)
            protocol = self.make_protocol(base)
            output = base / "output"
            summary = run(protocol, input_root, output)
            self.assertEqual({row["method"] for row in summary["primary_rows"]}, set(METHODS))
            report = validate_evidence(output, protocol, input_root)
            self.assertTrue(report["valid"])
            self.assertEqual(report["cells"], 4)
            self.assertEqual(report["trials"], 4 * 2 * len(METHODS) * 4)
            self.assertIn(
                report["scientific_result"],
                {"direction_passed", "direction_failed", "underpowered"},
            )

    def test_fixed_probe_is_complete_and_information_budget_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            input_root = self.make_input(base)
            protocol = self.make_protocol(base)
            output = base / "output"
            run(protocol, input_root, output)
            rows = [
                json.loads(line)
                for line in gzip.decompress((output / "trials.jsonl.gz").read_bytes())
                .decode().splitlines()
                if line.strip()
            ]
            fixed = [r for r in rows if r["method"] == "coded_fixed_probe"]
            self.assertTrue(fixed)
            self.assertTrue(all(r["probe_coverage"] == 1.0 for r in fixed))
            info_low = [
                r for r in rows
                if r["method"] == "coded_fixed_probe_info" and r["budget"] == 2
            ]
            info_high = [
                r for r in rows
                if r["method"] == "coded_fixed_probe_info" and r["budget"] == 4
            ]
            self.assertTrue(all(not r["information_budget_passed"] for r in info_low))
            self.assertTrue(all(r["decision"] == "inconclusive" for r in info_low))
            self.assertTrue(all(r["information_budget_passed"] for r in info_high))

    def test_reconstruction_is_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            input_root = self.make_input(base)
            protocol = self.make_protocol(base)
            output = base / "output"
            summary = run(protocol, input_root, output)
            self.assertTrue(summary["reconstruction_diagnostics"])
            self.assertTrue(
                all(not row["assumptions_met"] for row in summary["reconstruction_diagnostics"])
            )

    def test_rehashed_trial_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            input_root = self.make_input(base)
            protocol = self.make_protocol(base)
            output = base / "output"
            run(protocol, input_root, output)
            trials_path = output / "trials.jsonl.gz"
            rows = [
                json.loads(line)
                for line in gzip.decompress(trials_path.read_bytes()).decode().splitlines()
                if line.strip()
            ]
            rows[0]["harmful_detected"] = 1 - int(rows[0]["harmful_detected"])
            text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            trials_path.write_bytes(gzip.compress(text.encode(), compresslevel=9, mtime=0))
            manifest = json.loads((output / "manifest.json").read_text())
            manifest["files"]["trials.jsonl.gz"] = sha256(trials_path)
            write_json(output / "manifest.json", manifest)
            with self.assertRaises(AssertionError):
                validate_evidence(output, protocol, input_root)


if __name__ == "__main__":
    unittest.main()
