import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_real_llm_active_audit import (
    ARMS,
    load_real_candidate_cells,
    run,
    sha256,
)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


class RealLLMActiveAuditContractTest(unittest.TestCase):
    def make_input(self, base: Path):
        root = base / "learned"
        seed = root / "seed-17"
        seed.mkdir(parents=True)
        y = np.asarray([0.0, 0.0, 1.0, 1.0])
        scores = {
            "oracle": [0.0, 0.0, 1.0, 1.0],
            "fresh": [0.1, 0.2, 0.8, 0.9],
            "stale": [0.3, 0.6, 0.4, 0.7],
            "shuffled": [0.9, 0.8, 0.2, 0.1],
        }
        rows = []
        for i in range(4):
            rows.append(
                {
                    "task_id": "effect-0",
                    "candidate_index": i,
                    "candidate_id": f"effect-0:candidate-{i}",
                    "trusted_reward": float(y[i]),
                    **{f"{arm}_score": scores[arm][i] for arm in ARMS},
                }
            )
        write_jsonl(seed / "effect_verifier_scores.jsonl", rows)
        geometry = {}
        p = np.full(4, 0.25)
        for arm in ARMS:
            v = np.asarray(scores[arm], float)
            geometry[arm] = {
                "cov_y_v": float(p @ (y * (v - float(p @ v))))
            }
        write_json(
            seed / "seed_summary.json",
            {
                "seed": 17,
                "arms": {arm: {} for arm in ARMS},
                "mean_preference_shift": {
                    "oracle": 0.4,
                    "fresh": 0.3,
                    "stale": -0.2,
                    "shuffled": -0.5,
                },
                "pre_update_geometry": geometry,
            },
        )
        write_json(root / "environment.json", {"research_source_sha": "source-sha"})
        files = {
            str(path.relative_to(root)): sha256(path)
            for path in root.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        write_json(root / "manifest.json", {"status": "completed", "files": files})
        return root

    def make_protocol(self, base: Path):
        protocol = base / "protocol.json"
        write_json(
            protocol,
            {
                "status": "prospective real-LLM active-audit replay locked before learned-verifier GPU outcomes",
                "trusted_label_budgets": [2],
                "audit_replicates": 4,
                "confidence_delta": 0.05,
                "propensity_minimum": 0.1,
                "seed_offset": 1000,
                "primary_budget": 2,
                "limitations": ["unit fixture"],
            },
        )
        return protocol

    def test_real_candidate_loader_preserves_terminal_and_covariance_ground_truths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_input(Path(tmp))
            cells = load_real_candidate_cells(root)
            self.assertEqual(len(cells), 4)
            by_arm = {cell["arm"]: cell for cell in cells}
            self.assertTrue(by_arm["stale"]["harmful_update"])
            self.assertTrue(by_arm["fresh"]["beneficial_update"])
            self.assertLess(by_arm["shuffled"]["exact_covariance"], 0)
            self.assertGreater(by_arm["fresh"]["exact_covariance"], 0)

    def test_end_to_end_replay_runs_existing_three_estimators(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = self.make_input(base)
            protocol = self.make_protocol(base)
            output = base / "out"
            result = run(protocol, root, output)
            self.assertEqual(len(result["input_cells"]), 4)
            self.assertEqual(len(result["primary_rows"]), 3)
            methods = {row["method"] for row in result["primary_rows"]}
            self.assertEqual(
                methods,
                {"passive_fixed", "propensity_ht", "active_minimax"},
            )
            self.assertTrue((output / "audit_trials.jsonl.gz").exists())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["rows"], 4 * 3 * 4)
            self.assertEqual(manifest["input_research_source_sha"], "source-sha")

    def test_tampered_input_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = self.make_input(base)
            protocol = self.make_protocol(base)
            (root / "environment.json").write_text('{"research_source_sha":"tampered"}\n')
            with self.assertRaises(ValueError):
                run(protocol, root, base / "out")


if __name__ == "__main__":
    unittest.main()
