import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DistributedScalingSummaryTest(unittest.TestCase):
    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            single = root / "single.json"
            multi = root / "multi.json"
            output = root / "summary.json"
            single.write_text(
                json.dumps({"world_size": 1, "tokens_per_s": 100.0}),
                encoding="utf-8",
            )
            multi.write_text(
                json.dumps({"world_size": 2, "tokens_per_s": 180.0}),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.summarize_distributed_scaling",
                    "--single",
                    str(single),
                    "--multi",
                    str(multi),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(output.read_text(encoding="utf-8"))
        self.assertAlmostEqual(summary["speedup"], 1.8)
        self.assertAlmostEqual(summary["scaling_efficiency"], 0.9)


if __name__ == "__main__":
    unittest.main()
