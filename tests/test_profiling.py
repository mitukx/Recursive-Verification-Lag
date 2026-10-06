import json
import tempfile
import time
import unittest
from pathlib import Path

from src.rvl_systems.benchmark_report import BenchmarkReport
from src.rvl_systems.profiling import TraceRecorder


class ProfilingTest(unittest.TestCase):
    def test_trace_summary_and_export(self):
        recorder = TraceRecorder()
        with recorder.span("rollout", worker="w0"):
            time.sleep(0.001)
        with recorder.span("rollout", worker="w1"):
            time.sleep(0.001)
        summary = recorder.summary()
        self.assertEqual(summary["rollout"]["count"], 2.0)
        self.assertGreater(summary["rollout"]["mean_ms"], 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            recorder.export_chrome_trace(path)
            data = json.loads(path.read_text())
            self.assertEqual(len(data["traceEvents"]), 2)
            self.assertEqual(data["traceEvents"][0]["ph"], "X")

    def test_benchmark_report_records_environment(self):
        report = BenchmarkReport(
            name="scheduler",
            metrics={"samples_per_s": 10.0},
            config={"workers": 2},
            git_sha="abc",
        )
        payload = report.payload()
        self.assertEqual(payload["git_sha"], "abc")
        self.assertIn("python", payload["environment"])


if __name__ == "__main__":
    unittest.main()
