import unittest

from src.rvl_systems.telemetry import Telemetry


class TelemetryTest(unittest.TestCase):
    def test_snapshot_reports_percentiles(self):
        t = Telemetry()
        for value in [1.0, 2.0, 3.0, 4.0, 5.0]:
            t.observe("latency", value)
        snap = t.snapshot()
        self.assertEqual(snap["latency.p50"], 3.0)
        self.assertGreater(snap["latency.p95"], 4.0)
        self.assertEqual(snap["latency.max"], 5.0)


if __name__ == "__main__":
    unittest.main()
