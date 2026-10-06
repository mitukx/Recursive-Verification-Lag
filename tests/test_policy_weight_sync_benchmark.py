import math
import unittest

from src.benchmark_policy_weight_sync import summarize_latencies


class WeightSyncBenchmarkTests(unittest.TestCase):
    def test_summary_is_finite_and_uses_median(self):
        report = summarize_latencies([0.2,0.1,0.3],1024**3)
        self.assertEqual(report["iterations"],3)
        self.assertAlmostEqual(report["latency_ms_p50"],200.0)
        self.assertAlmostEqual(report["effective_gib_per_s_p50"],5.0)
        self.assertTrue(math.isfinite(report["latency_ms_p95"]))

    def test_invalid_samples_fail_closed(self):
        with self.assertRaises(ValueError):
            summarize_latencies([],1024)
        with self.assertRaises(ValueError):
            summarize_latencies([0.1],0)
        with self.assertRaises(ValueError):
            summarize_latencies([float("nan")],1024)


if __name__ == "__main__":
    unittest.main()
