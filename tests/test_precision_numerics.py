import math
import unittest

from src.rvl_systems.numerics import safe_importance_ratio, summarize_ratios
from src.rvl_systems.precision import resolve_precision_name


class PrecisionNumericsTest(unittest.TestCase):
    def test_auto_precision_policy(self):
        self.assertEqual(resolve_precision_name("cpu", "auto"), "fp32")
        self.assertEqual(resolve_precision_name("mps", "auto"), "fp32")
        self.assertEqual(resolve_precision_name("cuda", "auto", bf16_supported=True), "bf16")
        self.assertEqual(resolve_precision_name("cuda", "auto", bf16_supported=False), "fp16")

    def test_rejects_unsupported_bf16(self):
        with self.assertRaises(ValueError):
            resolve_precision_name("cuda", "bf16", bf16_supported=False)

    def test_safe_ratio_clamps_and_rejects_nonfinite(self):
        self.assertAlmostEqual(safe_importance_ratio(0.0, 0.0), 1.0)
        self.assertTrue(math.isfinite(safe_importance_ratio(100.0, 0.0)))
        with self.assertRaises(FloatingPointError):
            safe_importance_ratio(float("nan"), 0.0)

    def test_ratio_diagnostics(self):
        d = summarize_ratios([(0.0, 0.0), (math.log(2), 0.0), (float("nan"), 0.0)])
        self.assertEqual(d.count, 2)
        self.assertEqual(d.nonfinite_count, 1)
        self.assertGreater(d.clip_fraction, 0.0)


if __name__ == "__main__":
    unittest.main()
