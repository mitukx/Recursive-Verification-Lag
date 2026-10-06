import math
import unittest

from src.rvl_systems.quantization import (
    dequantize_int8,
    evaluate_int8_linear_parity,
    linear,
    quantize_per_row_symmetric_int8,
)


class QuantizationNumericsTests(unittest.TestCase):
    def test_per_row_symmetric_int8_bounds_and_zero_row(self):
        weights = [[-2.0,-1.0,0.0,1.0,2.0],[0.0]*5]
        q = quantize_per_row_symmetric_int8(weights)
        self.assertEqual(max(abs(v) for v in q.qvalues[0]),127)
        self.assertEqual(q.qvalues[1],(0,0,0,0,0))
        self.assertEqual(q.scales[1],1.0)
        restored = dequantize_int8(q)
        self.assertEqual(restored[1],[0.0]*5)

    def test_linear_parity_error_is_bounded_on_representative_grid(self):
        weights = [
            [math.sin(i+j/3) * (1+j/8) for j in range(32)]
            for i in range(12)
        ]
        inputs = [
            [math.cos(seed + j/7) for j in range(32)]
            for seed in range(9)
        ]
        d = evaluate_int8_linear_parity(weights,inputs)
        self.assertLess(d.mean_relative_output_error,0.05)
        self.assertLess(d.max_abs_output_error,0.2)
        self.assertGreater(d.theoretical_storage_ratio,3.0)

    def test_rejects_malformed_and_nonfinite_inputs(self):
        with self.assertRaises(ValueError):
            quantize_per_row_symmetric_int8([[1.0],[1.0,2.0]])
        with self.assertRaises(ValueError):
            quantize_per_row_symmetric_int8([[float("nan")]])
        with self.assertRaises(ValueError):
            linear([[1.0,2.0]],[1.0])


if __name__ == "__main__":
    unittest.main()
