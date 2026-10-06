import unittest

from src.rvl_systems.precision_parity import (
    evaluate_logprob_parity,
    quantize_bf16,
    quantize_fp16,
)


class PrecisionParityTest(unittest.TestCase):
    def test_quantizers_are_finite(self):
        for value in (-3.25, -0.1, 0.0, 0.1, 3.25):
            self.assertTrue(abs(quantize_fp16(value)) < 10)
            self.assertTrue(abs(quantize_bf16(value)) < 10)

    def test_representative_ratio_parity(self):
        values = [
            (-4.0, -4.1),
            (-2.0, -2.2),
            (-1.0, -0.9),
            (-0.1, -0.2),
            (0.0, -0.1),
            (1.5, 1.2),
            (3.0, 2.5),
        ]
        fp16 = evaluate_logprob_parity(values, precision="fp16")
        bf16 = evaluate_logprob_parity(values, precision="bf16")
        self.assertLess(fp16.max_relative_ratio_error, 0.01)
        self.assertLess(bf16.max_relative_ratio_error, 0.05)


if __name__ == "__main__":
    unittest.main()
