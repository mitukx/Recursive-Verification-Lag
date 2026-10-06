import unittest

from src.rvl_systems.triton_grpo import (
    triton_available,
    validate_objective_backend,
)


class TritonGRPOContractTest(unittest.TestCase):
    def test_backend_validation(self):
        self.assertEqual(validate_objective_backend("torch"), "torch")
        self.assertEqual(validate_objective_backend("triton"), "triton")
        with self.assertRaises(ValueError):
            validate_objective_backend("cuda")

    def test_availability_probe_is_boolean(self):
        self.assertIsInstance(triton_available(), bool)


if __name__ == "__main__":
    unittest.main()
