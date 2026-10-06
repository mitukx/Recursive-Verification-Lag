import unittest

import numpy as np

from src.finite_kl_alignment_identity import (
    direct_quantities,
    error_alignment_decomposition,
    exponential_tilt,
    verify_path_identities,
)


class FiniteKLAlignmentIdentityTest(unittest.TestCase):
    def test_path_identities_match_direct_quantities(self):
        rng = np.random.default_rng(20261007)
        for _ in range(12):
            p = rng.dirichlet(np.full(9, 1.5))
            y = rng.normal(size=9)
            v = rng.normal(size=9)
            for beta in (0.0, 0.1, 0.5, 1.5, 3.0):
                result = verify_path_identities(p, y, v, beta, atol=2e-10)
                self.assertTrue(result["passed"], result)
                self.assertGreaterEqual(result["proxy_progress"], -1e-12)
                self.assertGreaterEqual(result["kl"], -1e-12)

    def test_same_kl_can_have_opposite_true_progress(self):
        p = np.full(4, 0.25)
        y = np.array([0.0, 0.0, 1.0, 1.0])
        harmful_v = np.array([2.0, 1.0, -1.0, -2.0])
        benign_v = -harmful_v

        def solve_beta(v, target):
            lo, hi = 0.0, 1.0
            while direct_quantities(p, y, v, hi)["kl"] < target:
                hi *= 2
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                if direct_quantities(p, y, v, mid)["kl"] < target:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)

        target = 0.12
        bh = solve_beta(harmful_v, target)
        bb = solve_beta(benign_v, target)
        harmful = direct_quantities(p, y, harmful_v, bh)
        benign = direct_quantities(p, y, benign_v, bb)

        self.assertAlmostEqual(harmful["kl"], target, places=10)
        self.assertAlmostEqual(benign["kl"], target, places=10)
        self.assertGreater(harmful["proxy_progress"], 0)
        self.assertGreater(benign["proxy_progress"], 0)
        self.assertLess(harmful["true_progress"], 0)
        self.assertGreater(benign["true_progress"], 0)

    def test_error_alignment_decomposition_is_exact(self):
        p = np.array([0.1, 0.2, 0.3, 0.4])
        y = np.array([0.0, 1.0, 0.0, 1.0])
        verifier = np.array([1.0, 0.5, -0.2, 2.0])
        q = exponential_tilt(p, verifier, 0.7)
        d = error_alignment_decomposition(q, y, verifier)
        self.assertLess(d["decomposition_error"], 1e-12)
        self.assertAlmostEqual(
            d["total_true_progress_derivative"],
            d["signal_variance"] + d["error_alignment_covariance"],
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
