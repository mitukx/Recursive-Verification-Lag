import unittest

import numpy as np

from src.rsi_controller.hitting_verification import (
    LinearErrorHittingVerifier,
    evaluator_residual_diagnostic,
    select_hitting_rows,
)


class LinearHittingVerifierTests(unittest.TestCase):
    def test_exact_linear_error_is_recovered_from_rank_many_fixed_probes(self):
        x = np.asarray([
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 0., 1.],
            [1., 1., 1.],
            [1., 2., -1.],
        ])
        ids = [f"row-{i}" for i in range(len(x))]
        probes = select_hitting_rows(x, ids)
        self.assertEqual(len(probes), np.linalg.matrix_rank(x))
        beta = np.asarray([0.1, -0.04, 0.03])
        truth = np.asarray([0.3, 0.5, 0.7, 0.2, 0.8])
        proxy = truth + x @ beta
        verifier = LinearErrorHittingVerifier(x, proxy, probes)
        probe_truth = truth[list(probes)]
        for delta in (
            np.asarray([-.2, .1, .1, 0., 0.]),
            np.asarray([0., -.1, 0., .3, -.2]),
        ):
            interval = verifier.interval(delta, probe_truth, residual_radius=0.0)
            self.assertAlmostEqual(interval.radius, 0.0)
            self.assertAlmostEqual(interval.estimate, float(delta @ truth), places=10)

    def test_bounded_residual_interval_contains_true_gain(self):
        rng = np.random.default_rng(17)
        x = np.column_stack([np.ones(20), rng.normal(size=(20, 3))])
        ids = [f"row-{i:02d}" for i in range(len(x))]
        probes = select_hitting_rows(x, ids)
        beta = rng.normal(scale=.05, size=x.shape[1])
        rho = .03
        residual = rng.uniform(-rho, rho, size=len(x))
        truth = rng.uniform(.2, .8, size=len(x))
        proxy = truth + x @ beta + residual
        verifier = LinearErrorHittingVerifier(x, proxy, probes)
        for _ in range(20):
            delta = rng.normal(size=len(x))
            delta -= delta.mean()
            interval = verifier.interval(
                delta, truth[list(probes)], residual_radius=rho
            )
            actual = float(delta @ truth)
            self.assertLessEqual(interval.lower - 1e-10, actual)
            self.assertGreaterEqual(interval.upper + 1e-10, actual)

    def test_probe_selection_is_label_blind_and_deterministic(self):
        x = np.asarray([
            [1., 0.],
            [1., 1.],
            [1., 2.],
            [1., -1.],
        ])
        ids = ["b", "a", "d", "c"]
        self.assertEqual(select_hitting_rows(x, ids), select_hitting_rows(x, ids))

    def test_misspecification_diagnostic_is_evaluator_only_statistic(self):
        x = np.column_stack([np.ones(4), np.arange(4.)])
        truth = np.asarray([0., 1., 0., 1.])
        proxy = np.asarray([.1, .9, .2, .8])
        diagnostic = evaluator_residual_diagnostic(x, proxy, truth)
        self.assertGreaterEqual(diagnostic["lstsq_residual_inf"], 0.0)
        self.assertGreaterEqual(diagnostic["lstsq_residual_rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
