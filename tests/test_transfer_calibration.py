import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.transfer_calibration import matched_kl_policy, score_family, pairwise_disagreements, build_policies, evaluate
from src.candidate_bank_experiment import best_of_n


class TransferCalibrationTest(unittest.TestCase):
    def test_affine_invariance_and_kl_constraint_on_nonuniform_support(self):
        rng = np.random.default_rng(73)
        for _ in range(30):
            p = rng.dirichlet(np.ones(9)); p[0] = 0.; p /= p.sum()
            score = rng.normal(size=9)
            target = .25*score_family(p, score)[2]
            a = matched_kl_policy(p, score, target)
            b = matched_kl_policy(p, 17*score+31, target)
            q = np.array(a['policy']); ok = q > 0
            self.assertEqual(q[0], 0.)
            self.assertAlmostEqual(q.sum(), 1., places=13)
            measured = float(q[ok] @ np.log(q[ok]/p[ok]))
            self.assertAlmostEqual(measured, target, places=10)
            np.testing.assert_allclose(q, b['policy'], rtol=0, atol=2e-11)
        # Tiny positive scale must not be mistaken for a constant vector.
        p = np.array([.2, .3, .5]); s = np.array([0., 1., 2.])
        np.testing.assert_allclose(matched_kl_policy(p, s, .1)['policy'],
                                  matched_kl_policy(p, 1e-250*s, .1)['policy'])

    def test_tied_top_limit_constants_and_invalid_inputs(self):
        p = np.array([.1, .2, .3, .4]); s = np.array([0., 1., 1., 0.])
        self.assertAlmostEqual(score_family(p, s)[2], np.log(2))
        q = np.array(matched_kl_policy(p, s, .2)['policy'])
        self.assertAlmostEqual(q[1]/q[2], 2/3)
        np.testing.assert_array_equal(matched_kl_policy(p, [7.]*4, 0)['policy'], p)
        for bad_p, bad_s, target in [(p, s, -.1), (p, s, np.log(2)),
                                    (p, [7.]*4, .1), ([.5, .6], [0, 1], .1),
                                    (p, [0, 1, np.nan, 0], .1), (p, s, np.inf)]:
            with self.assertRaises(ValueError):
                matched_kl_policy(bad_p, bad_s, target)

    def test_order_ties_bon_invariance_and_negative_slope_counterexample(self):
        p = np.array([.2, .3, .1, .4]); s = np.array([0., 1., 1., 2.])
        self.assertEqual(pairwise_disagreements(s, 3*s+9)['order_or_tie_changes'], 0)
        np.testing.assert_allclose(best_of_n(p, s, 4), best_of_n(p, 3*s+9, 4))
        self.assertGreater(pairwise_disagreements(s, -s)['strict_reversals'], 0)
        self.assertFalse(np.allclose(matched_kl_policy(p, s, .1)['policy'],
                                    matched_kl_policy(p, -s, .1)['policy']))
        changed = s.copy(); changed[2] += 1e-12
        self.assertEqual(pairwise_disagreements(s, changed)['tie_changes'], 1)
        self.assertEqual(pairwise_disagreements(s, changed, 1e-10)['tie_changes'], 0)

    def test_builder_blocks_outcomes_and_preserves_zero_radius_tasks(self):
        df = pd.DataFrame({'task_id': ['a', 'a', 'b', 'b'],
            'candidate_id': ['a0', 'a1', 'b0', 'b1'], 'base_logprob': [0.]*4,
            'f::public_score': [0., 1., .5, .5]})
        model = {'feature_columns': ['f::public_score'], 'theta': [0., 1.]}
        models = {arm: model for arm in ['all_frozen', 'all_refreshed', 'public_frozen',
                                         'public_refreshed', 'shuffled_refreshed']}
        a, ranks = build_policies(df, models, [.1, .3])
        self.assertEqual(len(a), 24)
        self.assertTrue(all(d['target_kl'] == 0 for d in a if d['task_id'] == 'b'))
        self.assertTrue(all(r['order_or_tie_changes'] == 0 for r in ranks))
        df['trusted_score'] = [1., 0., 1., 1.]
        with self.assertRaisesRegex(ValueError, 'unlabeled'):
            build_policies(df, models, [.1])

    def test_worst_case_unknown_reward_is_negative_total_variation(self):
        rng = np.random.default_rng(81)
        for _ in range(25):
            p, q = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))
            d = q-p
            completions = np.array([[int(bool(k & (1 << j))) for j in range(5)] for k in range(32)])
            exact = float((completions@d).min())
            self.assertAlmostEqual(exact, -abs(d).sum()/2)
            self.assertLess(exact, 0.)

    def test_complete_archive_diagnostic_and_hash_abort(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)/'out'
            evaluate(root/'data/mbppplus_qwen15b_development_v2_scored.jsonl',
                     root/'configs/transfer_calibration_diagnostic_v1.json',
                     root/'results/fresh_task_verifier_transfer_v1', out)
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['decision_time_evaluation_task_trusted_queries'], 0)
            self.assertEqual(len(manifest['evaluation_tasks']), 16)
            for name, h in manifest['output_sha256'].items():
                self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(), h)
            rows = pd.read_csv(out/'evaluation.csv')
            self.assertEqual(len(rows), 384)
            np.testing.assert_allclose(rows.distribution_free_gain_lower_bound, -rows.total_variation)
            contrasts = pd.read_csv(out/'contrasts.csv')
            self.assertTrue((contrasts.tasks == 16).all())
            # This tests the established mathematical invariant, not a positive
            # research effect or the hand-entered original endpoint.
            self.assertLess(abs(contrasts[contrasts.arm == 'public_refreshed'].matched_kl_gain_difference).max(), 1e-10)
            bad = json.loads((root/'configs/transfer_calibration_diagnostic_v1.json').read_text())
            bad['bank_sha256'] = 'wrong'
            path = Path(temporary)/'bad.json'; path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'provenance'):
                evaluate(root/'data/mbppplus_qwen15b_development_v2_scored.jsonl', path,
                         root/'results/fresh_task_verifier_transfer_v1', Path(temporary)/'bad-out')


if __name__ == '__main__':
    unittest.main()
