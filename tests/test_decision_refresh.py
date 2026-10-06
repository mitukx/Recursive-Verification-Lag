import unittest
import numpy as np
import pandas as pd
from src.candidate_bank_experiment import Config
from src.decision_refresh import decision_refresh_run
from src.source_audit import SourceAudit


class DecisionRefreshTest(unittest.TestCase):
    def frame(self, rewards):
        return pd.DataFrame({'task_id': ['t']*4, 'candidate_id': list('abcd'),
            'base_logprob': [0.]*4, 'trusted_score': rewards,
            'f::public_score': [0., .25, .75, 1.]})

    def test_all_strategies_baseline_safety_and_budget(self):
        for strategy in ('impact', 'decision', 'stream'):
            for seed in range(4):
                history, trace = decision_refresh_run(self.frame([1., 1., 0., 0.]),
                    list('abcd'), Config(rounds=4, eta=4., seed=seed), budget=3,
                    strategy=strategy, return_trace=True)
                self.assertTrue((history.gain_evaluation_only >= -1e-10).all())
                self.assertTrue((history.paid_source_labels <= 3).all())
                for row, point in zip(history.itertuples(), trace):
                    self.assertEqual(len(point['paid_sources']), row.paid_source_labels)
                    if row.accepted:
                        self.assertGreaterEqual(row.candidate_lower, .01)
                        np.testing.assert_array_equal(point['policy'], point['proposal'])

    def test_unqueried_rewards_do_not_change_policy_or_queries(self):
        ids = np.array(list('abcd'))
        for strategy in ('impact', 'decision', 'stream'):
            for seed in range(4):
                cfg = Config(rounds=5, eta=3., seed=seed)
                sampler = SourceAudit(ids, seed)
                known = {sampler.groups[i] for i in sampler.order[:2]}
                y = np.array([1., 0., 1., 0.])
                altered = np.array([v if g in known else 1-v for g, v in zip(ids, y)])
                h, t = decision_refresh_run(self.frame(y), ids, cfg,
                    budget=2, strategy=strategy, return_trace=True)
                other, ot = decision_refresh_run(self.frame(altered), ids, cfg,
                    budget=2, strategy=strategy, return_trace=True)
                for col in ('accepted', 'status', 'paid_source_labels',
                            'candidate_lower', 'candidate_upper'):
                    np.testing.assert_array_equal(h[col], other[col])
                for a, b in zip(t, ot):
                    self.assertEqual(a['queries'], b['queries'])
                    np.testing.assert_array_equal(a['proposal'], b['proposal'])
                    np.testing.assert_array_equal(a['policy'], b['policy'])

    def test_constant_zero_rewards_cannot_create_gain(self):
        for strategy in ('impact', 'decision', 'stream'):
            h = decision_refresh_run(self.frame([0.]*4), list('abcd'),
                Config(rounds=3), budget=4, strategy=strategy)
            self.assertTrue((h.gain_evaluation_only == 0.).all())
            self.assertTrue((h.accepted == 0).all())

    def test_multitask_input_and_fractional_dp_fail(self):
        df = self.frame([0., 1., 0., 1.])
        df.loc[0, 'task_id'] = 'other'
        with self.assertRaises(ValueError):
            decision_refresh_run(df, list('abcd'), Config(), budget=4)
        with self.assertRaises(ValueError):
            decision_refresh_run(self.frame([0., .5, 0., 1.]),
                list('abcd'), Config(), budget=4, strategy='decision')


if __name__ == '__main__':
    unittest.main()
