import itertools
import unittest
import numpy as np
import pandas as pd
from src.candidate_bank_experiment import Config
from src.endogenous_audit import EndogenousAuditPlanner, endogenous_run


class EndogenousAuditTest(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'task_id': ['t']*4, 'candidate_id': list('abcd'),
                             'base_logprob': [0.]*4,
                             'f::public_score': [0., .25, .75, 1.]})

    def planner(self, **kwargs):
        return EndogenousAuditPlanner(self.frame(), list('abcd'),
                                      Config(eta=4., representation='public'), **kwargs)

    def test_zero_label_trap_and_future_information(self):
        known = {'a': 0., 'b': 0.}
        myopic = endogenous_run(self.planner(), known,
                               lambda g: self.fail('myopic must not query'),
                               budget=2, updates=2, strategy='myopic')
        np.testing.assert_allclose(myopic['policy'], [.25]*4)
        planner = self.planner()
        value, cost, action = planner.plan(known, planner.baseline, 2, 2)
        self.assertGreater(value, 0.)
        self.assertGreater(cost, 0.)
        self.assertEqual(action[0], 'audit')
        out = endogenous_run(planner, known, lambda g: float(g == 'd'), budget=2, updates=2)
        self.assertGreater((out['policy']-planner.baseline) @ [0., 0., 0., 1.], 0.)

    def test_all_binary_worlds_safety_cost_and_prior_value(self):
        prior = (.2, .4, .7, .8)
        planner = self.planner(prior=prior)
        known = {'a': 0.}
        expected_value, expected_cost, _ = planner.plan(known, planner.baseline, 2, 2)
        realized_value, realized_cost = 0., 0.
        for tail in itertools.product((0., 1.), repeat=3):
            rewards = dict(zip('abcd', (0.,)+tail))
            weight = np.prod([pi if v else 1-pi for pi, v in zip(prior[1:], tail)])
            for strategy in ('lookahead', 'myopic', 'stream', 'escape', 'one_update', 'direct'):
                paid = []
                def oracle(g):
                    self.assertNotIn(g, known)
                    self.assertNotIn(g, paid)
                    paid.append(g)
                    return rewards[g]
                out = endogenous_run(planner, known, oracle, budget=2, updates=2,
                                     strategy=strategy, order=list('abcd'))
                gain = float((out['policy']-planner.baseline) @ list(rewards.values()))
                self.assertGreaterEqual(gain, -1e-12)
                self.assertLessEqual(len(paid), 2)
                self.assertLessEqual(sum(t['action'] == 'update' for t in out['trace']), 2)
                self.assertEqual(len(paid), len(out['queries']))
                if strategy == 'direct' and any(t['action'] == 'update' for t in out['trace']):
                    self.assertAlmostEqual(float(out['policy'] @ list(rewards.values())), 1.)
                if strategy == 'lookahead':
                    realized_value += weight*gain
                    realized_cost += weight*len(paid)
        self.assertAlmostEqual(realized_value, expected_value, places=11)
        self.assertAlmostEqual(realized_cost, expected_cost, places=11)

    def test_optimum_against_independent_brute_force_action_trees(self):
        planner = self.planner(prior=(.2, .3, .7, .8))
        # No memoization and explicit legal action enumeration; tests the
        # controller recurrence rather than trusting its own returned value.
        def brute(known, p, b, h):
            expected = np.array([known.get(g, pi) for g, pi in zip(planner.groups, planner.prior)])
            options = [(float((np.array(p)-planner.baseline) @ expected), 0.)]
            if h:
                state = planner.state(known)
                q, lo, _, _ = planner._proposal(state, p)
                if lo >= planner.gain_floor and q != p:
                    options.append(brute(known, q, b, h-1))
                if b:
                    for g, pi in zip(planner.groups, planner.prior):
                        if g not in known:
                            a = brute({**known, g: 0}, p, b-1, h)
                            c = brute({**known, g: 1}, p, b-1, h)
                            options.append(((1-pi)*a[0]+pi*c[0], 1+(1-pi)*a[1]+pi*c[1]))
            return min(options, key=lambda x: (-x[0], x[1]))
        known = {'a': 0., 'b': 1.}
        expected = brute(known, tuple(planner.baseline), 2, 2)
        actual = planner.plan(known, planner.baseline, 2, 2)
        np.testing.assert_allclose(actual[:2], expected, atol=1e-11)

    def test_unknown_outcomes_do_not_enter_controller(self):
        frame = self.frame()
        frame['trusted_score'] = [0., 0., 1., 1.]
        with self.assertRaisesRegex(ValueError, 'unlabeled'):
            EndogenousAuditPlanner(frame, list('abcd'), Config())
        planner = self.planner()
        a = planner.plan({'a': 0., 'b': 0.}, planner.baseline, 1, 1)
        b = self.planner().plan({'a': 0., 'b': 0.}, planner.baseline, 1, 1)
        self.assertEqual(a, b)

    def test_escape_is_one_query_when_zero_labels_persist(self):
        paid = []
        def oracle(g):
            paid.append(g)
            return 0.
        out = endogenous_run(self.planner(), {'a': 0., 'b': 0.}, oracle,
                             budget=2, updates=2, strategy='escape', order=list('abcd'))
        self.assertEqual(paid, ['c'])
        np.testing.assert_allclose(out['policy'], [.25]*4)

    def test_stop_without_updates_and_explicit_state_cap(self):
        planner = self.planner()
        self.assertEqual(planner.plan({'a': 0.}, planner.baseline, 3, 0), (0., 0., ('stop', None)))
        with self.assertRaisesRegex(RuntimeError, 'state cap'):
            self.planner(max_states=1).plan({'a': 0.}, planner.baseline, 2, 2)

    def test_binary_and_resource_validation(self):
        planner = self.planner()
        for labels in ({'a': .2}, {'z': 0}):
            with self.assertRaises(ValueError):
                planner.plan(labels, planner.baseline, 2, 2)
        for budget in (-1, .5, True):
            with self.assertRaises(ValueError):
                endogenous_run(planner, {'a': 0}, lambda g: 0., budget=budget, updates=2)
        with self.assertRaises(ValueError):
            self.planner(prior=(.5, .5, .5, 1.))


if __name__ == '__main__':
    unittest.main()
