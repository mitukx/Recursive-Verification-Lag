import itertools
import unittest
import numpy as np
from src.decision_audit import (BinaryDecisionPlanner, audit_comparison,
    contrast_bounds, decision, minimum_witness, source_contrast)


class DecisionAuditTest(unittest.TestCase):
    def test_witness_matches_every_subset_and_fractional_rewards(self):
        groups = ('a', 'b', 'c', 'd')
        weights = (-.375, -.125, .25, .25)
        for values in itertools.product((0., .5, 1.), repeat=4):
            rewards = dict(zip(groups, values))
            for initially_known in ((), ('a',), ('c', 'd')):
                revealed = {g: rewards[g] for g in initially_known}
                for threshold in (0., .125):
                    actual, witness, status = minimum_witness(
                        groups, weights, rewards, revealed, threshold=threshold)
                    remaining = [g for g in groups if g not in revealed]
                    expected = None
                    for k in range(len(remaining)+1):
                        if any(decision(*contrast_bounds(groups, weights,
                            {**revealed, **{g: rewards[g] for g in subset}}), threshold) == status
                            for subset in itertools.combinations(remaining, k)):
                            expected = k
                            break
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual, len(witness))

    def test_dp_matches_exhaustive_query_trees_under_prior(self):
        groups = ('a', 'b', 'c')
        weights = (-.5, .25, .25)
        prior = (.2, .6, .8)
        # Independent reference enumerates trees without the planner's cache.
        def enumerate_trees(revealed, budget):
            if decision(*contrast_bounds(groups, weights, revealed)) != 'unresolved':
                return (1., 0.)
            if budget == 0:
                return (0., 0.)
            results = []
            for i, g in enumerate(groups):
                if g in revealed:
                    continue
                zero = enumerate_trees({**revealed, g: 0}, budget-1)
                one = enumerate_trees({**revealed, g: 1}, budget-1)
                results.append(((1-prior[i])*zero[0]+prior[i]*one[0],
                    1+(1-prior[i])*zero[1]+prior[i]*one[1]))
            return min(results+[(0., 0.)], key=lambda x: (-x[0], x[1]))
        planner = BinaryDecisionPlanner(groups, weights, prior=prior)
        for budget in range(4):
            success, cost, _ = planner.plan({}, budget)
            expected = enumerate_trees({}, budget)
            self.assertAlmostEqual(success, expected[0])
            self.assertAlmostEqual(cost, expected[1])

    def test_decisions_are_sound_in_all_binary_worlds_and_cost_is_exact(self):
        p = np.array([.125, .125, .25, .5])
        b = np.ones(4)/4
        groups = ['a', 'b', 'c', 'd']
        for values in itertools.product((0, 1), repeat=4):
            rewards = dict(zip(groups, values))
            for strategy in ('decision', 'impact', 'stream'):
                for budget in range(5):
                    calls = []
                    def oracle(g):
                        self.assertNotIn(g, calls)
                        calls.append(g)
                        return rewards[g]
                    out = audit_comparison(p, b, groups, {}, oracle,
                        budget=budget, strategy=strategy, order=groups)
                    self.assertEqual(tuple(calls), out['queries'])
                    self.assertEqual(len(calls), out['paid_additional_labels'])
                    self.assertLessEqual(len(calls), budget)
                    gain = float((p-b) @ np.array(values))
                    if out['status'] == 'accept':
                        self.assertGreaterEqual(gain, 0.)
                    elif out['status'] == 'reject':
                        self.assertLess(gain, 0.)
                    if budget == 4:
                        self.assertNotEqual(out['status'], 'unresolved')

    def test_hidden_world_changes_cannot_change_prequery_decisions(self):
        p = [.125, .125, .75]
        b = [.25, .25, .5]
        first_queries = []
        for values in itertools.product((0, 1), repeat=3):
            calls = []
            rewards = dict(zip(('a', 'b', 'c'), values))
            def oracle(g):
                calls.append(g)
                return rewards[g]
            out = audit_comparison(p, b, ['a', 'b', 'c'], {}, oracle,
                budget=1, strategy='decision')
            first_queries.append(calls[0])
            # Flipping every outcome outside the paid transcript cannot change
            # the returned bounds or action, even when actual gain changes.
            flipped = {g: (rewards[g] if g in calls else 1-rewards[g]) for g in rewards}
            other = audit_comparison(p, b, ['a', 'b', 'c'], {},
                flipped.__getitem__, budget=1, strategy='decision')
            for key in ('status', 'lower', 'upper', 'queries'):
                self.assertEqual(out[key], other[key])
        self.assertEqual(len(set(first_queries)), 1)

    def test_duplicate_source_contrast_and_zero_movement(self):
        g, w = source_contrast([.125, .375, .5], [.25, .25, .5], ['a', 'a', 'b'])
        np.testing.assert_array_equal(w, [0., 0.])
        out = audit_comparison([.125, .375, .5], [.25, .25, .5],
            ['a', 'a', 'b'], {}, lambda _: self.fail('unnecessary audit'), budget=2,
            strategy='decision')
        self.assertEqual(out['status'], 'accept')
        self.assertEqual(out['paid_additional_labels'], 0)

    def test_safety_does_not_depend_on_prior_calibration(self):
        planner = BinaryDecisionPlanner(('a', 'b'), (-.5, .5), prior=(.99, .01))
        for values in itertools.product((0, 1), repeat=2):
            known = {}
            for _ in range(2):
                _, _, g = planner.plan(known, 2-len(known))
                if g is None:
                    break
                known[g] = dict(zip(('a', 'b'), values))[g]
            status = decision(*contrast_bounds(('a', 'b'), (-.5, .5), known))
            self.assertEqual(status, 'accept' if values[1] >= values[0] else 'reject')

    def test_bad_contracts_and_source_cap_fail_loudly(self):
        with self.assertRaises(ValueError):
            BinaryDecisionPlanner(tuple(range(11)), [1.]*11)
        with self.assertRaises(ValueError):
            BinaryDecisionPlanner(('a',), (.5,), prior=(0.,))
        with self.assertRaises(ValueError):
            audit_comparison([.25, .75], [.5, .5], ['a', 'b'], {},
                lambda _: .5, budget=1, strategy='decision')
        with self.assertRaises(ValueError):
            minimum_witness(('a',), (.5,), {'a': 1.}, {'a': 0.})
        with self.assertRaises(ValueError):
            audit_comparison([.25, .75], [.5, .5], ['a', 'b'], {},
                lambda _: 0., budget=1, strategy='stream', order=['a', 'a'])

    def test_impossible_budget_abstains_without_wasting_a_label(self):
        out = audit_comparison([0., 0., .5, .5], [.25]*4, list('abcd'), {},
            lambda _: self.fail('no possible resolution within one query'),
            budget=1, threshold=.01, strategy='decision')
        self.assertEqual(out['status'], 'unresolved')
        self.assertEqual(out['paid_additional_labels'], 0)


if __name__ == '__main__':
    unittest.main()
