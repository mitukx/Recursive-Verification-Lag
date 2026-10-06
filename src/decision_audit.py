"""Decision-directed exact source audits on one frozen policy comparison.

The certificate is distribution-free on a deterministic finite bank. The
dynamic program is optimal ONLY under its supplied independent binary prior;
the prior controls query order, never the certificate. Hindsight witnesses
are evaluation-only and must never be used to choose a paid query.
"""
from functools import lru_cache
from math import fsum
import numpy as np
from src.identified_gain import identified_gain


def source_contrast(proposal, baseline, sources):
    sources = np.asarray(sources)
    identified_gain(proposal, baseline, sources, {})  # canonical validation
    groups = tuple(sorted(set(sources.tolist()), key=str))
    weights = np.array([fsum(float(x) for x in
        (np.asarray(proposal)-np.asarray(baseline))[sources == g]) for g in groups])
    return groups, weights


def contrast_bounds(groups, weights, revealed):
    if set(revealed)-set(groups):
        raise ValueError('unknown source label')
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in revealed.values()):
        raise ValueError('reward outside [0,1]')
    fixed = [float(w)*float(revealed[g]) for g, w in zip(groups, weights) if g in revealed]
    lower = fsum(fixed + [min(0., float(w)) for g, w in zip(groups, weights) if g not in revealed])
    upper = fsum(fixed + [max(0., float(w)) for g, w in zip(groups, weights) if g not in revealed])
    return lower, upper


def decision(lower, upper, threshold=0.):
    """Numerical rule: certify gain >= threshold, otherwise gain < threshold.

No relaxed safety tolerance: even a tiny negative lower bound is not accepted.
The exact mathematical rule is subject to ordinary float64 rounding.
"""
    if lower >= threshold:
        return 'accept'
    if upper < threshold:
        return 'reject'
    return 'unresolved'


class BinaryDecisionPlanner:
    """Exact budgeted stochastic threshold evaluation, at most 10 sources.

Maximize prior probability of resolving the sign within remaining paid calls.
Among ties minimize prior expected paid calls, then use source lexical order.
All prior branches are considered; no unqueried trusted outcome is input.
The target contrast is frozen throughout this planning problem.
"""
    def __init__(self, groups, weights, *, prior=None, threshold=0., max_sources=10):
        self.groups = tuple(groups)
        self.weights = tuple(float(x) for x in weights)
        if (not self.groups or len(self.groups) != len(self.weights)
                or len(set(self.groups)) != len(self.groups)
                or not np.isfinite(self.weights).all() or not np.isfinite(threshold)):
            raise ValueError('invalid contrast')
        if len(groups) > max_sources:
            raise ValueError('exact decision planner exceeds declared source cap')
        self.prior = tuple(.5 for _ in groups) if prior is None else tuple(prior)
        if len(self.prior) != len(groups) or any(not np.isfinite(v) or not 0 < v < 1 for v in self.prior):
            raise ValueError('prior must have full binary support')
        self.threshold = float(threshold)
        self._solve = lru_cache(None)(self._solve_state)

    def _solve_state(self, state, budget):
        revealed = {g: x for g, x in zip(self.groups, state) if x != -1}
        if decision(*contrast_bounds(self.groups, self.weights, revealed), self.threshold) != 'unresolved':
            return 1., 0., None
        if budget == 0:
            return 0., 0., None
        choices = []
        for i, g in enumerate(self.groups):
            if state[i] != -1 or self.weights[i] == 0.:
                continue
            branches = []
            for label in (0, 1):
                child = state[:i] + (label,) + state[i+1:]
                branches.append(self._solve(child, budget-1))
            pi = self.prior[i]
            success = (1-pi)*branches[0][0] + pi*branches[1][0]
            cost = 1 + (1-pi)*branches[0][1] + pi*branches[1][1]
            choices.append((-success, cost, str(g), i))
        if not choices:
            return 0., 0., None
        neg_success, cost, _, i = min(choices)
        if neg_success == 0.:
            # Abstention is feasible: if no binary world can resolve within
            # this cap, spending a label has no value for the stated objective.
            return 0., 0., None
        return -neg_success, cost, self.groups[i]

    def plan(self, revealed, budget):
        if isinstance(budget, bool) or int(budget) != budget or budget < 0:
            raise ValueError('invalid audit budget')
        if set(revealed)-set(self.groups) or any(v not in (0, 1) for v in revealed.values()):
            raise ValueError('binary source labels required')
        state = tuple(int(revealed[g]) if g in revealed else -1 for g in self.groups)
        return self._solve(state, min(int(budget), state.count(-1)))

    @property
    def states_visited(self):
        return self._solve.cache_info().currsize


def minimum_witness(groups, weights, rewards, revealed, *, threshold=0.):
    """Evaluation-only minimum *additional* labels for the realized decision.

Unit cost and independent [0,1] reward bounds. Works for fractional rewards too.
The minimizing set knows the full reward vector and is not a usable controller.
"""
    if set(rewards) != set(groups):
        raise ValueError('full realized source rewards required')
    contrast_bounds(groups, weights, rewards)
    if any(rewards[g] != v for g, v in revealed.items()):
        raise ValueError('revealed labels disagree with evaluator')
    lo, hi = contrast_bounds(groups, weights, revealed)
    status = decision(lo, hi, threshold)
    if status != 'unresolved':
        return 0, (), status
    gain = fsum(w*rewards[g] for g, w in zip(groups, weights))
    status = 'accept' if gain >= threshold else 'reject'
    # Revealing a source raises L by c or lowers U by d, always nonnegative.
    corrections = []
    for g, w in zip(groups, weights):
        if g in revealed:
            continue
        c = w*rewards[g]-min(0., w) if status == 'accept' else max(0., w)-w*rewards[g]
        corrections.append((-c, str(g), g))
    selected = {}
    for _, _, g in sorted(corrections):
        selected[g] = rewards[g]
        new_lo, new_hi = contrast_bounds(groups, weights, {**revealed, **selected})
        if decision(new_lo, new_hi, threshold) == status:
            return len(selected), tuple(selected), status
    raise ArithmeticError('all source labels failed to identify realized gain')


def audit_comparison(proposal, baseline, sources, revealed, oracle, *,
                     budget, strategy='impact', threshold=0., order=None):
    """Freeze proposal; charge and reveal one source at a time, then decide.

oracle(source) is the ONLY trusted-label channel. No verifier refits mid-audit.
Initial labels are sunk cost and excluded from this additional-label budget.
"""
    if strategy not in ('impact', 'decision', 'stream'):
        raise ValueError('unknown acquisition strategy')
    if isinstance(budget, bool) or int(budget) != budget or budget < 0:
        raise ValueError('invalid audit budget')
    if not np.isfinite(threshold):
        raise ValueError('invalid gain threshold')
    groups, weights = source_contrast(proposal, baseline, sources)
    revealed = dict(revealed)
    contrast_bounds(groups, weights, revealed)
    planner = BinaryDecisionPlanner(groups, weights, threshold=threshold) if strategy == 'decision' else None
    if strategy == 'stream' and (order is None or len(order) != len(groups) or set(order) != set(groups)):
        raise ValueError('stream order must be a source permutation')
    queries = []
    planning_states = 0
    while True:
        lo, hi = contrast_bounds(groups, weights, revealed)
        status = decision(lo, hi, threshold)
        if status != 'unresolved' or len(queries) == budget:
            break
        remaining = [g for g, w in zip(groups, weights) if g not in revealed and w != 0.]
        if not remaining:
            raise ArithmeticError('no informative source remains')
        if planner:
            _, _, target = planner.plan(revealed, budget-len(queries))
            planning_states = planner.states_visited
            if target is None:
                break
        elif strategy == 'impact':
            w_by_g = dict(zip(groups, weights))
            target = min(remaining, key=lambda g: (-abs(w_by_g[g]), str(g)))
        else:
            target = next(g for g in order if g in remaining)
        reward = float(oracle(target))
        if not np.isfinite(reward) or not 0 <= reward <= 1 or (planner and reward not in (0, 1)):
            raise ValueError('oracle returned invalid reward')
        revealed[target] = reward
        queries.append(target)
    return {'status': status, 'lower': lo, 'upper': hi, 'revealed': revealed,
            'queries': tuple(queries), 'paid_additional_labels': len(queries),
            'planning_states': planning_states}
