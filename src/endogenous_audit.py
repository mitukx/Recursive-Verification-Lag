"""Finite-bank lookahead: paid labels change the next fitted proposal.

This is a small exact Bayes-adaptive control benchmark, not a scalable algorithm
or a new Bellman theorem. Planning uses an explicit independent binary prior;
every deployed update uses a distribution-free source-box certificate. No
unqueried trusted outcomes are accepted by the planner.
"""
from functools import lru_cache
import numpy as np
from src.candidate_bank_experiment import fit_verifier, propose, task_policy
from src.decision_audit import source_contrast, contrast_bounds, decision


class EndogenousAuditPlanner:
    """Maximize expected terminal baseline gain; ties minimize label cost.

Allowed actions: stop, query a new source (refit), or apply the currently fitted
proposal if certified. Queries do not consume update slots; accepted updates
do. A rejected proposal need not consume a slot. This action space differs from
freezing a proposal during several queries. Optimality is restricted to these
actions and the declared prior, at most eight sources by default.
"""
    def __init__(self, df, sources, cfg, *, prior=None, gain_floor=.01,
                 max_sources=8, max_states=200000):
        if 'trusted_score' in df:
            raise ValueError('planner must receive an unlabeled feature bank')
        self.df = df.reset_index(drop=True).copy()
        self.sources = np.asarray(sources)
        if self.df.task_id.nunique() != 1 or self.sources.shape != (len(df),):
            raise ValueError('one task with aligned sources required')
        self.baseline = task_policy(self.df, self.df.base_logprob.to_numpy(float))
        self.groups, _ = source_contrast(self.baseline, self.baseline, self.sources)
        if len(self.groups) > max_sources:
            raise ValueError('exact planner exceeds source cap')
        if (not np.isfinite([gain_floor, cfg.ridge, cfg.eta, cfg.score_scale, cfg.best_of_n]).all()
                or gain_floor < 0 or cfg.ridge <= 0
                or cfg.representation not in ('all', 'public')
                or cfg.optimizer not in ('soft', 'bon') or cfg.eta < 0
                or cfg.score_scale <= 0 or cfg.best_of_n < 1 or int(cfg.best_of_n) != cfg.best_of_n):
            raise ValueError('invalid model or certificate configuration')
        self.cfg, self.gain_floor = cfg, float(gain_floor)
        self.prior = tuple(.5 for _ in self.groups) if prior is None else tuple(prior)
        if len(self.prior) != len(self.groups) or any(not np.isfinite(v) or not 0 < v < 1 for v in self.prior):
            raise ValueError('independent binary prior must have full support')
        self.first = [int(np.flatnonzero(self.sources == g)[0]) for g in self.groups]
        self.members = [np.flatnonzero(self.sources == g) for g in self.groups]
        cols = sorted(c for c in df if c.startswith('f::') and
                      (cfg.representation == 'all' or c == 'f::public_score'))
        if not cols:
            raise ValueError('no verifier features')
        self.X = np.column_stack([np.ones(len(df)), self.df[cols].to_numpy(float)])
        if not np.isfinite(self.X).all():
            raise ValueError('nonfinite features')
        if isinstance(max_states, bool) or int(max_states) != max_states or max_states < 1:
            raise ValueError('positive integer state cap required')
        self.max_states = max_states
        self._visited = 0
        self._fit = lru_cache(None)(self._fit_state)
        self._proposal = lru_cache(None)(self._proposal_state)
        self._solve = lru_cache(None)(self._solve_state)

    def state(self, known):
        if set(known)-set(self.groups) or any(v not in (0, 1) for v in known.values()):
            raise ValueError('paid binary source labels required')
        return tuple(int(known[g]) if g in known else -1 for g in self.groups)

    def _fit_state(self, state):
        ii = [i for i, v in enumerate(state) if v != -1]
        if not ii:
            return np.zeros(len(self.df))
        theta = fit_verifier(self.X[[self.first[i] for i in ii]],
                             np.array([state[i] for i in ii]), self.cfg.ridge)
        return (self.X @ theta)*self.cfg.score_scale

    def _proposal_state(self, state, policy):
        q = propose(self.df, np.array(policy), self._fit(state), self.cfg)
        groups, weights = source_contrast(q, self.baseline, self.sources)
        known = {g: v for g, v in zip(groups, state) if v != -1}
        lo, hi = contrast_bounds(groups, weights, known)
        return tuple(q.tolist()), lo, hi, tuple(weights.tolist())

    def terminal_value(self, state, policy):
        expected = [self.prior[i] if v == -1 else v for i, v in enumerate(state)]
        delta = np.asarray(policy)-self.baseline
        return float(sum(delta[ii].sum()*r for ii, r in zip(self.members, expected)))

    def _solve_state(self, state, policy, budget, updates):
        self._visited += 1
        if self._visited > self.max_states:
            raise RuntimeError('declared planning state cap exceeded; no fallback')
        best = (self.terminal_value(state, policy), 0., ('stop', None))
        if updates == 0:
            return best
        q, lo, _, _ = self._proposal(state, policy)
        choices = []
        if lo >= self.gain_floor and q != policy:
            value, cost, _ = self._solve(state, q, budget, updates-1)
            choices.append((value, cost, ('update', None)))
        if budget:
            for i, g in enumerate(self.groups):
                if state[i] != -1:
                    continue
                children = [self._solve(state[:i]+(v,)+state[i+1:], policy,
                                        budget-1, updates) for v in (0, 1)]
                pi = self.prior[i]
                value = (1-pi)*children[0][0]+pi*children[1][0]
                cost = 1+(1-pi)*children[0][1]+pi*children[1][1]
                choices.append((value, cost, ('audit', g)))
        # Ordinary floating point arithmetic; 1e-12 affects utility ties only,
        # never relaxes the acceptance certificate.
        for candidate in choices:
            if (candidate[0] > best[0]+1e-12 or
                    (abs(candidate[0]-best[0]) <= 1e-12 and candidate[1] < best[1]-1e-12)):
                best = candidate
        return best

    def plan(self, known, policy, budget, updates):
        for n in (budget, updates):
            if isinstance(n, bool) or int(n) != n or n < 0:
                raise ValueError('nonnegative integer resources required')
        state = self.state(known)
        source_contrast(policy, self.baseline, self.sources)
        budget = min(int(budget), state.count(-1))
        return self._solve(state, tuple(np.asarray(policy, float).tolist()), budget, int(updates))

    @property
    def states_visited(self):
        return self._visited


def endogenous_run(planner, known, oracle, *, budget, updates, strategy='lookahead', order=None):
    """Deploy a controller using paid oracle(source) as its only label channel.

Returns the final policy and replayable actions, without evaluator outcomes.
myopic refits after each query and queries only for an unresolved comparison.
    stream spends the full additional cap before attempting certified updates.
    escape adds one stream query if all paid labels are zero and myopic stops.
    one_update replans for one accepted update at a time, with the full label cap.
    direct selects among paid known positives, querying the same source stream
    until that policy is certified. It deliberately leaves the fitted-proposal
    action family; the exact planner does not optimize over this baseline.
"""
    if strategy not in ('lookahead', 'myopic', 'stream', 'escape', 'one_update', 'direct'):
        raise ValueError('unknown strategy')
    for n in (budget, updates):
        if isinstance(n, bool) or int(n) != n or n < 0:
            raise ValueError('nonnegative integer resources required')
    known = dict(known)
    planner.state(known)
    if strategy in ('stream', 'escape', 'direct') and (order is None or len(order) != len(planner.groups) or set(order) != set(planner.groups)):
        raise ValueError('stream order must be a source permutation')
    p = tuple(planner.baseline.tolist())
    paid, trace = [], []
    escape_used = False
    while True:
        state = planner.state(known)
        q, lo, hi, weights = planner._proposal(state, p)
        if strategy == 'direct':
            mask = np.isin(planner.sources, [g for g, v in known.items() if v == 1.])
            proposal = planner.baseline*mask
            q = tuple((proposal/proposal.sum()).tolist()) if proposal.sum() else p
            groups, weights = source_contrast(q, planner.baseline, planner.sources)
            lo, hi = contrast_bounds(groups, weights, known)
        value, expected_cost = None, None
        if updates == 0:
            action = ('stop', None)
        elif strategy == 'direct':
            if lo >= planner.gain_floor:
                action = ('update', None) if q != p else ('stop', None)
            elif budget and state.count(-1):
                action = ('audit', next(g for g in order if g not in known))
            else:
                action = ('stop', None)
        elif strategy in ('lookahead', 'one_update'):
            horizon = min(1, updates) if strategy == 'one_update' else updates
            value, expected_cost, action = planner.plan(known, p, budget, horizon)
        elif strategy == 'stream' and budget and state.count(-1):
            action = ('audit', next(g for g in order if g not in known))
        elif lo >= planner.gain_floor and q != p:
            action = ('update', None)
        elif strategy in ('myopic', 'escape') and budget and decision(lo, hi, planner.gain_floor) == 'unresolved':
            remaining = [i for i, g in enumerate(planner.groups) if g not in known and weights[i] != 0.]
            i = min(remaining, key=lambda i: (-abs(weights[i]), str(planner.groups[i])))
            action = ('audit', planner.groups[i])
        elif strategy == 'escape' and not escape_used and budget and state.count(-1) and not any(known.values()):
            action = ('audit', next(g for g in order if g not in known))
            escape_used = True
        else:
            action = ('stop', None)
        entry = {'action': action[0], 'source': action[1], 'known': dict(known),
                 'budget_remaining': budget, 'updates_remaining': updates,
                 'policy': list(p), 'proposal': list(q), 'lower': lo, 'upper': hi,
                 'prior_terminal_value': value, 'prior_expected_additional_cost': expected_cost}
        if action[0] == 'stop':
            trace.append(entry)
            break
        if action[0] == 'audit':
            g = action[1]
            if budget <= 0 or g in known:
                raise AssertionError('invalid or repeated source charge')
            label = float(oracle(g))
            if label not in (0., 1.):
                raise ValueError('binary oracle required')
            known[g] = label
            entry['paid_label'] = label
            paid.append(g)
            budget -= 1
        else:
            if lo < planner.gain_floor:
                raise AssertionError('uncertified update')
            p = q
            updates -= 1
        trace.append(entry)
    return {'policy': np.array(p), 'revealed': known, 'queries': tuple(paid),
            'trace': trace, 'planning_states': planner.states_visited}
