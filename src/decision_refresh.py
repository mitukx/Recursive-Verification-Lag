"""Freeze each proposal while certifying; refit only between decisions.

Adaptive query order is separate from refresh timing. Safety relative to the
original finite-bank baseline follows from the paid-label box certificate.
It does not imply monotone stepwise progress or improvement outside the bank.
"""
import numpy as np
import pandas as pd
from src.candidate_bank_experiment import task_policy, fit_verifier, propose
from src.decision_audit import audit_comparison
from src.source_audit import SourceAudit


def decision_refresh_run(df, sources, cfg, *, budget=6, initial_audits=2,
                         strategy='impact', gain_floor=.01, return_trace=False):
    if df.task_id.nunique() != 1:
        raise ValueError('one task per certified loop required')
    sources = np.asarray(sources)
    if sources.shape != (len(df),) or not 1 <= initial_audits <= budget <= len(set(sources)):
        raise ValueError('invalid source budget or alignment')
    if not np.isfinite(gain_floor) or gain_floor < 0:
        raise ValueError('nonnegative finite gain floor required')
    if cfg.rounds < 1 or cfg.ridge <= 0 or cfg.representation not in ('public', 'all'):
        raise ValueError('invalid verifier configuration')
    df = df.reset_index(drop=True)
    # Labels here are an offline oracle. Only paid oracle() calls enter decisions.
    y = df.trusted_score.to_numpy(float)
    first = {g: int(np.flatnonzero(sources == g)[0]) for g in set(sources)}
    if strategy == 'decision' and not np.isin(y, [0., 1.]).all():
        raise ValueError('decision strategy requires binary trusted labels')
    cols = sorted(c for c in df if c.startswith('f::') and
        (cfg.representation == 'all' or c == 'f::public_score'))
    X = np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)])
    p = task_policy(df, df.base_logprob.to_numpy(float))
    baseline = p.copy()
    sampler = SourceAudit(sources, cfg.seed, 'stream')
    order = [sampler.groups[i] for i in sampler.order]
    paid = []
    def oracle(g):
        if g in paid:
            raise AssertionError('repeat source charge')
        paid.append(g)
        return float(y[first[g]])
    known = {g: oracle(g) for g in order[:initial_audits]}
    rows, trace = [], []
    for t in range(cfg.rounds):
        indices = [first[g] for g in known]
        labels = np.array([known[g] for g in known])
        theta = fit_verifier(X[indices], labels, cfg.ridge)
        proposal = propose(df, p, (X @ theta)*cfg.score_scale, cfg)
        # The proposal remains frozen for the full paid-query sequence.
        out = audit_comparison(proposal, baseline, sources, known, oracle,
            budget=budget-len(paid), strategy=strategy, threshold=gain_floor, order=order)
        known = out['revealed']
        accepted = out['status'] == 'accept'
        if accepted:
            p = proposal
        # These outcomes are computed AFTER the policy decision, for evaluation.
        gain = float((p-baseline) @ y)
        if gain < -1e-9:
            raise AssertionError('finite-bank baseline safety violated')
        masses = np.array([p[sources == g].sum() for g in order])
        rows.append({'round': t+1, 'strategy': strategy, 'budget_cap': budget,
            'gain_floor': gain_floor, 'accepted': int(accepted), 'status': out['status'],
            'paid_source_labels': len(paid), 'new_source_labels': len(out['queries']),
            'candidate_lower': out['lower'], 'candidate_upper': out['upper'],
            'gain_evaluation_only': gain, 'planning_states': out['planning_states'],
            'audited_policy_mass': float(p[np.isin(sources, list(known))].sum()),
            'effective_source_support': float(1/(masses @ masses))})
        trace.append({'policy': p.copy(), 'proposal': proposal.copy(),
            'queries': tuple(out['queries']), 'paid_sources': tuple(paid)})
    result = pd.DataFrame(rows)
    return (result, trace) if return_trace else result
