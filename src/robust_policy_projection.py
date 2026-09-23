"""Closest certified policy to a proposed finite-bank distribution.

The LP uses only paid deterministic source labels. The arbitrary reward on
unobserved source classes remains in [0,1]. This is a geometric diagnostic,
not a model of out-of-bank generalization or a guarantee of strict progress.
"""
import numpy as np
from scipy.optimize import linprog
from src.identified_gain import identified_gain


def closest_certified_policy(proposal, baseline, source_ids, revealed):
    """Minimize ||q-proposal||_1 over the simplex with sharp gain lower >= 0.

    Returns (q, distance). The baseline itself is feasible, so numerical
    solver failure is reported rather than silently falling back to it.
    """
    z = np.asarray(proposal, float)
    b = np.asarray(baseline, float)
    ids = np.asarray(source_ids)
    identified_gain(z, b, ids, revealed)  # common validation contract
    groups = list(dict.fromkeys(ids))
    unknown = [g for g in groups if g not in revealed]
    n = len(z)
    m = len(unknown)
    # x = (q[n], absolute_deviation[n], missing_mass[m]).
    objective = np.r_[np.zeros(n), np.ones(n), np.zeros(m)]
    rows = []
    rhs = []
    for i in range(n):
        row = np.zeros(2*n+m); row[i] = 1.; row[n+i] = -1.
        rows.append(row); rhs.append(z[i])
        row = np.zeros(2*n+m); row[i] = -1.; row[n+i] = -1.
        rows.append(row); rhs.append(-z[i])
    for j,g in enumerate(unknown):
        row = np.zeros(2*n+m); row[:n] = -(ids == g).astype(float)
        row[2*n+j] = -1.
        rows.append(row); rhs.append(-float(b[ids == g].sum()))
    # L(q,b) = sum_known (q_g-b_g)*r_g - sum_unknown missing_g.
    safety = np.zeros(2*n+m)
    for g,r in revealed.items():
        safety[:n] -= (ids == g)*r
    safety[2*n:] = 1.
    rows.append(safety)
    rhs.append(-sum(float(b[ids == g].sum())*r for g,r in revealed.items()))
    eq = np.zeros((1,2*n+m)); eq[0,:n] = 1.
    result = linprog(objective, A_ub=np.asarray(rows), b_ub=np.asarray(rhs),
                     A_eq=eq, b_eq=np.ones(1), bounds=(0,None),method='highs',
                     options={'primal_feasibility_tolerance':1e-10,
                              'dual_feasibility_tolerance':1e-10})
    if not result.success:
        raise RuntimeError(f'certified projection LP failed: {result.message}')
    q = np.maximum(result.x[:n],0.)
    # Ensure any LP feasibility tolerance cannot turn into a false guarantee.
    lo = identified_gain(q,b,ids,revealed)[0]
    if abs(q.sum()-1.)>1e-9 or lo < -1e-9:
        raise RuntimeError(f'projection violates certificate: {lo}')
    return q, float(np.abs(q-z).sum())
