"""Sharp, bounded-reward policy-gain identification from paid source labels.

No unqueried trusted rewards are consulted. Sources are equivalence classes of
identical candidate text inside a single task. The grouping assumption means
that all occurrences of one source share the same deterministic trusted score.
"""
import numpy as np


def identified_gain(p, q, source_ids, revealed, *, lower=0., upper=1.):
    """Exact feasible [min,max] for (p-q)@r under independent group bounds.

    `revealed` maps source ID to its paid trusted reward. Constant group bounds
    [lower,upper] apply only to as-yet-unrevealed source IDs. Policy p and q
    must be normalized on the *same* finite bank; no assumed reward model.
    """
    p=np.asarray(p,dtype=float);q=np.asarray(q,dtype=float)
    ids=np.asarray(source_ids)
    if p.ndim!=1 or p.shape!=q.shape or ids.shape!=p.shape or len(p)==0:
        raise ValueError('policies and sources must be aligned nonempty vectors')
    if (not np.isfinite(p).all() or not np.isfinite(q).all() or
            np.min(p)<0 or np.min(q)<0 or
            not np.isclose(p.sum(),1.,atol=1e-10) or
            not np.isclose(q.sum(),1.,atol=1e-10)):
        raise ValueError('invalid probability vectors')
    if not (np.isfinite(lower) and np.isfinite(upper) and lower<=upper):
        raise ValueError('invalid reward range')
    groups={}
    for source,delta in zip(ids,p-q):
        groups[source]=groups.get(source,0.)+float(delta)
    if set(revealed)-set(groups): raise ValueError('unknown revealed source')
    lo=hi=0.
    for source,w in groups.items():
        if source in revealed:
            reward=float(revealed[source])
            if not np.isfinite(reward) or not lower<=reward<=upper:
                raise ValueError('invalid revealed reward')
            lo+=w*reward;hi+=w*reward
        else:
            lo+=min(w*lower,w*upper)
            hi+=max(w*lower,w*upper)
    return lo,hi
