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


def maximal_certified_mix(current, proposal, baseline, source_ids, revealed,
                          *, iterations=55, tolerance=1e-12):
    """Largest [current,proposal] mixture safe relative to fixed baseline.

    The sharp lower bound is concave in policy. Its nonnegative superlevel set
    along this line is an interval containing alpha=0 when current is safe.
    We evaluate the bound at the returned mixture; numerical error is bounded
    by the bisection interval, not a statistical uncertainty statement.
    """
    current=np.asarray(current,float);proposal=np.asarray(proposal,float)
    baseline=np.asarray(baseline,float)
    if current.shape!=proposal.shape or current.shape!=baseline.shape:
        raise ValueError('policy shapes differ')
    if identified_gain(current,baseline,source_ids,revealed)[0]<-tolerance:
        raise ValueError('current policy lacks a safety certificate')
    if identified_gain(proposal,baseline,source_ids,revealed)[0]>=0.:
        return proposal.copy(),1.
    lo=0.;hi=1.
    for _ in range(iterations):
        mid=(lo+hi)/2
        candidate=(1-mid)*current+mid*proposal
        if identified_gain(candidate,baseline,source_ids,revealed)[0]>=0.:
            lo=mid
        else:hi=mid
    candidate=(1-lo)*current+lo*proposal
    if identified_gain(candidate,baseline,source_ids,revealed)[0]<-tolerance:
        raise AssertionError('mixed policy failed safety certificate')
    return candidate,lo


def lower_bound_right_slope(current,proposal,baseline,source_ids,revealed,
                            *,lower=0.,upper=1.,zero_tolerance=1e-13):
    """Exact right directional derivative of the sharp gain lower bound.

    Zero-mass groups use the lower of the two possible directional endpoint
    products. This is a local feasibility diagnostic, not a reward estimate.
    """
    current=np.asarray(current,float);proposal=np.asarray(proposal,float)
    baseline=np.asarray(baseline,float);ids=np.asarray(source_ids)
    if ids.shape!=current.shape or proposal.shape!=current.shape or baseline.shape!=current.shape:
        raise ValueError('misaligned policies and source IDs')
    # Validate all probabilities/labels with the canonical certificate first.
    identified_gain(current,baseline,ids,revealed,lower=lower,upper=upper)
    base={};velocity={}
    for source,w,v in zip(ids,current-baseline,proposal-current):
        base[source]=base.get(source,0.)+float(w)
        velocity[source]=velocity.get(source,0.)+float(v)
    slope=0.
    for source,w in base.items():
        v=velocity[source]
        if source in revealed:slope+=v*float(revealed[source])
        elif w>zero_tolerance:slope+=v*lower
        elif w< -zero_tolerance:slope+=v*upper
        else:slope+=min(v*lower,v*upper)
    return slope
