"""Sharp finite-bank robust certificate, separate from heuristic refresh.

For deterministic exhaustive rewards, paid labels are exact on audited atoms.
Unknown atoms have reward in [0,1]. Bounds are simultaneous for every adaptive
comparison; no sampling confidence interval or hidden reward access is used.
This standard box-uncertainty result is not an optimal label-budget theorem.
"""
import numpy as np


def exact_box_gain_bounds(p,q,known):
    p=np.asarray(p,float); q=np.asarray(q,float); known=np.asarray(known,float)
    if p.shape!=q.shape or p.shape!=known.shape or p.ndim!=1: raise ValueError('shape mismatch')
    if not np.isfinite(p).all() or not np.isfinite(q).all() or min(p.min(),q.min())<0 or not np.isclose(p.sum(),1) or not np.isclose(q.sum(),1):
        raise ValueError('invalid policies')
    seen=~np.isnan(known)
    if not np.isfinite(known[seen]).all() or np.any((known[seen]<0)|(known[seen]>1)): raise ValueError('invalid labels')
    d=q-p; center=float(d[seen]@known[seen])
    return center+float(np.minimum(d[~seen],0).sum()),center+float(np.maximum(d[~seen],0).sum())


def error_envelope_lower_bound(p,q,features,theta,precision,beta,epsilon):
    """Conditional bound ONLY under the explicitly assumed ellipsoid + L-infinity residual.

    Does not estimate/validate beta or epsilon. Passing zero is an assumption,
    not an observed proof of realizability. No statistical theorem is implicit.
    """
    if beta<0 or epsilon<0: raise ValueError('negative envelope')
    delta=np.asarray(q)-np.asarray(p); d=delta@features
    radius=beta*np.sqrt(max(0.,float(d@np.linalg.solve(precision,d))))+epsilon*np.abs(delta).sum()
    return float(d@theta-radius)
