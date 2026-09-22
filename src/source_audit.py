"""Exact-source caching and label-blind acquisition for timing interventions.

Source hashes are observable text identities, not hidden semantic equivalence.
Every selected source costs one exact oracle call, and no source is queried twice.
"""
import numpy as np

class SourceAudit:
    def __init__(self, source_ids, seed, mode='stream', exploration=.05):
        if mode not in ('stream','policy'):raise ValueError('unknown acquisition')
        if not 0<exploration<=1:raise ValueError('exploration must be in (0,1]')
        self.ids=np.asarray(source_ids)
        self.groups,self.first,self.inverse=np.unique(self.ids,return_index=True,return_inverse=True)
        self.rng=np.random.default_rng(seed)
        self.order=self.rng.permutation(len(self.groups))
        self.mode=mode;self.exploration=exploration
    def __call__(self,p,n,audit):
        used=set(self.inverse[list(audit)]) if audit else set()
        remaining=np.asarray([i for i in self.order if i not in used],dtype=int)
        if n>len(remaining):raise ValueError('source audit budget exceeds remaining sources')
        # Initial group sequence is shared by acquisition modes and all timings.
        if self.mode=='stream' or not audit:chosen=remaining[:n]
        else:
            mass=np.bincount(self.inverse,weights=p,minlength=len(self.groups))[remaining]
            weights=(mass/mass.sum()) if mass.sum()>0 else np.ones(len(mass))/len(mass)
            weights=(1-self.exploration)*weights+self.exploration/len(weights)
            chosen=self.rng.choice(remaining,size=n,replace=False,p=weights)
        return self.first[chosen]
