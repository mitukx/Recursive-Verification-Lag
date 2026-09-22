"""Recursive selection on a frozen bank, with observable pre-update refresh.

Empirical sample occurrences have equal base mass (duplicates retained). Explicit
base_logprob is supported for legacy enumerated banks, not computed from LM
sequence likelihoods. All metrics are on the finite bank, not full LM support.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from .refresh_controller import MovementState, ScalarThreshold, GeometryMarginController
except ImportError:
    from refresh_controller import MovementState, ScalarThreshold, GeometryMarginController


@dataclass
class Config:
    eta: float = 1.0
    refresh_interval: int = 4
    rounds: int = 24
    audit_per_refresh: int = 64
    seed: int = 0
    ridge: float = 1e-4
    optimizer: str = 'soft'
    best_of_n: int = 4
    controller: str = 'fixed'
    threshold: float = 0.5
    total_audit_budget: int = 256
    representation: str = 'all'
    score_scale: float = 1.0
    failure_tolerance: float = 1e-10


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows: raise ValueError('candidate bank is empty')
    names=sorted(set().union(*(set(r.get('features',{})) for r in rows)) | {'public_score'})
    flat=[]
    for r in rows:
        if r.get('base_weight_semantics')=='empirical_sample_occurrence' and r.get('base_logprob',0)!=0:
            raise ValueError('empirical occurrences must not be weighted by likelihood again')
        item={k:r[k] for k in ('task_id','candidate_id','trusted_score')}
        item['base_logprob']=float(r.get('base_logprob',0))
        for name in names:
            value=r.get('features',{}).get(name,r.get(name) if name=='public_score' else None)
            if value is None: raise ValueError(f'missing feature: {name}')
            item['f::'+name]=float(value)
        if 'public_score' in r and item['f::public_score']!=float(r['public_score']):
            raise ValueError('inconsistent public score')
        flat.append(item)
    df=pd.DataFrame(flat)
    if df[['task_id','candidate_id']].duplicated().any(): raise ValueError('duplicate candidate IDs')
    if not np.isfinite(df.select_dtypes('number')).all().all(): raise ValueError('non-finite bank')
    if not df.trusted_score.between(0,1).all(): raise ValueError('reward outside [0,1]')
    return df


def softmax(logits):
    e=np.exp(logits-np.max(logits)); return e/e.sum()


def fit_verifier(X,y,ridge):
    return np.linalg.solve(X.T@X+ridge*np.eye(X.shape[1]),X.T@y)


def task_policy(df,logits):
    probs=np.zeros(len(df)); groups=df.groupby('task_id',sort=True).indices
    for ii in groups.values(): probs[ii]=softmax(logits[ii])/len(groups)
    return probs


def expected_score(df,probs,col): return float(probs@df[col].to_numpy(float))


def best_of_n(p, scores, n):
    """Exact winner law for n IID draws, random tie-breaking among tied draws.

    Group mass is F(score)^n-F(score-)^n; within ties preserve base proportions.
    Zero probabilities remain zero: no artificial epsilon support injection.
    """
    if n<1 or int(n)!=n: raise ValueError('N must be a positive integer')
    q=np.zeros_like(p); cdf=0.0
    for score in np.unique(scores):
        ii=np.flatnonzero(scores==score); mass=float(p[ii].sum())
        hi=min(1.0,cdf+mass)
        if mass>0:
            win=hi**n*(-np.expm1(n*np.log(cdf/hi))) if cdf>0 else hi**n
            q[ii]=win*p[ii]/mass
        cdf=hi
    return q/q.sum()


def propose(df,p,score,cfg):
    q=np.zeros_like(p)
    for ii in df.groupby('task_id',sort=True).indices.values():
        mass=p[ii].sum(); local=p[ii]/mass
        if cfg.optimizer=='soft':
            logp=np.full_like(local,-np.inf); mask=local>0; logp[mask]=np.log(local[mask])
            q[ii]=mass*softmax(logp+cfg.eta*score[ii])
        else: q[ii]=mass*best_of_n(local,score[ii],cfg.best_of_n)
    return q


def divergence(q,p):
    mask=q>0
    if np.any(p[mask]<=0): return float('inf'),float('inf')
    logs=np.log(q[mask])-np.log(p[mask])
    return max(0.,float(q[mask]@logs)),float(np.max(logs))


def restricted_geometry(p,q,X,mu):
    d=(q-p)@X; cov=X.T@(mu[:,None]*X)
    inv=np.linalg.pinv(cov,hermitian=True)
    if np.linalg.norm(d-cov@inv@d)>1e-8*(1+np.linalg.norm(d)): return float('inf')
    return max(0.,float(d@inv@d))


def movement(p,q,X,score):
    kl,mx=divergence(q,p)
    return {'kl_from_refresh':kl,'max_log_density_ratio':mx,
            'restricted_geometry':restricted_geometry(p,q,X,p),
            'proxy_margin':float((q-p)@score),'total_variation':float(abs(q-p).sum()/2)}


def run(df,cfg,*,return_trace=False):
    """Refit from accumulated paid audits; no outcome reads in refresh decisions.

    Equal *maximum* label budget, actual paid draws and distinct labels reported.
    Threshold refresh is preventive (proposed movement), then proposal recomputed.
    It is a heuristic, not a certified safety gate. Budget exhaustion freezes the
    verifier; it does not stop optimization or make uncharged oracle queries.
    """
    if cfg.optimizer not in ('soft','bon') or cfg.controller not in ('fixed','kl','maxlog','geometry'):
        raise ValueError('unknown optimizer/controller')
    if min(cfg.refresh_interval,cfg.rounds,cfg.audit_per_refresh,cfg.total_audit_budget)<=0 or cfg.eta<0 or cfg.ridge<=0 or cfg.score_scale<=0:
        raise ValueError('invalid nonpositive experiment parameter')
    if cfg.representation not in ('public','all'): raise ValueError('unknown representation')
    if cfg.best_of_n<1 or cfg.threshold<0: raise ValueError('invalid N/threshold')
    df=df.reset_index(drop=True)
    cols=sorted(c for c in df if c.startswith('f::') and (cfg.representation=='all' or c=='f::public_score'))
    if not cols: raise ValueError('no verifier features')
    X=np.column_stack([np.ones(len(df)),df[cols].to_numpy(float)])
    p=task_policy(df,df.base_logprob.to_numpy(float)); initial=p.copy()
    rng=np.random.default_rng(cfg.seed); audit=[]; theta=None; history=[]; traces=[]
    anchor=p.copy(); age=0; refreshes=0
    # y is used ONLY inside the paid refit and post-decision outcome section.
    y=df.trusted_score.to_numpy(float)
    def refit():
        nonlocal theta,anchor,age,refreshes
        n=min(cfg.audit_per_refresh,cfg.total_audit_budget-len(audit))
        if n<=0: return False
        idx=rng.choice(len(df),size=n,replace=True,p=p)
        audit.extend(idx.tolist())
        theta=fit_verifier(X[audit],y[audit],cfg.ridge)
        anchor=p.copy(); age=0; refreshes+=1
        return True
    for t in range(cfg.rounds):
        refreshed=False; reason='none'; trigger=False
        if theta is None or (cfg.controller=='fixed' and age>=cfg.refresh_interval):
            refreshed=refit(); reason='initial' if t==0 else 'cadence'
        score=(X@theta)*cfg.score_scale
        proposed=propose(df,p,score,cfg)
        pre=movement(anchor,proposed,X,score)
        if cfg.controller!='fixed' and age>0:
            state=MovementState(age,cumulative_kl=pre['kl_from_refresh'],
                max_log_density_ratio=pre['max_log_density_ratio'],
                restricted_geometry=pre['restricted_geometry'],proxy_margin=pre['proxy_margin']/cfg.score_scale)
            if cfg.controller=='geometry': trigger=GeometryMarginController(cfg.threshold).should_refresh(state)
            else:
                metric='cumulative_kl' if cfg.controller=='kl' else 'max_log_density_ratio'
                trigger=ScalarThreshold(metric,cfg.threshold).should_refresh(state)
            if trigger:
                refreshed=refit(); reason='threshold' if refreshed else 'budget_exhausted'
                if refreshed:
                    score=(X@theta)*cfg.score_scale
                    proposed=propose(df,p,score,cfg)
        accepted=movement(anchor,proposed,X,score)
        old=p.copy(); p=proposed; age+=1
        kl_step,_=divergence(p,old)
        # Outcome-only quantities below cannot affect the already selected p.
        true=float(p@y); baseline=float(initial@y)
        row={'round':t+1,'true_reward':true,'initial_true_reward':baseline,
             'true_gain_step':float((p-old)@y),'proxy_reward':float(p@score),
             'proxy_gain_step':float((p-old)@score),
             'below_initial':int(true<baseline-cfg.failure_tolerance),
             'harmful_step':int((p-old)@y < -cfg.failure_tolerance),
             'kl_from_previous':kl_step,**accepted,
             **{'pre_'+k:v for k,v in pre.items()},
             'oracle_error_contrast':float((p-anchor)@(score/cfg.score_scale-y)),
             'oracle_true_gain_from_refresh':float((p-anchor)@y),
             'refresh':int(refreshed),'refresh_reason':reason,'trigger':int(trigger),
             'zero_mass_candidates':int((p==0).sum()),
             'effective_support':float(1/(p@p)),
             'refresh_count':refreshes,'audit_labels':len(audit),'unique_audit_labels':len(set(audit)),
             'rounds_since_refresh':age,'eta_times_age':cfg.eta*age if cfg.optimizer=='soft' else np.nan,
             **cfg.__dict__}
        history.append(row)
        if return_trace: traces.append({'policy':p.copy(),'audit_indices':tuple(audit)})
    out=pd.DataFrame(history)
    return (out,traces) if return_trace else out


def main():
    p=argparse.ArgumentParser(); p.add_argument('candidate_bank',type=Path)
    for name,field in Config.__dataclass_fields__.items():
        value=field.default; p.add_argument('--'+name.replace('_','-'),type=type(value),default=value)
    p.add_argument('--output',type=Path,default=Path('candidate_bank_run.csv'))
    a=p.parse_args(); cfg=Config(**{k:getattr(a,k) for k in Config.__dataclass_fields__})
    out=run(load_jsonl(a.candidate_bank),cfg); out.to_csv(a.output,index=False)
    print(out.tail().to_string(index=False))

if __name__=='__main__': main()
