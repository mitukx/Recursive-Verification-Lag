"""Exploratory held-group transfer, onset-only (no post-failure feature maxima).

Thresholds and directions are fitted on training groups only. Pilot LOTO is
cross-validation, not a sealed final test. Single-class folds are unidentified.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

METRICS={'eta_times_age':'eta_times_age','KL':'kl_from_refresh',
         'max_log_ratio':'max_log_density_ratio','restricted_geometry':'restricted_geometry',
         'geometry_over_margin2':'geometry_over_margin2'}


def balanced_accuracy(y,pred):
    y=np.asarray(y,bool); pred=np.asarray(pred,bool)
    if not y.any() or y.all(): return np.nan
    return float((pred[y].mean()+(~pred[~y]).mean())/2)


def fit_threshold(x,y):
    x=np.asarray(x,float); y=np.asarray(y,bool)
    ok=np.isfinite(x); x=x[ok]; y=y[ok]
    if not len(x) or not y.any() or y.all(): return None
    order=np.argsort(x,kind='stable'); x=x[order]; y=y[order]
    # A split only between distinct x values; thresholds include constant rules.
    ends=np.r_[0,np.flatnonzero(np.diff(x)!=0)+1,len(x)]
    pos=np.r_[0,np.cumsum(y)]; neg=np.r_[0,np.cumsum(~y)]
    scores=((pos[-1]-pos[ends])/pos[-1]+neg[ends]/neg[-1])/2
    both=np.r_[scores,1-scores]; k=int(np.argmax(both)); direction=1 if k<len(ends) else -1
    cut=ends[k%len(ends)]
    if cut==0: t=float('-inf')
    elif cut==len(x): t=float('inf')
    else: t=float(x[cut-1]+(x[cut]-x[cut-1])/2)
    return t,direction


def task_bootstrap(df, column, seed=123, draws=1000):
    values=df.groupby('task_id')[column].mean().to_numpy(float)
    rng=np.random.default_rng(seed)
    means=values[rng.integers(len(values),size=(draws,len(values)))].mean(axis=1)
    return float(values.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))


def transfer(rows):
    # All state features were computed before reading the outcome of that update.
    fixed=rows[rows.controller=='fixed'].copy()
    prev=fixed.groupby('run_id').below_initial.cumsum()-fixed.below_initial
    fixed=fixed[prev==0].copy()
    fixed['geometry_over_margin2']=fixed.restricted_geometry/np.maximum(np.abs(fixed.proxy_margin/fixed.score_scale),1e-8)**2
    outputs=[]
    for axis in ['task_id','optimizer','representation']:
        for metric,col in METRICS.items():
            for group in sorted(fixed[axis].unique()):
                tr=fixed[fixed[axis]!=group]; te=fixed[fixed[axis]==group]
                tr=tr[np.isfinite(tr[col])]; te=te[np.isfinite(te[col])]
                fit=fit_threshold(tr[col],tr.below_initial)
                result={'axis':axis,'held_group':group,'metric':metric,'train_n':len(tr),'test_n':len(te),
                        'test_failures':int(te.below_initial.sum())}
                if fit is None or te.empty or te.below_initial.nunique()<2:
                    result.update(status='unidentified_single_class_or_missing',balanced_accuracy=np.nan)
                else:
                    threshold,direction=fit; pred=te[col]>=threshold if direction==1 else te[col]<threshold
                    result.update(status='ok',threshold=threshold,direction=direction,
                        balanced_accuracy=balanced_accuracy(te.below_initial,pred))
                outputs.append(result)
    return pd.DataFrame(outputs)


def main():
    p=argparse.ArgumentParser(); p.add_argument('results',type=Path); a=p.parse_args()
    manifest=json.loads((a.results/'manifest.json').read_text())
    runs=pd.read_csv(a.results/'runs.csv'); rows=pd.read_csv(a.results/'trajectories.csv.gz')
    transfer(rows).to_csv(a.results/'transfer.csv',index=False)
    groups=['controller','refresh_interval','threshold','total_audit_budget','representation']
    summary=[]
    for factors,g in runs.groupby(groups):
        record=dict(zip(groups,factors))
        for column in ['ever_below_initial','final_gain','audit_labels','unique_audit_labels']:
            mean,low,high=task_bootstrap(g,column)
            record.update({column:mean,column+'_lo':low,column+'_hi':high})
        summary.append(record)
    pd.DataFrame(summary).to_csv(a.results/'controller_summary.csv',index=False)
    first=rows[rows.below_initial.eq(1)].sort_values('round').groupby('run_id').first()
    audit={'runs':len(runs),'tasks':runs.task_id.nunique(),
        'collapse_runs':int(runs.ever_below_initial.sum()),
        'first_round_crossings':int((first.index.map(runs.set_index('run_id').first_failure_round)==1).sum()),
        'first_crossing_with_stale_age_gt_1':int((first.rounds_since_refresh>1).sum()),
        'collapse_fraction':float(runs.ever_below_initial.mean()),
        'decline_gt_1pct_runs':int(((runs.min_true_reward-runs.initial_true_reward)<-.01).sum()),
        'decline_gt_5pct_runs':int(((runs.min_true_reward-runs.initial_true_reward)<-.05).sum()),
        'worst_baseline_change':float((runs.min_true_reward-runs.initial_true_reward).min()),
        'mean_final_gain':float(runs.final_gain.mean()),
        'evidence':manifest['evidence'],
        'scope':'Exploratory analysis; bootstrap unit is task; budget caps not matched actual cost',
        'prediction_target':'first baseline crossing at the proposed update; no rows after prior collapse; fixed cadence only'}
    (a.results/'summary.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(audit,indent=2))

if __name__=='__main__': main()
