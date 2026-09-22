"""Post-hoc mechanism diagnostic on a shared initially-safe cohort.

Cadences have identical initial audits, optimizer, features and first update.
Condition on that *common* first outcome; do not select cohorts separately per
cadence. This is an offline diagnostic, not an observable deployment filter and
not a cost-matched comparison (fresh cadence can spend more paid draws).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.analyze_candidate_sweep import task_bootstrap

KEYS=['task_id','optimizer','eta','best_of_n','representation','seed','total_audit_budget']


def compare(runs,rows):
    first=rows[(rows.controller=='fixed')&(rows['round']==1)&(rows.total_audit_budget==96)].copy()
    for _,g in first.groupby(KEYS):
        for col in ['true_reward','initial_true_reward','proxy_reward','audit_labels']:
            if not np.allclose(g[col],g[col].iloc[0],rtol=0,atol=1e-12):
                raise ValueError('first update not shared across cadences')
    common=first[first.refresh_interval==1][KEYS+['below_initial']]
    keep=common[common.below_initial==0][KEYS]
    selected=runs[(runs.controller=='fixed')&(runs.total_audit_budget==96)].merge(keep,on=KEYS)
    fresh=selected[selected.refresh_interval==1]; pairs=[]; summary=[]
    for cadence in [2,4,12]:
        stale=selected[selected.refresh_interval==cadence]
        paired=fresh.merge(stale,on=KEYS,suffixes=('_fresh','_stale'))
        if paired.empty:continue
        paired['stale_cadence']=cadence
        paired['failure_difference']=paired.ever_below_initial_stale-paired.ever_below_initial_fresh
        paired['gain_difference']=paired.final_gain_stale-paired.final_gain_fresh
        paired['cost_difference']=paired.audit_labels_stale-paired.audit_labels_fresh
        rec={'stale_cadence':cadence,'comparisons':len(paired),
            'fresh_failure':paired.groupby('task_id').ever_below_initial_fresh.mean().mean(),
            'stale_failure':paired.groupby('task_id').ever_below_initial_stale.mean().mean(),
            'fresh_pair_rate':paired.ever_below_initial_fresh.mean(),
            'stale_pair_rate':paired.ever_below_initial_stale.mean(),
            'fresh_safe_stale_fails':int(((paired.ever_below_initial_fresh==0)&(paired.ever_below_initial_stale==1)).sum()),
            'fresh_fails_stale_safe':int(((paired.ever_below_initial_fresh==1)&(paired.ever_below_initial_stale==0)).sum()),
            'fresh_draws':paired.audit_labels_fresh.mean(),'stale_draws':paired.audit_labels_stale.mean()}
        for metric in ['failure','gain','cost']:
            mean,lo,hi=task_bootstrap(paired,metric+'_difference')
            rec.update({metric+'_difference':mean,metric+'_lo':lo,metric+'_hi':hi})
        pairs.append(paired);summary.append(rec)
    return pd.concat(pairs,ignore_index=True),pd.DataFrame(summary)


def main():
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);a=p.parse_args()
    paired,summary=compare(pd.read_csv(a.results/'runs.csv'),pd.read_csv(a.results/'trajectories.csv.gz'))
    paired.to_csv(a.results/'initial_safe_pairs.csv',index=False)
    summary.to_csv(a.results/'initial_safe_summary.csv',index=False)
    print(summary.to_string(index=False))

if __name__=='__main__':main()
