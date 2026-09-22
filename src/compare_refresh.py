"""Exploratory leave-one-task-out schedule selection at matched budget caps.

Actual cost is always reported; a common cap alone does not prove cost dominance.
An optional actual-draw ceiling is imposed on *training means* only, then its
held-task violations are measured (not hidden by retrospective filtering).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.analyze_candidate_sweep import task_bootstrap

FACTORS=['optimizer','eta','best_of_n','representation','total_audit_budget']
SCHEDULE=['controller','refresh_interval','threshold']


def crossfit(runs, cost_ceiling=None):
    records=[]
    for task in sorted(runs.task_id.unique()):
        train=runs[runs.task_id!=task]; test=runs[runs.task_id==task]
        for factors,g in train.groupby(FACTORS):
            for family in ['fixed','adaptive']:
                part=g[g.controller.eq('fixed') if family=='fixed' else ~g.controller.eq('fixed')]
                stats=part.groupby(SCHEDULE)[['ever_below_initial','final_gain','audit_labels']].mean().reset_index()
                if cost_ceiling is not None: stats=stats[stats.audit_labels<=cost_ceiling]
                if stats.empty: continue
                # Preregistered lexicographic objective; no held-task outcomes used.
                winner=stats.sort_values(['ever_below_initial','final_gain','audit_labels',*SCHEDULE],ascending=[True,False,True,True,True,True]).iloc[0]
                selected=test.copy()
                for key,val in zip(FACTORS,factors): selected=selected[selected[key]==val]
                for key in SCHEDULE: selected=selected[selected[key]==winner[key]]
                if selected.empty: raise ValueError('selected schedule absent in test')
                records.append({'task_id':task,'method':family,**dict(zip(FACTORS,factors)),
                    **{k:winner[k] for k in SCHEDULE},
                    'failure':selected.ever_below_initial.mean(),'gain':selected.final_gain.mean(),
                    'cost':selected.audit_labels.mean(),
                    'cost_ceiling':np.nan if cost_ceiling is None else cost_ceiling,'test_cost_exceeds_ceiling':int(cost_ceiling is not None and selected.audit_labels.mean()>cost_ceiling)})
    return pd.DataFrame(records)


def main():
    p=argparse.ArgumentParser(); p.add_argument('results',type=Path); a=p.parse_args()
    runs=pd.read_csv(a.results/'runs.csv'); all_results=[]; summary=[]
    for ceiling in [None,16.,24.,32.,48.,64.,96.]:
        out=crossfit(runs,ceiling)
        if out.empty: continue
        all_results.append(out)
        for budget,g in out.groupby('total_audit_budget'):
            key=['task_id',*FACTORS]
            fixed=g[g.method=='fixed']; adaptive=g[g.method=='adaptive']
            paired=fixed.merge(adaptive,on=key,suffixes=('_fixed','_adaptive'))
            if paired.empty: continue
            rec={'training_cost_ceiling':ceiling,'budget_cap':budget,'paired_cells':len(paired)}
            for m in ['failure','gain','cost']:
                paired[m+'_difference']=paired[m+'_adaptive']-paired[m+'_fixed']
                mean,lo,hi=task_bootstrap(paired,m+'_difference')
                rec.update({m+'_difference':mean,m+'_difference_lo':lo,m+'_difference_hi':hi})
            rec['adaptive_mean_cost']=paired.cost_adaptive.mean(); rec['fixed_mean_cost']=paired.cost_fixed.mean()
            rec['cost_violation_fraction']=paired.test_cost_exceeds_ceiling_adaptive.mean()
            summary.append(rec)
    pd.concat(all_results).to_csv(a.results/'crossfit_refresh.csv',index=False)
    pd.DataFrame(summary).to_csv(a.results/'crossfit_refresh_summary.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__': main()
