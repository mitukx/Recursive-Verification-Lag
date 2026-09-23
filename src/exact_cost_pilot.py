"""Exploratory refresh timing comparison with exactly 32 paid draws per run."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.candidate_bank_experiment import Config, load_jsonl, run
from src.analyze_candidate_sweep import task_bootstrap

DESIGNS = {
    'early': ('fixed', .5, (1,2,3,4)),
    'uniform': ('fixed', .5, (1,4,7,10)),
    'late': ('fixed', .5, (1,10,11,12)),
    'kl_01': ('kl', .1, None),
    'kl_05': ('kl', .5, None),
    'maxlog_05': ('maxlog', .5, None),
    'maxlog_15': ('maxlog', 1.5, None),
    'geometry_10': ('geometry', 10., None),
    'geometry_100': ('geometry', 100., None),
}


def main():
    a=argparse.ArgumentParser();a.add_argument('bank',type=Path);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--record-coordinates',action='store_true')
    a.add_argument('--evidence-status',default='exploratory on already inspected development bank; not preregistered confirmatory evidence')
    args=a.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest={'bank_sha256':hashlib.sha256(args.bank.read_bytes()).hexdigest(),
      'designs':DESIGNS,'rounds':12,'draws_per_audit':8,'audit_events':4,'seeds':list(range(5)),
      'status':args.evidence_status,'record_coordinates':args.record_coordinates,
      'source_sha256':{n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['src/exact_cost_pilot.py','src/candidate_bank_experiment.py']}}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    bank=load_jsonl(args.bank);records=[]
    # Write every completed run immediately; never keep the sole copy in memory.
    with (args.output/'runs.jsonl').open('w') as f:
        for task,df in bank.groupby('task_id',sort=True):
            for optimizer,strengths in [('soft',[.25,1.,4.]),('bon',[2,4,16])]:
                for strength in strengths:
                    for representation in ['public','all']:
                        for seed in range(5):
                            initial=None
                            for name,(controller,threshold,schedule) in DESIGNS.items():
                                cfg=Config(optimizer=optimizer,eta=strength if optimizer=='soft' else 1.,
                                    best_of_n=int(strength) if optimizer=='bon' else 4,representation=representation,
                                    seed=seed,controller=controller,threshold=threshold,rounds=12,
                                    audit_per_refresh=8,total_audit_budget=32)
                                h,tr=run(df,cfg,return_trace=True,audit_schedule=schedule,exact_audit_events=4)
                                if initial is None:initial=tr[0]
                                assert initial['audit_indices']==tr[0]['audit_indices']
                                np.testing.assert_allclose(initial['policy'],tr[0]['policy'],rtol=0,atol=0)
                                rec={'task_id':task,'optimizer':optimizer,'strength':strength,'representation':representation,
                                    'seed':seed,'design':name,'failure':int(h.below_initial.any()),
                                    'initial_failure':int(h.iloc[0].below_initial),
                                    'gain':float(h.iloc[-1].true_reward-h.iloc[-1].initial_true_reward),
                                    'cost':int(h.iloc[-1].audit_labels),'unique':int(h.iloc[-1].unique_audit_labels),
                                    'deadline_events':int((h.refresh_reason=='deadline').sum()),
                                    'audit_rounds':h.loc[h.refresh==1,'round'].tolist(),
                                    'audit_indices':list(tr[-1]['audit_indices']),
                                    'rewards':h.true_reward.tolist()}
                                if args.record_coordinates:
                                    rec['coordinates']={c:[None if not np.isfinite(v) else float(v) for v in h[c]] for c in
                                        ['kl_from_refresh','max_log_density_ratio','restricted_geometry','proxy_margin','eta_times_age']}
                                    rec['below_initial']=h.below_initial.tolist()
                                f.write(json.dumps(rec)+'\n');f.flush();records.append(rec)
            print('completed',task,flush=True)
    rows=pd.DataFrame(records);summaries=[]
    keys=['task_id','optimizer','strength','representation','seed']
    baseline=rows[rows.design=='uniform']
    for name,g in rows.groupby('design'):
        pair=g.merge(baseline,on=keys,suffixes=('','_uniform'),validate='one_to_one')
        for cohort in ['all','initially_safe_posthoc']:
            p=pair if cohort=='all' else pair[pair.initial_failure==0]
            rec={'design':name,'cohort':cohort,'pairs':len(p),'tasks':p.task_id.nunique(),
                 'paid_draws':p.cost.mean(),'unique_labels':p.unique.mean(),'deadline_events':p.deadline_events.mean()}
            for metric in ['failure','gain','unique']:
                p=p.copy();p['difference']=p[metric]-p[metric+'_uniform']
                mean,lo,hi=task_bootstrap(p,'difference')
                rec.update({metric+'_difference':mean,metric+'_lo':lo,metric+'_hi':hi})
                rec[metric+'_task_mean']=p.groupby('task_id')[metric].mean().mean()
            summaries.append(rec)
    pd.DataFrame(summaries).to_csv(args.output/'summary.csv',index=False)
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__=='__main__':main()
