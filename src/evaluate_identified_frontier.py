"""Measure source-aware, worst-case reward identification on observed banks.

Diagnostic only: computes the interval *after* each paid audit and update.
It is not a prospective intervention, and trusted rewards never enter interval
computation except through the audited source identities.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.candidate_bank_experiment import Config,load_jsonl,run,task_policy
from src.identified_gain import identified_gain


def evaluate(bank_path,output):
    output.mkdir(parents=True,exist_ok=False)
    raw=[json.loads(line) for line in bank_path.read_text().splitlines()]
    source_by_id={r['candidate_id']:r['source_sha256'] for r in raw}
    bank=load_jsonl(bank_path);records=[]
    schedules={'early':(1,2,3,4),'uniform':(1,4,7,10),'late':(1,10,11,12)}
    for task,df in bank.groupby('task_id',sort=True):
        df=df.reset_index(drop=True)
        sources=[source_by_id[c] for c in df.candidate_id]
        y=df.trusted_score.to_numpy(float)
        by_source={}
        for key,reward in zip(sources,y):
            if key in by_source and by_source[key]!=reward:
                raise ValueError('one source has inconsistent trusted scores')
            by_source[key]=reward
        q=task_policy(df,df.base_logprob.to_numpy(float))
        for optimizer,strength in [('soft',1.),('bon',4.)]:
            for representation in ['public','all']:
                for seed in range(5):
                    for design,schedule in schedules.items():
                        cfg=Config(optimizer=optimizer,
                            eta=strength if optimizer=='soft' else 1.,
                            best_of_n=int(strength) if optimizer=='bon' else 4,
                            representation=representation,seed=seed,
                            controller='fixed',rounds=12,audit_per_refresh=8,
                            total_audit_budget=32)
                        history,trace=run(df,cfg,return_trace=True,
                            audit_schedule=schedule,exact_audit_events=4)
                        for row,point in zip(history.itertuples(index=False),trace):
                            revealed={sources[i]:y[i] for i in point['audit_indices']}
                            lo,hi=identified_gain(point['policy'],q,sources,revealed)
                            truth=float((point['policy']-q)@y)
                            if truth<lo-1e-9 or truth>hi+1e-9:
                                raise AssertionError('actual gain escapes sharp interval')
                            records.append({'task_id':task,'optimizer':optimizer,
                                'representation':representation,'seed':seed,
                                'design':design,'round':row.round,
                                'audited_sources':len(revealed),
                                'gain_lower':lo,'gain_upper':hi,
                                'interval_width':hi-lo,'true_gain_evaluation_only':truth,
                                'certified_nonnegative':int(lo>=-1e-12),
                                'certified_negative':int(hi< -1e-12),
                                'ambiguous':int(lo< -1e-12 and hi>=-1e-12)})
    out=pd.DataFrame(records)
    out.to_csv(output/'frontier_rows.csv.gz',index=False)
    summary=out.groupby('design')[['certified_nonnegative','certified_negative','ambiguous',
        'interval_width','audited_sources']].mean().reset_index()
    summary['runs']=len(out)//(len(summary)*12)
    summary.to_csv(output/'summary.csv',index=False)
    bytask=out.groupby(['task_id','design'])[['certified_nonnegative',
        'certified_negative','ambiguous','interval_width']].mean().reset_index()
    bytask.to_csv(output/'by_task.csv',index=False)
    (output/'manifest.json').write_text(json.dumps({
        'bank_sha256':hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        'scope':'post-update identification; exploratory, not a controller',
        'assumptions':'per-source deterministic reward in [0,1], no constraints across distinct sources',
        'designs':schedules,'optimizers':{'soft':[1.],'bon':[4]},
        'representations':['public','all'],'audit_seeds':list(range(5)),
        'rows':len(out),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print(summary.to_string(index=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('bank',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    evaluate(a.bank,a.output)


if __name__=='__main__':main()
