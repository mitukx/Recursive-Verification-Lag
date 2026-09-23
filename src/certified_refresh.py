"""Finite-bank source-audited, safety-certified adaptive refresh.

The accepted policy is guaranteed no worse than its initial policy *on this
finite bank* when source identities share reward and r in [0,1]. This is an
abstaining controller; it does not guarantee a positive improvement or that
the fixed budget suffices for certification. Trusted labels enter the decision
only when charged as a source audit. Results must report abstention and cost.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.candidate_bank_experiment import (load_jsonl,task_policy,fit_verifier,propose,Config)
from src.identified_gain import identified_gain
from src.source_audit import SourceAudit


def certified_run(df,sources,cfg,*,budget=6,initial_audits=2):
    if not 1<=initial_audits<=budget<=len(set(sources)):
        raise ValueError('invalid distinct source budget')
    df=df.reset_index(drop=True);sources=np.asarray(sources)
    if len(sources)!=len(df):raise ValueError('misaligned source identities')
    y=df.trusted_score.to_numpy(float)
    names=sorted(c for c in df if c.startswith('f::') and (cfg.representation=='all' or c=='f::public_score'))
    X=np.column_stack([np.ones(len(df)),df[names].to_numpy(float)])
    p=task_policy(df,df.base_logprob.to_numpy(float));initial=p.copy()
    sampler=SourceAudit(sources,cfg.seed,'stream')
    acquired=list(sampler(p,initial_audits,tuple()))
    revealed={sources[i]:float(y[i]) for i in acquired}
    records=[]
    for t in range(cfg.rounds):
        new_this_round=initial_audits if t==0 else 0
        while True:
            theta=fit_verifier(X[acquired],y[acquired],cfg.ridge)
            score=(X@theta)*cfg.score_scale
            candidate=propose(df,p,score,cfg)
            lo,hi=identified_gain(candidate,initial,sources,revealed)
            if lo>=-1e-12:
                accepted=True;p=candidate;break
            if len(acquired)==budget:
                accepted=False;break
            # Audit the unaudited source with largest possible contribution to
            # uncertainty, |sum_{i in group}(candidate-initial)_i|. Tie by
            # lexical source ID, independently of hidden reward.
            remaining=[key for key in np.unique(sources) if key not in revealed]
            target=min(remaining,key=lambda key:(
                -abs(float((candidate-initial)[sources==key].sum())),str(key)))
            idx=int(np.flatnonzero(sources==target)[0]);acquired.append(idx)
            revealed[target]=float(y[idx]);new_this_round+=1
        outcome=float((p-initial)@y)
        cert_lo,cert_hi=identified_gain(p,initial,sources,revealed)
        if cert_lo< -1e-9 or outcome< -1e-9:
            raise AssertionError('safety certificate violated')
        records.append({'round':t+1,'accepted':int(accepted),
            'new_source_labels':new_this_round,'paid_source_labels':len(acquired),
            'candidate_lower':lo,'candidate_upper':hi,
            'certified_lower':cert_lo,
            'gain_evaluation_only':outcome,
            'true_reward_evaluation_only':float(p@y)})
    return pd.DataFrame(records)


def evaluate(bank_path,output):
    output.mkdir(parents=True,exist_ok=False)
    raw=[json.loads(s) for s in bank_path.read_text().splitlines()]
    sources={r['candidate_id']:r['source_sha256'] for r in raw}
    bank=load_jsonl(bank_path);all_rows=[]
    for task,df in bank.groupby('task_id',sort=True):
        ids=[sources[c] for c in df.candidate_id]
        # Dataset integrity is checked offline, before entering the controller.
        # This is not an observable decision feature or a free audit label.
        true_y=df.trusted_score.to_numpy(float);id_array=np.asarray(ids)
        for key in np.unique(id_array):
            if np.ptp(true_y[id_array==key])>1e-12:
                raise ValueError('same source has inconsistent trusted scores')
        for optimizer,strength in [('soft',1.),('bon',4.)]:
            for representation in ['public','all']:
                for seed in range(5):
                    cfg=Config(optimizer=optimizer,eta=strength if optimizer=='soft' else 1.,
                        best_of_n=int(strength) if optimizer=='bon' else 4,
                        representation=representation,seed=seed,rounds=12)
                    h=certified_run(df,ids,cfg)
                    h['task_id']=task;h['optimizer']=optimizer;h['representation']=representation
                    h['seed']=seed;all_rows.append(h)
    rows=pd.concat(all_rows,ignore_index=True)
    rows.to_csv(output/'runs.csv.gz',index=False)
    final=rows[rows['round']==12]
    summary=final.groupby('task_id').agg(mean_gain=('gain_evaluation_only','mean'),
        mean_paid_sources=('paid_source_labels','mean'),
        accepted_last=('accepted','mean'),
        nonzero_gain=('gain_evaluation_only',lambda s:float((s>1e-10).mean())))
    summary.to_csv(output/'by_task.csv')
    print('runs',len(final),'tasks',final.task_id.nunique(),
          'failure_rate',float((rows.gain_evaluation_only< -1e-10).mean()),
          'mean_gain',float(final.gain_evaluation_only.mean()),
          'mean_paid',float(final.paid_source_labels.mean()),
          'accept_rate',float(rows.accepted.mean()))
    print(summary.to_string())


def main():
    p=argparse.ArgumentParser();p.add_argument('bank',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    evaluate(a.bank,a.output)


if __name__=='__main__':main()
