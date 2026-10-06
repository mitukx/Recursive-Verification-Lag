"""Offline falsification check for observable Lipschitz verifier-error geometry.

Uses trusted outcomes ONLY for retrospective evaluation. Nothing produced here
is a valid pre-update bound or an admissible metric calibration for the same
tasks. Identical-source occurrences are collapsed before comparisons.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.candidate_bank_experiment import load_jsonl


def diagnose(bank_path):
    bank=load_jsonl(bank_path)
    raw=[json.loads(line) for line in bank_path.read_text().splitlines() if line.strip()]
    sources=pd.DataFrame([{'task_id':str(r['task_id']),
                           'candidate_id':str(r['candidate_id']),
                           'source_sha256':str(r['source_sha256'])} for r in raw])
    bank['task_id']=bank.task_id.astype(str)
    bank['candidate_id']=bank.candidate_id.astype(str)
    bank=bank.merge(sources,on=['task_id','candidate_id'],validate='one_to_one')
    out=[]
    for task,frame in bank.groupby('task_id',sort=True):
        # One deterministic reward and observable representation per text.
        for _,dupes in frame.groupby('source_sha256'):
            if dupes.trusted_score.max()-dupes.trusted_score.min()>1e-12:
                raise ValueError('source has inconsistent trusted rewards')
        frame=frame.drop_duplicates('source_sha256')
        e=(frame.trusted_score-frame['f::public_score']).to_numpy(float)
        for representation in ('public','all'):
            names=sorted(c for c in frame if c.startswith('f::') and
                         (representation=='all' or c=='f::public_score'))
            X=frame[names].to_numpy(float)
            dist=np.linalg.norm(X[:,None,:]-X[None,:,:],axis=-1)
            difference=np.abs(e[:,None]-e[None,:])
            upper=np.triu(np.ones(dist.shape,bool),k=1)
            collisions=upper & (dist==0)
            conflicts=collisions & (difference>1e-10)
            positive=upper & (dist>0)
            slope=float(np.max(difference[positive]/dist[positive])) if positive.any() else 0.
            out.append(dict(task_id=task,representation=representation,
                            sources=len(frame),zero_distance_pairs=int(collisions.sum()),
                            contradictory_pairs=int(conflicts.sum()),
                            worst_collision_residual_gap=float(difference[conflicts].max()) if conflicts.any() else 0.,
                            minimum_lipschitz_constant=float('inf') if conflicts.any() else slope))
    return pd.DataFrame(out)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('bank',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();result=diagnose(args.bank)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(result.groupby('representation').agg(tasks=('task_id','size'),
          impossible_tasks=('contradictory_pairs',lambda x:int((x>0).sum())),
          contradictory_pairs=('contradictory_pairs','sum'),
          largest_collision_gap=('worst_collision_residual_gap','max')).to_string())


if __name__=='__main__': main()
