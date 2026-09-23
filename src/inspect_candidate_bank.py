"""Verify deterministic scoring and report bank support/format diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.finite_code_tasks import Task, score


def main():
    p=argparse.ArgumentParser();p.add_argument('bank',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    meta=json.loads(a.bank.with_suffix('.manifest.json').read_text())
    if not meta['complete'] or hashlib.sha256(a.bank.read_bytes()).hexdigest()!=meta['bank_sha256']:
        raise ValueError('bank is incomplete or modified')
    rows=[json.loads(l) for l in a.bank.read_text().splitlines()]
    known={t['task_id']:Task(**t) for t in meta['tasks']}
    for row in rows:
        rescored=score(row['source'],known[row['task_id']])
        for col in ['public_score','trusted_score','exhaustive_score','features']:
            if row[col]!=rescored[col]:raise ValueError('inconsistent deterministic evaluation')
        row['valid']=row['features']['valid']
        if not np.isfinite(row['sampling_logprob']):raise ValueError('invalid sampling probability')
    df=pd.DataFrame(rows); out=[]
    for task,g in df.groupby('task_id'):
        frontier=g[g.public_score==g.public_score.max()]
        out.append({'task_id':task,'candidates':len(g),'distinct_sources':g.source_sha256.nunique(),
            'valid_fraction':g.valid.mean(),'initial_public':g.public_score.mean(),
            'initial_hidden':g.trusted_score.mean(),'fully_correct_candidates':int((g.trusted_score==1).sum()),
            'public_perfect_hidden_imperfect':int(((g.public_score==1)&(g.trusted_score<1)).sum()),
            'public_frontier_count':len(frontier),'public_frontier_hidden':frontier.trusted_score.mean(),
            'trusted_score_std':g.trusted_score.std()})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result=pd.DataFrame(out);result.to_csv(a.output,index=False);print(result.to_string(index=False))

if __name__=='__main__':main()
