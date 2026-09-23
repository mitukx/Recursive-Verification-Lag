"""Check composed soft-update identity against recorded endpoint rewards."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
from src.candidate_bank_experiment import load_jsonl,fit_verifier,softmax

def main():
    ap=argparse.ArgumentParser();ap.add_argument('bank',type=Path);ap.add_argument('results',type=Path);a=ap.parse_args()
    bank=load_jsonl(a.bank);tasks={k:g.reset_index(drop=True) for k,g in bank.groupby('task_id')};errors=[]
    with gzip.open(a.results/'runs.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['optimizer']!='soft' or r['acquisition']!='stream':continue
            df=tasks[r['task_id']];cols=sorted(c for c in df if c.startswith('f::') and (r['representation']=='all' or c=='f::public_score'))
            X=np.column_stack([np.ones(len(df)),df[cols].to_numpy(float)]);y=df.trusted_score.to_numpy(float)
            logits=df.base_logprob.to_numpy(float).copy();starts=r['audit_rounds'];ends=starts[1:]+[13]
            for k,(start,end) in enumerate(zip(starts,ends)):
                idx=r['audit_indices'][:2*(k+1)];theta=fit_verifier(X[idx],y[idx],1e-4)
                logits+=r['strength']*(end-start)*(X@theta)
            errors.append(abs(float(softmax(logits)@y)-r['rewards'][-1]))
    out={'soft_stream_runs':len(errors),'max_absolute_endpoint_reward_error':max(errors),
         'interpretation':'Standard composed score-exposure identity, not a new theorem or observable error predictor.'}
    assert max(errors)<1e-10
    (a.results/'exposure_identity.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
