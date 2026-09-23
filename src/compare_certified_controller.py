"""Matched-config exploratory cost/progress comparison to six-source schedules."""
import argparse
import json
from pathlib import Path
import pandas as pd
from src.analyze_candidate_sweep import task_bootstrap


def compare(certified_root,source_root,output):
    certified=pd.read_csv(certified_root/'runs.csv.gz')
    final=certified[certified['round']==12].copy()
    final['cert_failure']=certified.groupby(['task_id','optimizer','representation','seed'])[
        'gain_evaluation_only'].transform(lambda s:(s< -1e-10).any()).loc[final.index].astype(int)
    old=pd.read_json(source_root/'runs.jsonl.gz',lines=True)
    old=old[(old.acquisition=='stream')&(
        ((old.optimizer=='soft')&(old.strength==1.))|
        ((old.optimizer=='bon')&(old.strength==4)))].copy()
    keys=['task_id','optimizer','representation','seed']
    out=[]
    for design,group in old.groupby('design'):
        if design not in ['early','uniform','late']:continue
        joined=final.merge(group,on=keys,validate='one_to_one')
        if len(joined)!=len(final):raise ValueError('incomplete paired runs')
        joined['gain_delta']=joined.gain_evaluation_only-joined.gain
        joined['failure_delta']=joined.cert_failure-joined.failure
        joined['cost_delta']=joined.paid_source_labels-joined.unique_sources
        record={'design':design,'pairs':len(joined),'tasks':joined.task_id.nunique(),
            'certified_mean_gain':joined.gain_evaluation_only.mean(),
            'fixed_mean_gain':joined.gain.mean(),
            'certified_failure_rate':joined.cert_failure.mean(),
            'fixed_failure_rate':joined.failure.mean(),
            'certified_mean_labels':joined.paid_source_labels.mean(),
            'fixed_labels':joined.unique_sources.mean()}
        for col in ['gain_delta','failure_delta','cost_delta']:
            mean,lo,hi=task_bootstrap(joined,col)
            record[col]=mean;record[col+'_lo']=lo;record[col+'_hi']=hi
        out.append(record)
    summary=pd.DataFrame(out);summary.to_csv(output,index=False)
    print(summary.to_string(index=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('certified',type=Path)
    p.add_argument('source',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();compare(a.certified,a.source,a.output)


if __name__=='__main__':main()
