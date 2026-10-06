"""Prespecified paired transfer comparisons, clustered by task, not run count."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pandas as pd
from src.analyze_candidate_sweep import task_bootstrap


KEYS=['task_id','optimizer','strength','representation','seed']


def contrasts(rows):
    """Preserve bank identity; compare every design against its own uniform arm."""
    outputs=[]
    for bank,g in rows.groupby('bank'):
        ref=g[g.design=='uniform']
        for design,h in g.groupby('design'):
            p=h.merge(ref,on=KEYS,suffixes=('','_uniform'),validate='one_to_one')
            if len(p)!=len(ref):raise ValueError('unmatched design cells')
            if not (p.cost==p.cost_uniform).all():raise ValueError('unequal paid cost')
            for metric in ['failure','gain']:
                p[metric+'_difference']=p[metric]-p[metric+'_uniform']
            p['bank']=bank;p['design']=design
            outputs.append(p[['bank','design',*KEYS,'failure_difference','gain_difference']])
    return pd.concat(outputs,ignore_index=True)


def summarize(pairs):
    out=[]
    for design,g in pairs.groupby('design'):
        rec={'design':design,'tasks':g.task_id.nunique(),'banks':g.bank.nunique(),'pairs':len(g)}
        # Seeds share tasks; task_bootstrap keeps all seed/config rows in cluster.
        for metric in ['failure','gain']:
            col=metric+'_difference';mean,lo,hi=task_bootstrap(g,col)
            perbank=g.groupby('bank')[col].mean()
            rec.update({col:mean,metric+'_lo':lo,metric+'_hi':hi,
                        metric+'_each_bank':json.dumps(perbank.to_dict(),sort_keys=True),
                        metric+'_desired_in_every_bank':bool((perbank<0).all() if metric=='failure' else (perbank>0).all())})
        rec['joint_desired_in_every_bank']=rec['failure_desired_in_every_bank'] and rec['gain_desired_in_every_bank']
        out.append(rec)
    return pd.DataFrame(out)


def main():
    p=argparse.ArgumentParser();p.add_argument('results',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);frames=[];inputs={}
    for root in a.results:
        path=root/'runs.jsonl.gz'
        if not path.exists():
            # Preserve raw records in compact form before analysis.
            data=(root/'runs.jsonl').read_bytes();path.write_bytes(gzip.compress(data,mtime=0))
        df=pd.read_json(path,lines=True);df['bank']=root.name;frames.append(df)
        inputs[root.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    rows=pd.concat(frames,ignore_index=True)
    sizes=rows.groupby(['bank','task_id']).size().unstack(0)
    if sizes.isna().any().any() or (sizes.nunique(axis=1)!=1).any():raise ValueError('unbalanced bank/task grid')
    pairs=contrasts(rows);summary=summarize(pairs)
    pairs.to_csv(a.output/'paired_contrasts.csv.gz',index=False)
    summary.to_csv(a.output/'summary.csv',index=False)
    bytask=pairs.groupby(['bank','task_id','design'])[['failure_difference','gain_difference']].mean().reset_index()
    bytask.to_csv(a.output/'task_contrasts.csv',index=False)
    manifest={'input_sha256':inputs,'runs':len(rows),'banks':rows.bank.nunique(),'tasks':rows.task_id.nunique(),
              'bootstrap_unit':'task; all bank, audit-seed and optimizer rows within task retained',
              'intervals':'descriptive percentile 95%, 1000 draws, no multiplicity correction',
              'primary_contrasts':['early versus uniform','kl_05 versus uniform'],
              'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(summary.to_string(index=False))


if __name__=='__main__':main()
