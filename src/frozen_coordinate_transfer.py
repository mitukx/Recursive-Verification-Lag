"""Fit thresholds on development only; test on new banks without refitting."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.analyze_candidate_sweep import METRICS,fit_threshold,balanced_accuracy


def fit(development,output):
    if output.exists():raise ValueError('refuse to replace frozen thresholds')
    rows=pd.read_csv(development)
    rows=rows[(rows.controller=='fixed')&(rows.total_audit_budget==32)].copy()
    prior=rows.groupby('run_id').below_initial.cumsum()-rows.below_initial
    rows=rows[prior==0]
    rows['geometry_over_margin2']=rows.restricted_geometry/np.maximum(abs(rows.proxy_margin/rows.score_scale),1e-8)**2
    models=[]
    for (optimizer,rep),g in rows.groupby(['optimizer','representation']):
        for metric,col in METRICS.items():
            if optimizer=='bon' and metric=='eta_times_age':continue
            result=fit_threshold(g[col],g.below_initial)
            models.append({'optimizer':optimizer,'representation':rep,'metric':metric,'column':col,
                           'fit':None if result is None else [str(result[0]),int(result[1])]})
    output.write_text(json.dumps({'training_sha256':hashlib.sha256(development.read_bytes()).hexdigest(),
       'training_scope':'development fixed controllers, cap32, pre-first-failure rows',
       'test_scope':'transfer fixed early/uniform/late, pre-first-failure rows, stratified optimizer and representation',
       'models':models},indent=2)+'\n')


def evaluate(models_path,roots,output):
    records=[]
    for root in roots:
        with gzip.open(root/'runs.jsonl.gz','rt') as f:
            for line in f:
                r=json.loads(line)
                if r['design'] not in ['early','uniform','late']:continue
                for t,y in enumerate(r['below_initial']):
                    row={'bank':root.name,'task_id':r['task_id'],'optimizer':r['optimizer'],'representation':r['representation'],'label':y}
                    row.update({c:(np.nan if v[t] is None else v[t]) for c,v in r['coordinates'].items()})
                    row['geometry_over_margin2']=row['restricted_geometry']/max(abs(row['proxy_margin']),1e-8)**2
                    records.append(row)
                    if y:break
    rows=pd.DataFrame(records);out=[]
    for bank,g in rows.groupby('bank'):
        for model in json.loads(models_path.read_text())['models']:
            te=g[(g.optimizer==model['optimizer'])&(g.representation==model['representation'])]
            te=te[np.isfinite(te[model['column']])]
            rec={**{k:model[k] for k in ['optimizer','representation','metric']},'bank':bank,'rows':len(te),'failures':int(te.label.sum())}
            if model['fit'] is None or te.label.nunique()<2:
                rec.update(status='unidentified',balanced_accuracy=None)
            else:
                threshold,direction=model['fit'];threshold=float(threshold)
                pred=te[model['column']]>=threshold if direction==1 else te[model['column']]<threshold
                rec.update(status='ok',balanced_accuracy=balanced_accuracy(te.label,pred),threshold=threshold,direction=direction)
            out.append(rec)
    pd.DataFrame(out).to_csv(output,index=False)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('fit');f.add_argument('development',type=Path);f.add_argument('output',type=Path)
    e=sub.add_parser('evaluate');e.add_argument('models',type=Path);e.add_argument('output',type=Path);e.add_argument('results',type=Path,nargs='+')
    a=p.parse_args()
    if a.command=='fit':fit(a.development,a.output)
    else:evaluate(a.models,a.results,a.output)

if __name__=='__main__':main()
