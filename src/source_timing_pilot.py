"""Equal unique-source oracle cost; replay separates timing from acquisition."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.candidate_bank_experiment import Config,load_jsonl,run
from src.source_audit import SourceAudit
from src.analyze_candidate_sweep import task_bootstrap

DESIGNS={'early':('fixed',.5,(1,2,3)), 'uniform':('fixed',.5,(1,5,9)),
         'late':('fixed',.5,(1,11,12)), 'kl_05':('kl',.5,None),
         'geometry_10':('geometry',10.,None)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('bank',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    raw=[json.loads(x) for x in a.bank.read_text().splitlines()];by_id={r['candidate_id']:r for r in raw}
    bank=load_jsonl(a.bank)
    manifest={'designs':DESIGNS,'acquisition':['stream','policy'],'events':3,'labels_per_event':2,
      'rounds':12,'seeds':list(range(5)),'policy_acquisition_exploration':.05,
      'bank_sha256':hashlib.sha256(a.bank.read_bytes()).hexdigest(),
      'status':'exploratory existing development bank; design fixed before this run after previous results',
      'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
        ['src/source_timing_pilot.py','src/source_audit.py','src/candidate_bank_experiment.py']}}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    records=[]
    with (a.output/'runs.jsonl').open('w') as f:
        for task,df in bank.groupby('task_id',sort=True):
            sources=[by_id[c]['source_sha256'] for c in df.candidate_id]
            assert len(set(sources))>=6
            # Offline data-integrity check, never passed to the acquisition rule.
            chk=df.copy();chk['source']=sources
            assert chk.groupby('source').trusted_score.nunique().max()==1
            for opt,ss in [('soft',[.25,1.,4.]),('bon',[2,4,16])]:
                for strength in ss:
                    for rep in ['public','all']:
                        for seed in range(5):
                            first=None;stream=None
                            for mode in ['stream','policy']:
                                for name,(ctrl,th,schedule) in DESIGNS.items():
                                    cfg=Config(optimizer=opt,eta=strength if opt=='soft' else 1.,best_of_n=int(strength) if opt=='bon' else 4,
                                        representation=rep,seed=seed,controller=ctrl,threshold=th,rounds=12,audit_per_refresh=2,total_audit_budget=6)
                                    h,tr=run(df,cfg,return_trace=True,audit_schedule=schedule,exact_audit_events=3,
                                        audit_sampler=SourceAudit(sources,seed,mode))
                                    if first is None:first=tr[0]
                                    assert first['audit_indices']==tr[0]['audit_indices']
                                    np.testing.assert_array_equal(first['policy'],tr[0]['policy'])
                                    labels=tr[-1]['audit_indices'];assert len(set(np.asarray(sources)[list(labels)]))==6
                                    if mode=='stream':
                                        if stream is None:stream=labels
                                        assert stream==labels
                                    rec={'task_id':task,'optimizer':opt,'strength':strength,'representation':rep,'seed':seed,
                                        'acquisition':mode,'design':name,'failure':int(h.below_initial.any()),
                                        'initial_failure':int(h.iloc[0].below_initial),'gain':float(h.iloc[-1].true_reward-h.iloc[-1].initial_true_reward),
                                        'cost':int(h.iloc[-1].audit_labels),'unique_sources':6,
                                        'deadline_events':int((h.refresh_reason=='deadline').sum()),
                                        'audit_indices':list(labels),'audit_rounds':h.loc[h.refresh==1,'round'].tolist(),'rewards':h.true_reward.tolist()}
                                    f.write(json.dumps(rec)+'\n');f.flush();records.append(rec)
            print('completed',task,flush=True)
    rows=pd.DataFrame(records);summaries=[];keys=['task_id','optimizer','strength','representation','seed']
    for mode,g in rows.groupby('acquisition'):
        baseline=g[g.design=='uniform']
        for name,h in g.groupby('design'):
            pair=h.merge(baseline,on=keys,suffixes=('','_uniform'),validate='one_to_one')
            for cohort in ['all','initially_safe_posthoc']:
                p=pair if cohort=='all' else pair[pair.initial_failure==0]
                rec={'acquisition':mode,'design':name,'cohort':cohort,'pairs':len(p),'tasks':p.task_id.nunique(),
                     'mean_deadline_events':p.deadline_events.mean()}
                for metric in ['failure','gain']:
                    p=p.copy();p['delta']=p[metric]-p[metric+'_uniform'];m,lo,hi=task_bootstrap(p,'delta')
                    rec.update({metric:p.groupby('task_id')[metric].mean().mean(),metric+'_difference':m,metric+'_lo':lo,metric+'_hi':hi})
                summaries.append(rec)
    pd.DataFrame(summaries).to_csv(a.output/'summary.csv',index=False)
    # Acquisition effect within each timing design; never mix timing baselines.
    comparisons=[]
    for name,g in rows.groupby('design'):
        p=g[g.acquisition=='policy'].merge(g[g.acquisition=='stream'],on=keys,suffixes=('','_stream'),validate='one_to_one')
        rec={'design':name}
        for metric in ['failure','gain']:
            p['delta']=p[metric]-p[metric+'_stream'];m,lo,hi=task_bootstrap(p,'delta');rec.update({metric+'_difference':m,metric+'_lo':lo,metric+'_hi':hi})
        comparisons.append(rec)
    pd.DataFrame(comparisons).to_csv(a.output/'acquisition_effect.csv',index=False)
    data=(a.output/'runs.jsonl').read_bytes();compressed=gzip.compress(data,mtime=0);(a.output/'runs.jsonl.gz').write_bytes(compressed)
    manifest.update({'completed_runs':len(rows),'runs_sha256':hashlib.sha256(data).hexdigest(),'gzip_sha256':hashlib.sha256(compressed).hexdigest()})
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(pd.DataFrame(summaries).to_string(index=False));print(pd.DataFrame(comparisons).to_string(index=False))

if __name__=='__main__':main()
