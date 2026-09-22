"""Per-task factorial sweep. Stores all paths and factors, never chooses winners."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import pandas as pd
from src.candidate_bank_experiment import Config, load_jsonl, run


def configurations(spec):
    optimizers=[('soft',eta,1) for eta in spec['soft_eta']]+[('bon',0.,n) for n in spec['best_of_n']]
    controls=[('fixed',L,0.) for L in spec['fixed_intervals']]
    for kind,key in [('kl','kl_thresholds'),('maxlog','maxlog_thresholds'),('geometry','geometry_thresholds')]:
        controls += [(kind,1,t) for t in spec[key]]
    for op,control,rep,budget,seed in itertools.product(optimizers,controls,spec['representations'],spec['total_audit_budgets'],spec['seeds']):
        yield Config(eta=op[1],optimizer=op[0],best_of_n=op[2],controller=control[0],
                     refresh_interval=control[1],threshold=control[2],representation=rep,
                     total_audit_budget=budget,seed=seed,rounds=spec['rounds'],
                     audit_per_refresh=spec['audit_per_refresh'],failure_tolerance=spec['failure_tolerance'])


def main():
    p=argparse.ArgumentParser(); p.add_argument('bank',type=Path)
    p.add_argument('--config',type=Path,default=Path('configs/pilot_sweep.json'))
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists(): p.error('refusing to overwrite results')
    spec=json.loads(a.config.read_text()); df=load_jsonl(a.bank); a.output.mkdir(parents=True)
    bank_meta_path=a.bank.with_suffix('.manifest.json')
    bank_meta=json.loads(bank_meta_path.read_text()) if bank_meta_path.exists() else {}
    bank_hash=hashlib.sha256(a.bank.read_bytes()).hexdigest()
    if bank_meta and (not bank_meta.get('complete') or bank_meta.get('bank_sha256')!=bank_hash):
        raise ValueError('incomplete or changed bank')
    all_rows=[]; summaries=[]; configs=list(configurations(spec))
    for task,group in df.groupby('task_id',sort=True):
        for ci,cfg in enumerate(configs):
            run_id=f'{task}:{ci}'
            result=run(group,cfg); result.insert(0,'task_id',task); result.insert(1,'run_id',run_id)
            all_rows.append(result)
            last=result.iloc[-1]
            failed=result[result.below_initial==1]
            summaries.append({'run_id':run_id,'task_id':task,**cfg.__dict__,
                'ever_below_initial':int(len(failed)>0),'first_failure_round':int(failed.iloc[0]['round']) if len(failed) else None,
                'min_true_reward':float(result.true_reward.min()),'final_true_reward':last.true_reward,
                'initial_true_reward':last.initial_true_reward,'final_gain':last.true_reward-last.initial_true_reward,
                'audit_labels':int(last.audit_labels),'unique_audit_labels':int(last.unique_audit_labels),
                'refresh_count':int(last.refresh_count)})
        print(task, len(summaries), 'runs',flush=True)
    pd.concat(all_rows,ignore_index=True).to_csv(a.output/'trajectories.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(summaries).to_csv(a.output/'runs.csv',index=False)
    manifest={'bank_sha256':hashlib.sha256(a.bank.read_bytes()).hexdigest(),
              'config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),'config':spec,
              'runs':len(summaries),'rows':sum(len(r) for r in all_rows),
              'budget_semantics':'matched caps, actual paid draws may differ; no matched-cost superiority claim',
              'evidence':bank_meta.get('evidence','unverified_input_provenance'),
              'scope':spec.get('evidence','unspecified'),
              'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src').glob('*.py')}}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__': main()
