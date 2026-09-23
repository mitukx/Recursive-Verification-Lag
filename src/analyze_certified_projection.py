"""Paired exploratory comparison of certified projection and abstention."""
import argparse
from pathlib import Path
import pandas as pd
from src.analyze_candidate_sweep import task_bootstrap


def compare(old_path, projected_path, bank):
    old=pd.read_csv(old_path); new=pd.read_csv(projected_path)
    keys=['task_id','optimizer','representation','seed','round']
    paired=old.merge(new,on=keys,validate='one_to_one',suffixes=('_old','_new'))
    if len(paired)!=len(old) or len(paired)!=len(new):
        raise ValueError('unmatched controller settings')
    end=paired[paired['round']==12].copy()
    end['gain_delta']=end.gain_evaluation_only_new-end.gain_evaluation_only_old
    delta,lo,hi=task_bootstrap(end,'gain_delta')
    projected=new[new.projection_distance.notna()]
    return dict(bank=bank,tasks=end.task_id.nunique(),paired_runs=len(end),
                old_mean_gain=end.gain_evaluation_only_old.mean(),
                projected_mean_gain=end.gain_evaluation_only_new.mean(),
                gain_delta=delta,task_bootstrap_95_low=lo,task_bootstrap_95_high=hi,
                mean_paid_sources=end.paid_source_labels_new.mean(),
                old_audited_mass=end.audited_policy_mass_old.mean(),
                projected_audited_mass=end.audited_policy_mass_new.mean(),
                initial_audited_mass=end.initial_audited_mass_new.mean(),
                old_effective_support=end.effective_source_support_old.mean(),
                projected_effective_support=end.effective_source_support_new.mean(),
                projected_states=len(projected),
                accepted_projected_states=int(projected.accepted.sum()),
                baseline_failures=int((new.gain_evaluation_only< -1e-10).sum()),
                smallest_certificate=float(new.certified_lower.min()))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    pairs=[('v1','results/certified_abstain_v1_coverage/runs.csv.gz',
            'results/certified_project_v1_coverage/runs.csv.gz'),
           ('v2','results/certified_abstain_v2_coverage/runs.csv.gz',
            'results/certified_project_v2_coverage/runs.csv.gz')]
    result=pd.DataFrame([compare(Path(o),Path(n),bank) for bank,o,n in pairs])
    result.to_csv(args.output,index=False)
    print(result.to_string(index=False))


if __name__=='__main__':main()
