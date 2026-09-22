"""Write a factual research record from archived pilot outputs."""
import argparse
import json
from pathlib import Path
import pandas as pd


def table(df):
    cols=list(df.columns)
    lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument('results',type=Path)
    p.add_argument('--bank-manifest',type=Path,required=True);p.add_argument('--quality',type=Path,required=True)
    p.add_argument('--run-url',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    summary=json.loads((a.results/'summary.json').read_text());manifest=json.loads(a.bank_manifest.read_text())
    quality=pd.read_csv(a.quality);transfer=pd.read_csv(a.results/'transfer.csv')
    metrics=transfer[transfer.status=='ok'].groupby(['axis','metric']).balanced_accuracy.agg(['mean','count']).reset_index()
    refresh=pd.read_csv(a.results/'crossfit_refresh_summary.csv')
    refresh=refresh[refresh.training_cost_ceiling.isna()][['budget_cap','failure_difference','failure_difference_lo','failure_difference_hi','gain_difference','cost_difference','adaptive_mean_cost','fixed_mean_cost']]
    cohort=pd.read_csv(a.results/'initial_safe_summary.csv')
    text=f'''# Progress XXIII — Auditable pretrained expression pilot

Status: completed exploratory pilot, **not** confirmatory real-world agent validation.
Execution: {a.run_url}
Bank SHA256: `{manifest['bank_sha256']}`
Model revision: `{manifest['configuration']['revision']}`

## Scope and the three tasks

1. Generated {manifest['rows']} sample occurrences from the immutable pretrained
Qwen2.5-Coder-0.5B-Instruct checkpoint on 12 finite-domain tasks. Public inputs are
(0,1,2,3); reward is exact pass fraction on the 29 disjoint hidden inputs. Generated
text is evaluated by an explicit bounded expression interpreter. LM weights are
frozen; recursive selection changes only the bank distribution.
2. Executed {summary['runs']:,} task/configuration/audit-seed runs and archived all
trajectories. Factors: soft eta (.25,1,4), BoN N (2,4,16), four fixed cadences,
three adaptive families, two representations, two nominal audit caps, five seeds.
These runs share one bank and are not independent model-generation replications.
3. Evaluated transfer, refresh cost/progress, and a shared initially-safe cadence
cohort. Proved only the conditional robust bounds and elementary identities in
`docs/observable_refresh_frontier.md`; no optimal adaptive frontier is claimed.

## H1: does proxy optimization degrade trusted reward?

Across the declared grid, {summary['collapse_runs']:,}/{summary['runs']:,}
runs ({100*summary['collapse_fraction']:.2f}%) crossed below their initial reward.
{summary['decline_gt_1pct_runs']:,} declined by more than .01 and
{summary['decline_gt_5pct_runs']:,} by more than .05. Worst change was
{summary['worst_baseline_change']:.6f}; mean final gain across the grid was
{summary['mean_final_gain']:.6f}. This grid frequency is not a deployment probability.

{summary['first_round_crossings']} first crossings occurred in round one;
{summary['first_crossing_with_stale_age_gt_1']} occurred with verifier age > 1.
A crossing is not by itself evidence that staleness caused it. Initial-fit error,
representation misspecification, support quality, and later budget exhaustion are
alternative explanations. Those counts are preserved rather than excluded from
the headline statistic.

## H2: does one observable failure coordinate transfer?

Balanced accuracy below is averaged over identifiable held-group folds, with the
number of such folds shown. Thresholds and directions are fitted without that
group. Single-class folds are unidentified. Eta-times-age is not defined for BoN.
Rows after a run's first failure are excluded. This is exploratory cross-validation
on development data, not a sealed final holdout or a significance-ranked leaderboard.

{table(metrics)}

Do not select a universal winner from inconsistent task/optimizer/representation
rankings. The feature geometry is an observable heuristic, not a validated bound
on arbitrary misspecification; oracle error contrasts are never controller inputs.

## H3: does adaptive refresh improve the cost/progress frontier?

The table compares leave-one-task-out selected adaptive and fixed schedules.
Differences are adaptive minus fixed. Failure-difference intervals use paired
bootstrap resampling of tasks. Caps are matched; actual draw counts are not.

{table(refresh)}

Savings accompanied by lower reward are a tradeoff, not dominance. Superiority
at the same realized cost and useful progress remains unestablished. The complete
training-cost-ceiling sweep, including held-task ceiling violations, is archived;
none of its favorable rows is promoted as a confirmatory result.

## Additional post-hoc mechanism diagnostic

For cap 96, initial audits and first updates are checked to coincide across fixed
cadences. A single common offline initial-safety criterion then defines the cohort.
This criterion uses trusted outcomes for analysis, so it is not a deployable filter.
The fresh schedule can buy more labels; this is a mechanism/cost diagnostic, not a
matched-cost intervention. Differences below are stale minus fresh. Failure means/differences and bootstrap
intervals give each task equal weight; separate pair-rate columns weight each
eligible configuration equally. These estimands differ because cohort sizes vary.

{table(cohort)}

## Bank quality and falsifiability

Valid-expression fraction: {quality.valid_fraction.mean():.4f}.
Tasks with no fully correct candidate: {int((quality.fully_correct_candidates==0).sum())}/12.
All invalid outputs and duplicates remain in the denominator; no hand-written
exploit was inserted to make the primary bank collapse. These limitations restrict
claims even when a failure is observed.

{table(quality)}

## Decision ledger

| Step | Hypothesis / experiment | Falsification criterion | Next decision |
| --- | --- | --- | --- |
| Pretrained bank and loop | Imperfect proxy optimization can degrade hidden reward; fixed frozen bank, independent factor sweeps | No degradation, or degradation dominated by initial fit/format artifacts | Separate initial failures; enlarge support and use a new task-family holdout |
| Transfer | A geometry coordinate transfers across optimization and representation changes | Reversed rankings or poor held-group accuracy | Retain failed coordinates; validate residual-error assumptions before claiming a law |
| Adaptive refresh | Timely refresh improves useful progress per paid label | Apparent safety bought by extra labels or reduced progress | Compare at equal realized costs; keep uncertainty and task clustering |
| Theory | An explicit error envelope supports simultaneous adaptive certificates | Assumptions do not cover misspecification, or a rule only abstains | Keep conditional propositions separate; optimality/completeness remain open |

## Evidence preservation and next decision

The earlier local pilot's raw outputs were lost on workspace reset; its console
summaries are recorded separately in `docs/pilot_recovery_record.md` and are not
substituted for this bank. This recovery has a distinct artifact identity. Raw bank
artifacts were saved before sweeps. Hardware/software and source hashes are in the
bank and run manifests. Exact agreement with the lost bank cannot be established.

The next substantive experiment should preregister an initial-verification
procedure, expand bank support and task coverage, and reserve a new task-family
holdout plus an independent model family. Compare deployment progress and failure
under equal realized audit costs, including abstention and independent generation
replicates. Do not expand the story beyond when verification must catch up with
policy movement. A universal scalar law and an optimal observable adaptive frontier
remain open.
'''
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)

if __name__=='__main__':main()
