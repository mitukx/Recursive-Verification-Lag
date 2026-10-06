"""Retrospective decision-cost study on previously scored binary MBPP+ pilot.

No new generation or scoring. Outcomes outside paid transcripts are used only
for evaluator diagnostics, including a clairvoyant witness lower bound. Never
apply this exploratory design to the frozen heldout split without a new lock.
"""
import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from src.candidate_bank_experiment import Config, load_jsonl, task_policy, fit_verifier, propose
from src.decision_audit import source_contrast, minimum_witness, audit_comparison
from src.decision_refresh import decision_refresh_run
from src.source_audit import SourceAudit
from src.analyze_candidate_sweep import task_bootstrap

STRENGTHS = [('soft', .25), ('soft', 1.), ('soft', 4.), ('bon', 2), ('bon', 4), ('bon', 16)]
STRATEGIES = ('stream', 'impact', 'decision')
GAIN_FLOOR = .01


def evaluate(bank_path, output, *, seeds=5, rounds=12):
    output.mkdir(parents=True, exist_ok=False)
    raw = [json.loads(s) for s in bank_path.read_text().splitlines() if s.strip()]
    sources_by_id = {r['candidate_id']: r['source_sha256'] for r in raw}
    bank = load_jsonl(bank_path)
    if not np.isin(bank.trusted_score, [0., 1.]).all():
        raise ValueError('binary benchmark required; never discretize fractional rewards')
    comparisons, trajectories, transcripts = [], [], []
    start = time.perf_counter()
    for task, frame in bank.groupby('task_id', sort=True):
        df = frame.reset_index(drop=True)
        sources = np.array([sources_by_id[c] for c in df.candidate_id])
        true_rewards = {}
        for g in set(sources):
            values = df.trusted_score.to_numpy(float)[sources == g]
            if np.ptp(values) > 0:
                raise ValueError('identical source has inconsistent reward')
            true_rewards[g] = float(values[0])
        for representation in ('public', 'all'):
            cols = sorted(c for c in df if c.startswith('f::') and
                (representation == 'all' or c == 'f::public_score'))
            X = np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)])
            baseline = task_policy(df, df.base_logprob.to_numpy(float))
            for optimizer, strength in STRENGTHS:
                for seed in range(seeds):
                    cfg = Config(optimizer=optimizer, eta=strength if optimizer == 'soft' else 1.,
                        best_of_n=int(strength) if optimizer == 'bon' else 4,
                        representation=representation, seed=seed, rounds=rounds)
                    sampler = SourceAudit(sources, seed, 'stream')
                    order = [sampler.groups[i] for i in sampler.order]
                    initial_sources = order[:2]
                    indices = [int(np.flatnonzero(sources == g)[0]) for g in initial_sources]
                    known = {g: true_rewards[g] for g in initial_sources}
                    theta = fit_verifier(X[indices], np.array(list(known.values())), cfg.ridge)
                    proposal = baseline.copy()
                    for _ in range(rounds):
                        proposal = propose(df, proposal, X @ theta, cfg)
                    groups, weights = source_contrast(proposal, baseline, sources)
                    min_calls, _, actual_status = minimum_witness(
                        groups, weights, true_rewards, known, threshold=GAIN_FLOOR)
                    true_gain = float((proposal-baseline) @ df.trusted_score.to_numpy(float))
                    keys = {'task_id': str(task), 'representation': representation,
                        'optimizer': optimizer, 'strength': strength, 'seed': seed}
                    for additional_cap in (0, 1, 2, 4, 6):
                        for strategy in STRATEGIES:
                            paid = []
                            def oracle(g):
                                if g in known or g in paid:
                                    raise AssertionError('unaccounted repeat source')
                                paid.append(g)
                                return true_rewards[g]
                            out = audit_comparison(proposal, baseline, sources, known, oracle,
                                budget=additional_cap, strategy=strategy,
                                threshold=GAIN_FLOOR, order=order)
                            if len(paid) != out['paid_additional_labels']:
                                raise AssertionError('label cost mismatch')
                            resolved = out['status'] != 'unresolved'
                            if resolved and out['status'] != actual_status:
                                raise AssertionError('unsound decision')
                            if resolved and len(paid) < min_calls:
                                raise AssertionError('witness lower bound violated')
                            comparisons.append({**keys, 'additional_cap': additional_cap,
                                'strategy': strategy, 'resolved': int(resolved),
                                'accepted': int(out['status'] == 'accept'),
                                'status': out['status'], 'actual_status_evaluation_only': actual_status,
                                'paid_additional_labels': len(paid), 'paid_total_labels': 2+len(paid),
                                'oracle_min_additional_evaluation_only': min_calls,
                                'oracle_resolvable_within_cap_evaluation_only': int(min_calls <= additional_cap),
                                'resolved_query_excess_evaluation_only': len(paid)-min_calls if resolved else np.nan,
                                'candidate_true_gain_evaluation_only': true_gain,
                                'accepted_gain_evaluation_only': true_gain if out['status'] == 'accept' else 0.,
                                'lower': out['lower'], 'upper': out['upper'],
                                'planning_states': out['planning_states']})
                            transcripts.append({**keys, 'phase': 'frozen_comparison',
                                'additional_cap': additional_cap, 'strategy': strategy,
                                'initial_sources': initial_sources,
                                'queries': out['queries'], 'status': out['status']})
                    # A separate recursive experiment: the candidate and target
                    # can differ between strategies, so it is NOT a timing effect.
                    for total_cap in (2, 4, 6):
                        cap = min(total_cap, len(groups))
                        for strategy in STRATEGIES:
                            history, trace = decision_refresh_run(df, sources, cfg,
                                budget=cap, strategy=strategy, gain_floor=GAIN_FLOOR,
                                return_trace=True)
                            trajectories.extend([{**keys, **row} for row in history.to_dict('records')])
                            transcripts.append({**keys, 'phase': 'recursive',
                                'budget_cap': cap, 'strategy': strategy,
                                'rounds': [{'round': t+1, 'paid_sources': point['paid_sources'],
                                    'queries': point['queries'], 'policy': point['policy'].tolist()}
                                    for t, point in enumerate(trace)]})
        print(f'task {task}: {time.perf_counter()-start:.1f}s', flush=True)
    fixed = pd.DataFrame(comparisons)
    recursive = pd.DataFrame(trajectories)
    fixed.to_csv(output/'comparisons.csv.gz', index=False)
    recursive.to_csv(output/'recursive_runs.csv.gz', index=False)
    with gzip.open(output/'query_transcripts.jsonl.gz', 'wt') as out:
        for row in transcripts:
            out.write(json.dumps(row)+'\n')
    metrics = ['resolved', 'accepted', 'paid_additional_labels',
        'oracle_min_additional_evaluation_only', 'oracle_resolvable_within_cap_evaluation_only',
        'resolved_query_excess_evaluation_only', 'accepted_gain_evaluation_only']
    fixed.groupby(['additional_cap', 'strategy'])[metrics].mean().reset_index().to_csv(
        output/'comparison_summary.csv', index=False)
    fixed.groupby(['task_id', 'additional_cap', 'strategy'])[metrics].mean().reset_index().to_csv(
        output/'comparison_by_task.csv', index=False)
    final = recursive[recursive['round'] == rounds]
    metrics = ['gain_evaluation_only', 'paid_source_labels', 'audited_policy_mass', 'effective_source_support']
    final.groupby(['budget_cap', 'strategy'])[metrics].mean().reset_index().to_csv(
        output/'recursive_summary.csv', index=False)
    final.groupby(['task_id', 'budget_cap', 'strategy'])[metrics].mean().reset_index().to_csv(
        output/'recursive_by_task.csv', index=False)
    contrasts = []
    keys = ['task_id', 'representation', 'optimizer', 'strength', 'seed']
    for phase, frame, cap, measures in [
            ('frozen', fixed, 'additional_cap', ['resolved', 'paid_additional_labels', 'accepted_gain_evaluation_only']),
            ('recursive', final, 'budget_cap', ['paid_source_labels', 'gain_evaluation_only'])]:
        for cap_value, cohort in frame.groupby(cap):
            reference = cohort[cohort.strategy == 'impact']
            for strategy in ('stream', 'decision'):
                pair = cohort[cohort.strategy == strategy].merge(reference, on=keys,
                    suffixes=('', '_impact'), validate='one_to_one')
                if len(pair) != len(reference):
                    raise AssertionError('unmatched acquisition comparison')
                for metric in measures:
                    pair['difference'] = pair[metric]-pair[metric+'_impact']
                    mean, lo, hi = task_bootstrap(pair, 'difference')
                    contrasts.append({'phase': phase, 'cap': cap_value, 'strategy_minus_impact': strategy,
                        'metric': metric, 'task_mean_difference': mean,
                        'descriptive_task_bootstrap_lo': lo, 'descriptive_task_bootstrap_hi': hi,
                        'tasks': pair.task_id.nunique(), 'paired_settings': len(pair)})
    pd.DataFrame(contrasts).to_csv(output/'paired_contrasts.csv', index=False)
    manifest = {'scope': 'retrospective eight-task finite-bank audit-acquisition study; not independent transfer',
        'bank_sha256': hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        'gain_floor': GAIN_FLOOR, 'initial_distinct_sources': 2,
        'strategies': STRATEGIES, 'decision_prior': 'independent Bernoulli(0.5); uncalibrated',
        'stale_block_rounds': rounds, 'seeds': list(range(seeds)), 'strengths': STRENGTHS,
        'tasks': sorted(str(t) for t in bank.task_id.unique()),
        'comparisons': len(fixed), 'recursive_rows': len(recursive),
        'elapsed_seconds': time.perf_counter()-start,
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__},
        'source_hashes': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ['src/decision_audit.py', 'src/decision_refresh.py', 'src/evaluate_decision_audit.py',
             'src/candidate_bank_experiment.py', 'src/identified_gain.py']}}
    manifest['artifact_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.iterdir() if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(pd.read_csv(output/'comparison_summary.csv').to_string(index=False))
    print(pd.read_csv(output/'recursive_summary.csv').to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bank', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.bank, args.output)
