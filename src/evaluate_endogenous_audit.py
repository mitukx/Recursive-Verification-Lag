"""Exploratory full-pilot endogenous audit benchmark; not heldout evidence."""
import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from src.candidate_bank_experiment import Config, load_jsonl
from src.source_audit import SourceAudit
from src.endogenous_audit import EndogenousAuditPlanner, endogenous_run
from src.analyze_candidate_sweep import task_bootstrap


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(bank_path, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'scope': 'exploratory retrospective full eight-task pilot, no new transfer evidence',
        'bank_sha256': sha(bank_path), 'initial_sources': 2, 'additional_source_cap': 2,
        'accepted_update_cap': 2, 'gain_floor': .01, 'seeds': list(range(5)),
        'representations': ['public', 'all'], 'optimizers': ['soft eta=1', 'bon N=4'],
        'controllers': ['myopic refit after each query', 'stream all queries before updates',
                        'direct paid-known-positive selection outside the fitted-proposal action family',
                        'escape one stream query from an all-zero paid transcript',
                        'one_update receding horizon with the full label cap',
                        'lookahead over stop/query/certified-update'],
        'lookahead_independent_binary_priors': [.1, .5, .9],
        'primary_prior': .5, 'prior_sensitivity': 'uncalibrated, not selected by outcomes',
        'uncertainty': 'descriptive paired task bootstrap, eight clusters, no confirmatory claim',
        'limitations': ['previously inspected tasks', 'frozen candidate support', 'no parameter training',
                        'deterministic binary finite-bank labels', 'small exact action space'],
        'code_sha256': {p: sha(p) for p in ['src/endogenous_audit.py',
            'src/evaluate_endogenous_audit.py', 'src/candidate_bank_experiment.py', 'src/decision_audit.py']}}
    # Written before execution as a reproducibility record, not preregistration.
    (output/'design.json').write_text(json.dumps(manifest, indent=2)+'\n')
    raw = [json.loads(s) for s in bank_path.read_text().splitlines() if s.strip()]
    source_by_id = {r['candidate_id']: r['source_sha256'] for r in raw}
    bank = load_jsonl(bank_path)
    if not np.isin(bank.trusted_score, [0., 1.]).all():
        raise ValueError('binary labels required')
    records = []
    started = time.perf_counter()
    with gzip.open(output/'paid_transcripts.jsonl.gz', 'wt') as sink:
        for task, frame in bank.groupby('task_id', sort=True):
            df = frame.reset_index(drop=True)
            sources = np.array([source_by_id[c] for c in df.candidate_id])
            rewards = {}
            for g in set(sources):
                values = df.trusted_score.to_numpy(float)[sources == g]
                if np.ptp(values) != 0:
                    raise ValueError('inconsistent identical-source labels')
                rewards[g] = float(values[0])
            unlabeled = df.drop(columns=['trusted_score'])
            for representation in ('public', 'all'):
                for optimizer in ('soft', 'bon'):
                    for seed in range(5):
                        cfg = Config(optimizer=optimizer, eta=1., best_of_n=4,
                                     representation=representation, seed=seed, rounds=2)
                        sampler = SourceAudit(sources, seed, 'stream')
                        order = [sampler.groups[i] for i in sampler.order]
                        initial = {g: rewards[g] for g in order[:2]}
                        keys = {'task_id': str(task), 'representation': representation,
                                'optimizer': optimizer, 'seed': seed}
                        for strategy, prior in [('myopic', .5), ('stream', .5),
                                                ('direct', .5),
                                                ('escape', .5), ('one_update', .5),
                                                ('lookahead', .5), ('lookahead', .1), ('lookahead', .9)]:
                            planner = EndogenousAuditPlanner(unlabeled, sources, cfg,
                                prior=[prior]*len(set(sources)), gain_floor=.01)
                            paid = []
                            def oracle(g):
                                if g in initial or g in paid:
                                    raise AssertionError('duplicate charge')
                                paid.append(g)
                                return rewards[g]
                            out = endogenous_run(planner, initial, oracle, budget=2, updates=2,
                                                 strategy=strategy, order=order)
                            gain = float((out['policy']-planner.baseline) @ df.trusted_score.to_numpy(float))
                            if gain < -1e-9 or len(paid) != len(out['queries']) or len(paid) > 2:
                                raise AssertionError('safety or cost accounting failed')
                            # All full-outcome diagnostics start after the decision.
                            trap = int(not any(initial.values()) and any(rewards.values()))
                            records.append({**keys, 'strategy': strategy, 'planning_prior': prior,
                                'gain_evaluation_only': gain, 'positive_gain': int(gain >= .01),
                                'paid_additional_labels': len(paid), 'paid_total_labels': 2+len(paid),
                                'accepted_updates': sum(t['action'] == 'update' for t in out['trace']),
                                'initial_zero_with_positive_bank_evaluation_only': trap,
                                'planning_states': out['planning_states']})
                            sink.write(json.dumps({**keys, 'strategy': strategy, 'planning_prior': prior,
                                'initial_paid_labels': initial, 'actions': out['trace'],
                                'final_policy': out['policy'].tolist()})+'\n')
            print(f'task {task}: {time.perf_counter()-started:.1f}s', flush=True)
    results = pd.DataFrame(records)
    results.to_csv(output/'final_runs.csv', index=False)
    metrics = ['gain_evaluation_only', 'positive_gain', 'paid_additional_labels',
               'paid_total_labels', 'accepted_updates', 'planning_states']
    results.groupby(['strategy', 'planning_prior'])[metrics].mean().reset_index().to_csv(
        output/'summary.csv', index=False)
    results.groupby(['task_id', 'strategy', 'planning_prior'])[metrics].mean().reset_index().to_csv(
        output/'by_task.csv', index=False)
    trap = results[results.initial_zero_with_positive_bank_evaluation_only == 1]
    trap.groupby(['task_id', 'strategy', 'planning_prior'])[metrics].mean().reset_index().to_csv(
        output/'trap_diagnostic.csv', index=False)
    reference = results[results.strategy == 'myopic']
    contrasts = []
    keys = ['task_id', 'representation', 'optimizer', 'seed']
    for (strategy, prior), cohort in results[results.strategy != 'myopic'].groupby(['strategy', 'planning_prior']):
        pairs = cohort.merge(reference, on=keys, validate='one_to_one', suffixes=('', '_myopic'))
        if len(pairs) != len(reference):
            raise AssertionError('unmatched settings')
        for metric in ['gain_evaluation_only', 'positive_gain', 'paid_additional_labels']:
            pairs['difference'] = pairs[metric]-pairs[metric+'_myopic']
            mean, lo, hi = task_bootstrap(pairs, 'difference')
            contrasts.append({'strategy_minus_myopic': strategy, 'planning_prior': prior, 'metric': metric,
                'task_mean_difference': mean, 'descriptive_task_bootstrap_lo': lo,
                'descriptive_task_bootstrap_hi': hi, 'tasks': pairs.task_id.nunique(), 'paired_settings': len(pairs)})
    pd.DataFrame(contrasts).to_csv(output/'paired_contrasts.csv', index=False)
    mechanism = []
    primary = results[results.planning_prior == .5]
    for treatment, reference_name in [('lookahead', 'escape'), ('direct', 'lookahead'),
                                       ('one_update', 'lookahead')]:
        pair = primary[primary.strategy == treatment].merge(primary[primary.strategy == reference_name],
            on=keys, validate='one_to_one', suffixes=('', '_reference'))
        for metric in ['gain_evaluation_only', 'paid_additional_labels']:
            pair['difference'] = pair[metric]-pair[metric+'_reference']
            mean, lo, hi = task_bootstrap(pair, 'difference')
            mechanism.append({'treatment': treatment, 'reference': reference_name, 'metric': metric,
                'task_mean_difference': mean, 'descriptive_task_bootstrap_lo': lo,
                'descriptive_task_bootstrap_hi': hi, 'tasks': pair.task_id.nunique(), 'paired_settings': len(pair)})
    pd.DataFrame(mechanism).to_csv(output/'mechanism_contrasts.csv', index=False)
    manifest.update({'runtime_seconds': time.perf_counter()-started, 'tasks': results.task_id.nunique(),
                     'rows': len(results), 'outputs_sha256': {p.name: sha(p) for p in sorted(output.iterdir())
                       if p.is_file() and p.name not in ('manifest.json',)}})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bank', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.bank, args.output)
