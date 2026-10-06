"""All-task development decision gate; never fit or open a heldout predictor."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.candidate_bank_experiment import load_jsonl, task_policy
from src.analyze_candidate_sweep import task_bootstrap


def onset_records(records, baselines):
    rows = []
    for record in records:
        if record['design'] not in ('early', 'uniform', 'late'):
            continue
        if record.get('acquisition', 'stream') != 'stream':
            continue
        row = dict(record)
        row['task_id'] = str(row['task_id'])
        baseline = baselines[row['task_id']]
        failures = np.array(row['rewards']) < baseline-1e-10
        row['initial_failure'] = int(failures[0])
        row['later_onset'] = int(not failures[0] and failures[1:].any())
        row['first_failure_round'] = int(np.flatnonzero(failures)[0]+1) if failures.any() else None
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(bank_path, draw_root, source_root, output):
    output.mkdir(parents=True, exist_ok=False)
    raw = [json.loads(s) for s in bank_path.read_text().splitlines() if s.strip()]
    bank = load_jsonl(bank_path)
    baselines = {str(task): float(task_policy(df, df.base_logprob.to_numpy(float)) @
        df.trusted_score.to_numpy(float)) for task, df in bank.groupby('task_id')}
    source_by_id = {r['candidate_id']: r['source_sha256'] for r in raw}
    support = []
    for task, df in bank.groupby('task_id', sort=True):
        support.append({'task_id': str(task), 'candidates': len(df),
            'unique_sources': len({source_by_id[c] for c in df.candidate_id}),
            'trusted_passes': int((df.trusted_score > 0).sum()),
            'initial_true_reward': baselines[str(task)],
            'finite_bank_reward_ceiling': float(df.trusted_score.max())})
    pd.DataFrame(support).to_csv(output/'task_support.csv', index=False)
    summaries, by_task, rows_by_arm = [], [], {}
    for arm, root in [('paid_draws', draw_root), ('identical_source_stream', source_root)]:
        path = root/'runs.jsonl'
        records = [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
        rows = onset_records(records, baselines)
        rows_by_arm[arm] = rows
        if rows.empty:
            summaries.append({'arm': arm, 'status': 'no_identified_source_timing_tasks'})
            continue
        rows.to_json(output/f'{arm}_onset_rows.jsonl', orient='records', lines=True)
        keys = ['task_id', 'optimizer', 'strength', 'representation', 'seed']
        uniform = rows[rows.design == 'uniform']
        for design in ('early', 'uniform', 'late'):
            paired = rows[rows.design == design].merge(uniform, on=keys,
                suffixes=('', '_uniform'), validate='one_to_one')
            if len(paired) != len(uniform):
                raise ValueError('unmatched timing configurations')
            if not (paired.initial_failure == paired.initial_failure_uniform).all():
                raise ValueError('timings differ before intervention')
            safe = paired[paired.initial_failure == 0].copy()
            rec = {'arm': arm, 'design': design, 'all_pairs': len(paired),
                'tasks': paired.task_id.nunique(), 'initial_failure_rate': paired.initial_failure.mean(),
                'initially_safe_pairs': len(safe), 'status': 'development_descriptive_only'}
            for name, frame, metric in [('all_task_gain', paired, 'gain'),
                                       ('initially_safe_later_onset', safe, 'later_onset')]:
                if frame.empty:
                    rec[name+'_status'] = 'unidentified_empty_cohort'
                    continue
                frame = frame.copy()
                frame['delta'] = frame[metric]-frame[metric+'_uniform']
                mean, lo, hi = task_bootstrap(frame, 'delta')
                rec.update({name+'_difference': mean, name+'_lo': lo, name+'_hi': hi})
            summaries.append(rec)
        for (task, design), g in rows.groupby(['task_id', 'design']):
            safe = g[g.initial_failure == 0]
            by_task.append({'arm': arm, 'task_id': task, 'design': design,
                'settings': len(g), 'initial_failure_rate': g.initial_failure.mean(),
                'initially_safe_settings': len(safe),
                'later_onset_rate_if_initially_safe': safe.later_onset.mean() if len(safe) else np.nan,
                'mean_final_gain_all_settings': g.gain.mean()})
    pd.DataFrame(summaries).to_csv(output/'timing_contrasts.csv', index=False)
    pd.DataFrame(by_task).to_csv(output/'timing_by_task.csv', index=False)
    onset_tasks = sorted(set().union(*(set(rows.loc[rows.later_onset == 1, 'task_id'])
        for rows in rows_by_arm.values() if not rows.empty)))
    manifest = {'bank_sha256': hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        'scope': 'development only; task-bootstrap intervals are descriptive',
        'later_onset_task_ids': onset_tasks, 'later_onset_task_count': len(onset_tasks),
        'development_gate': ('continue_to_freeze_predictor_and_cost_rule' if len(onset_tasks) >= 5
            else 'boundary_underidentified_do_not_open_heldout_to_search_for_effect'),
        'gate_status': 'development triage; does not satisfy the five-task heldout criterion',
        'all_task_count': len(baselines),
        'source_timing_unidentified_task_ids': [r['task_id'] for r in support if r['unique_sources'] < 6],
        'evaluation_only_test_note': 'released additional tests; pretraining contamination possible',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'decision.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bank', type=Path)
    parser.add_argument('draw_results', type=Path)
    parser.add_argument('source_results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.bank, args.draw_results, args.source_results, args.output)
