"""Locked development-only transfer of verifier refresh to unpaid new tasks.

The generator is fixed. Policy selection on unseen tasks uses only cheap
features and paid training-task labels. This is not generator self-improvement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.candidate_bank_experiment import Config, load_jsonl, fit_verifier, propose, task_policy
from src.source_audit import SourceAudit
from src.analyze_candidate_sweep import task_bootstrap


def transfer_policies(features, sources, train_ids, eval_ids, known, *, representation='all',
                      optimizer='soft', rounds=4, ridge=1e-4):
    """No evaluator rewards accepted. known[(training_task, source)] is paid."""
    if 'trusted_score' in features:
        raise ValueError('transfer must receive an unlabeled feature bank')
    train_ids, eval_ids = set(map(str, train_ids)), set(map(str, eval_ids))
    if train_ids & eval_ids or not train_ids or not eval_ids:
        raise ValueError('nonempty disjoint task sets required')
    df = features.reset_index(drop=True).copy()
    df['task_id'] = df.task_id.astype(str)
    sources = np.asarray(sources)
    if sources.shape != (len(df),) or set(df.task_id) != train_ids | eval_ids:
        raise ValueError('source alignment or task manifest mismatch')
    if (not known or any(t not in train_ids for t, g in known)
            or any(not np.isfinite(v) or not 0 <= v <= 1 for v in known.values())):
        raise ValueError('only paid training-task labels may fit verifier')
    if representation not in ('all', 'public') or optimizer not in ('soft', 'bon'):
        raise ValueError('invalid representation or optimizer')
    if isinstance(rounds, bool) or int(rounds) != rounds or rounds < 1 or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('invalid update count or ridge')
    cols = sorted(c for c in df if c.startswith('f::') and
                  (representation == 'all' or c == 'f::public_score'))
    X = np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)])
    first = {}
    for i, (task, source) in enumerate(zip(df.task_id, sources)):
        first.setdefault((task, source), i)
    if set(known)-set(first):
        raise ValueError('paid label outside bank')
    indices = [first[key] for key in known]
    theta = fit_verifier(X[indices], np.array(list(known.values())), ridge)
    scores = X @ theta
    cfg = Config(optimizer=optimizer, eta=1., best_of_n=4)
    policies = {}
    for task in sorted(eval_ids):
        ii = np.flatnonzero(df.task_id.to_numpy() == task)
        frame = df.iloc[ii].reset_index(drop=True)
        base = task_policy(frame, frame.base_logprob.to_numpy(float))
        p, public = base.copy(), base.copy()
        for _ in range(rounds):
            p = propose(frame, p, scores[ii], cfg)
            public = propose(frame, public, frame['f::public_score'].to_numpy(float), cfg)
        policies[task] = {'indices': ii.tolist(), 'policy': p.tolist(),
                          'uniform': base.tolist(), 'public': public.tolist()}
    return {'policies': policies, 'feature_columns': cols, 'theta': theta.tolist(),
            'paid_training_sources': len(known)}


def evaluate(bank_path, config_path, output):
    protocol_bytes = config_path.read_bytes()
    lock = json.loads(protocol_bytes)
    raw = [json.loads(s) for s in bank_path.read_text().splitlines() if s.strip()]
    score_manifest = json.loads(bank_path.with_suffix('.score_manifest.json').read_text())
    bank_hash = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    if (score_manifest['scored_bank_sha256'] != bank_hash or
            score_manifest['frozen_split_sha256'] != lock['split_manifest_sha256']):
        raise ValueError('scored-bank or frozen-split provenance mismatch')
    training, testing = lock['training_task_ids'], lock['evaluation_task_ids']
    if len(training) != 16 or len(testing) != 16 or set(training) & set(testing):
        raise ValueError('locked 16/16 task partition required')
    if any(r['model'] != lock['generator_model'] or r['revision'] != lock['generator_revision'] for r in raw):
        raise ValueError('generator revision mismatch')
    df = load_jsonl(bank_path)
    df['task_id'] = df.task_id.astype(str)
    if set(df.task_id) != set(training+testing):
        raise ValueError('development task IDs mismatch')
    for task, frame in df.groupby('task_id'):
        expected = {f'MBPPPlus/{task}:{i}' for i in range(lock['samples_per_task'])}
        if set(frame.candidate_id) != expected or len(frame) != lock['samples_per_task']:
            raise ValueError('locked sample count or IDs mismatch')
    source_by_id = {r['candidate_id']: r['source_sha256'] for r in raw}
    sources = np.array([source_by_id[c] for c in df.candidate_id])
    first, rewards = {}, {}
    for i, (task, g, y) in enumerate(zip(df.task_id, sources, df.trusted_score)):
        key = (task, g)
        if key in rewards and rewards[key] != y:
            raise ValueError('identical-source reward inconsistency')
        first.setdefault(key, i)
        rewards[key] = float(y)
    initial, later, accounting = [], [], []
    for task in training:
        ii = np.flatnonzero(df.task_id.to_numpy() == task)
        sampler = SourceAudit(sources[ii], lock['source_seed'], 'stream')
        order = [sampler.groups[i] for i in sampler.order]
        n0 = min(lock['initial_sources_per_training_task'], len(order))
        n = min(lock['total_sources_per_training_task'], len(order))
        initial.extend((task, g) for g in order[:n0])
        later.extend((task, g) for g in order[n0:n])
        accounting.append({'task_id': task, 'unique_sources': len(order),
                           'initial_paid_sources': n0, 'total_paid_sources': n,
                           'shortfall': lock['total_sources_per_training_task']-n})
    paid = {}
    def oracle(key):
        if key[0] not in set(training) or key in paid:
            raise AssertionError('evaluation-task query or repeated charge')
        paid[key] = rewards[key]
        return paid[key]
    initial_known = {key: oracle(key) for key in initial}
    features = df.drop(columns=['trusted_score'])
    combinations = [(rep, opt, steps) for rep in lock['secondary']['representations']
                    for opt in ('soft', 'bon') for steps in lock['secondary']['rounds']]
    frozen = {}
    for rep, opt, steps in combinations:
        frozen[(rep, opt, steps)] = transfer_policies(features, sources, training, testing,
            initial_known, representation=rep, optimizer=opt, rounds=steps, ridge=lock['ridge'])
    # The frozen arm's complete decisions exist BEFORE later labels are read.
    # Refreshed uses the same source identities, released before its decisions.
    known = {**initial_known, **{key: oracle(key) for key in later}}
    refreshed = {}
    for rep, opt, steps in combinations:
        refreshed[(rep, opt, steps)] = transfer_policies(features, sources, training, testing,
            known, representation=rep, optimizer=opt, rounds=steps, ridge=lock['ridge'])
    output.mkdir(parents=True, exist_ok=False)
    records, decisions = [], []
    # Evaluation-only outcome reads begin after all decisions for BOTH arms.
    for (rep, opt, steps), fixed in frozen.items():
        updated = refreshed[(rep, opt, steps)]
        for task in testing:
            for arm, result in [('frozen', fixed), ('refreshed', updated)]:
                policy = result['policies'][task]
                y = df.trusted_score.to_numpy(float)[policy['indices']]
                records.append({'task_id': task, 'representation': rep, 'optimizer': opt,
                    'rounds': steps, 'arm': arm, 'trusted_pass_evaluation_only': float(np.array(policy['policy']) @ y),
                    'uniform_pass_evaluation_only': float(np.array(policy['uniform']) @ y),
                    'public_policy_pass_evaluation_only': float(np.array(policy['public']) @ y),
                    'task_has_any_success_evaluation_only': int(any(y))})
                decisions.append({'task_id': task, 'representation': rep, 'optimizer': opt,
                    'rounds': steps, 'arm': arm, **policy})
    rows = pd.DataFrame(records)
    rows.to_csv(output/'evaluation.csv', index=False)
    pd.DataFrame(accounting).to_csv(output/'training_cost.csv', index=False)
    (output/'policies.json').write_text(json.dumps(decisions)+'\n')
    model_records = []
    for arm, results in [('frozen', frozen), ('refreshed', refreshed)]:
        for (rep, opt, steps), result in results.items():
            model_records.append({'arm': arm, 'representation': rep, 'optimizer': opt,
                'rounds': steps, 'theta': result['theta'], 'feature_columns': result['feature_columns'],
                'fit_source_count': result['paid_training_sources']})
    (output/'models.json').write_text(json.dumps(model_records, indent=2)+'\n')
    (output/'paid_training_transcript.json').write_text(json.dumps([
        {'task_id': t, 'source': g, 'paid_label': paid[(t, g)],
         'release': 'initial' if (t, g) in initial_known else 'after frozen decisions, before refreshed decisions'}
        for t, g in initial+later], indent=2)+'\n')
    metrics = ['trusted_pass_evaluation_only', 'uniform_pass_evaluation_only',
               'public_policy_pass_evaluation_only', 'task_has_any_success_evaluation_only']
    rows.groupby(['representation', 'optimizer', 'rounds', 'arm'])[metrics].mean().reset_index().to_csv(
        output/'summary.csv', index=False)
    contrasts = []
    for (rep, opt, steps), cohort in rows.groupby(['representation', 'optimizer', 'rounds']):
        pair = cohort[cohort.arm == 'refreshed'].merge(cohort[cohort.arm == 'frozen'],
            on='task_id', validate='one_to_one', suffixes=('', '_frozen'))
        pair['difference'] = pair.trusted_pass_evaluation_only-pair.trusted_pass_evaluation_only_frozen
        mean, lo, hi = task_bootstrap(pair, 'difference')
        contrasts.append({'representation': rep, 'optimizer': opt, 'rounds': steps,
            'primary': int(rep == lock['primary']['representation'] and opt == lock['primary']['optimizer']
                           and steps == lock['primary']['rounds']),
            'refreshed_minus_frozen': mean, 'descriptive_task_bootstrap_lo': lo,
            'descriptive_task_bootstrap_hi': hi, 'evaluation_tasks': len(pair)})
    pd.DataFrame(contrasts).to_csv(output/'contrasts.csv', index=False)
    manifest = {'status': 'locked development-only fresh-task verifier transfer; not heldout confirmation',
        'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(), 'bank_sha256': bank_hash,
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'actual_paid_training_sources_both_arms': len(paid), 'initial_fit_sources': len(initial_known),
        'label_cost_semantics': 'actual source count in replayed controller ledger, not physical study scoring cost',
        'research_bank_scoring_candidate_occurrences': len(raw),
        'research_bank_reference_validations': len(training)+len(testing),
        'stored_generated_token_positions_including_eos_padding': sum(len(r.get('generated_token_ids', [])) for r in raw),
        'decision_time_evaluation_task_trusted_queries': 0,
        'evaluator_candidate_occurrences': int(df.task_id.isin(testing).sum()),
        'training_tasks': training, 'evaluation_tasks': testing,
        'source_set_equality': True, 'frozen_decisions_before_additional_labels': True,
        'generator_parameter_updates': 0, 'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.iterdir()) if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bank', type=Path)
    parser.add_argument('--protocol', type=Path, default=Path('configs/fresh_task_verifier_transfer_v1.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.bank, args.protocol, args.output)
