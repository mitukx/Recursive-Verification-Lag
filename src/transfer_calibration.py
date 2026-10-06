"""Exploratory scale-versus-ranking diagnosis on the locked development bank.

Policy constructors accept no evaluation rewards. Equal KL is a diagnostic,
not a safety certificate or a claim of equal computational cost.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.candidate_bank_experiment import load_jsonl, fit_verifier, task_policy
from src.analyze_candidate_sweep import task_bootstrap


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_family(base, scores):
    """Canonical positive-affine-invariant scores on the baseline support."""
    p, s = np.asarray(base, float), np.asarray(scores, float)
    if (p.ndim != 1 or not len(p) or s.shape != p.shape or
            not np.isfinite(p).all() or not np.isfinite(s).all() or
            np.any(p < 0) or not np.isclose(p.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError('finite aligned scores and normalized baseline required')
    p = p/p.sum()
    support = p > 0
    span = float(s[support].max()-s[support].min())
    if not np.isfinite(span):
        raise ValueError('score range overflow')
    z = np.zeros_like(s)
    if span == 0:
        return p, z, 0.
    z[support] = (s[support]-s[support].max())/span
    maximum_mass = float(p[support & (s == s[support].max())].sum())
    return p, z, float(-np.log(maximum_mass))


def matched_kl_policy(base, scores, target):
    """Unique nonnegative exponential tilt for 0 <= KL < limiting KL.

    Exact constant-score/support cases retain the baseline. The limiting tied
    top-score distribution is not silently approximated or selected instead.
    """
    p, z, limit = score_family(base, scores)
    if not np.isfinite(target) or target < 0 or (target > 0 and target >= limit):
        raise ValueError('KL target must be nonnegative and strictly below attainable limit')
    if target == 0:
        return {'policy': p.tolist(), 'kl': 0., 'canonical_temperature': 0., 'limiting_kl': limit}
    support = p > 0
    logp = np.log(p[support])
    def tilt(beta):
        logits = logp+beta*z[support]
        logits -= logits.max()
        local = np.exp(logits); local /= local.sum()
        q = np.zeros_like(p); q[support] = local
        positive = local > 0
        kl = float(local[positive] @ (np.log(local[positive])-logp[positive]))
        return q, max(0., kl)
    lo, hi = 0., 1.
    for _ in range(100):
        if tilt(hi)[1] >= target:
            break
        hi *= 2
    else:
        raise ArithmeticError('failed to bracket KL target')
    for _ in range(112):
        beta = (lo+hi)/2
        q, kl = tilt(beta)
        if abs(kl-target) <= 1e-12:
            return {'policy': q.tolist(), 'kl': kl, 'canonical_temperature': beta, 'limiting_kl': limit}
        if kl < target:
            lo = beta
        else:
            hi = beta
    raise ArithmeticError('KL inversion did not converge')


def pairwise_disagreements(a, b, tolerance=0.):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all() or
            not np.isfinite(b).all() or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError('finite aligned ranking vectors required')
    i, j = np.triu_indices(len(a), 1)
    def relation(x):
        d = x[i]-x[j]
        return np.where(abs(d) <= tolerance, 0, np.sign(d))
    x, y = relation(a), relation(b)
    return {'pairs': len(i), 'order_or_tie_changes': int((x != y).sum()),
            'strict_reversals': int((x*y == -1).sum()),
            'tie_changes': int(((x == 0) != (y == 0)).sum())}


def build_policies(features, models, requested):
    """Build every task/arm policy before reward evaluation. No labels allowed."""
    if 'trusted_score' in features:
        raise ValueError('policy construction requires an unlabeled feature bank')
    if set(models) != {'all_frozen', 'all_refreshed', 'public_frozen', 'public_refreshed',
                       'shuffled_refreshed'}:
        raise ValueError('complete diagnostic model set required')
    if (not requested or len(set(requested)) != len(requested) or
            any(not np.isfinite(k) or k <= 0 for k in requested)):
        raise ValueError('distinct positive KL budgets required')
    decisions, rankings = [], []
    for task, frame in features.groupby('task_id', sort=True):
        base = task_policy(frame, frame.base_logprob.to_numpy(float))
        scores = {'public_score': frame['f::public_score'].to_numpy(float)}
        for arm, model in models.items():
            X = np.column_stack([np.ones(len(frame)), frame[model['feature_columns']].to_numpy(float)])
            theta = np.asarray(model['theta'], float)
            if theta.shape != (X.shape[1],):
                raise ValueError('model feature alignment mismatch')
            scores[arm] = X @ theta
        limits = {arm: score_family(base, s)[2] for arm, s in scores.items()}
        common_limit = min(v for arm, v in limits.items() if arm != 'shuffled_refreshed')
        for representation in ('all', 'public'):
            for tolerance in (0., 1e-10):
                rankings.append({'task_id': str(task), 'representation': representation,
                    'absolute_tie_tolerance': tolerance,
                    **pairwise_disagreements(scores[representation+'_frozen'],
                                             scores[representation+'_refreshed'], tolerance)})
        for budget in requested:
            common = min(budget, common_limit/2)
            for arm, s in scores.items():
                target = min(budget, limits[arm]/2) if arm == 'shuffled_refreshed' else common
                result = matched_kl_policy(base, s, target)
                decisions.append({'task_id': str(task), 'arm': arm, 'requested_kl': budget,
                    'target_kl': target, 'common_radius': arm != 'shuffled_refreshed',
                    'zero_common_radius': common == 0., 'candidate_ids': frame.candidate_id.tolist(),
                    'scores': s.tolist(), 'baseline': base.tolist(), **result})
    return decisions, rankings


def evaluate(bank_path, protocol_path, original_root, output):
    lock = json.loads(protocol_path.read_text())
    manifest = json.loads((original_root/'manifest.json').read_text())
    if (sha256(bank_path) != lock['bank_sha256'] or
            sha256(bank_path) != manifest['bank_sha256'] or
            manifest['protocol_sha256'] != lock['original_protocol_sha256']):
        raise ValueError('bank or original protocol provenance mismatch')
    for name, expected in manifest['output_sha256'].items():
        if sha256(original_root/name) != expected:
            raise ValueError('original transfer output hash mismatch: '+name)
    bank = load_jsonl(bank_path); bank['task_id'] = bank.task_id.astype(str)
    train, test = manifest['training_tasks'], manifest['evaluation_tasks']
    if set(train) & set(test) or set(bank.task_id) != set(train+test):
        raise ValueError('disjoint task partition mismatch')
    raw = [json.loads(s) for s in bank_path.read_text().splitlines() if s.strip()]
    sources = {r['candidate_id']: r['source_sha256'] for r in raw}
    first = {}
    for i, row in bank.iterrows():
        first.setdefault((row.task_id, sources[row.candidate_id]), i)
    transcript = json.loads((original_root/'paid_training_transcript.json').read_text())
    keys = [(r['task_id'], r['source']) for r in transcript]
    if len(set(keys)) != len(keys) or any(t not in train or (t, s) not in first for t, s in keys):
        raise ValueError('paid training transcript identity mismatch')
    models = {}
    recorded = json.loads((original_root/'models.json').read_text())
    for model in recorded:
        if model['rounds'] != 4 or model['optimizer'] != 'soft':
            continue
        arm = model['representation']+'_'+model['arm']
        if arm in models:
            raise ValueError('duplicate archived model')
        paid = [r for r in transcript if model['arm'] == 'refreshed' or r['release'] == 'initial']
        indices = [first[(r['task_id'], r['source'])] for r in paid]
        X = np.column_stack([np.ones(len(bank)), bank[model['feature_columns']].to_numpy(float)])
        theta = fit_verifier(X[indices], np.array([r['paid_label'] for r in paid]), 1e-4)
        if len(paid) != model['fit_source_count'] or not np.allclose(theta, model['theta'], atol=1e-9, rtol=1e-9):
            raise ValueError('original paid-transcript fit did not reproduce')
        models[arm] = model
    # A fixed label-alignment control: shuffle only paid training labels within
    # each training task, preserving counts and per-task prevalence.
    rng = np.random.default_rng(20261007)
    null_labels = np.array([r['paid_label'] for r in transcript], float)
    for task in train:
        ii = np.array([i for i, r in enumerate(transcript) if r['task_id'] == task])
        null_labels[ii] = rng.permutation(null_labels[ii])
    all_model = models['all_refreshed']
    X = np.column_stack([np.ones(len(bank)), bank[all_model['feature_columns']].to_numpy(float)])
    null_theta = fit_verifier(X[[first[key] for key in keys]], null_labels, 1e-4)
    models['shuffled_refreshed'] = {**all_model, 'theta': null_theta.tolist(),
        'label_permutation_seed': 20261007, 'scope': 'one alignment control, not a null distribution'}
    features = bank[bank.task_id.isin(test)].drop(columns=['trusted_score'])
    decisions, rankings = build_policies(features, models, lock['requested_kl_from_uniform'])
    output.mkdir(parents=True, exist_ok=False)
    # Persist complete decisions before evaluation-task outcome reads.
    (output/'policies.json').write_text(json.dumps(decisions)+'\n')
    (output/'models.json').write_text(json.dumps(models, indent=2)+'\n')
    pd.DataFrame(rankings).to_csv(output/'ranking_changes.csv', index=False)
    rewards = bank.set_index('candidate_id').trusted_score
    evaluations = []
    for decision in decisions:
        y = rewards.loc[decision['candidate_ids']].to_numpy(float)
        q, p = np.array(decision['policy']), np.array(decision['baseline'])
        evaluations.append({k: decision[k] for k in ('task_id', 'arm', 'requested_kl', 'target_kl',
                'kl', 'common_radius', 'zero_common_radius')} | {
            'trusted_pass_evaluation_only': float(q@y), 'uniform_pass_evaluation_only': float(p@y),
            'total_variation': float(abs(q-p).sum()/2),
            'distribution_free_gain_lower_bound': float(np.minimum(q-p, 0).sum()),
            'task_has_any_success_evaluation_only': int(any(y))})
    rows = pd.DataFrame(evaluations)
    rows.to_csv(output/'evaluation.csv', index=False)
    metrics = ['trusted_pass_evaluation_only', 'uniform_pass_evaluation_only', 'target_kl',
               'kl', 'total_variation', 'distribution_free_gain_lower_bound', 'zero_common_radius']
    rows.groupby(['arm', 'requested_kl'])[metrics].mean().reset_index().to_csv(output/'summary.csv', index=False)
    contrasts = []
    comparisons = [('all_refreshed', 'all_frozen'), ('public_refreshed', 'public_frozen'),
                   ('all_refreshed', 'public_score'), ('public_refreshed', 'public_score')]
    for budget in lock['requested_kl_from_uniform']:
        cohort = rows[rows.requested_kl == budget]
        for a, b in comparisons:
            pair = cohort[cohort.arm == a].merge(cohort[cohort.arm == b], on='task_id',
                                               validate='one_to_one', suffixes=('', '_control'))
            if len(pair) != len(test) or not np.allclose(pair.kl, pair.kl_control, atol=3e-12, rtol=0):
                raise AssertionError('unequal task KL in matched contrast')
            pair['difference'] = pair.trusted_pass_evaluation_only-pair.trusted_pass_evaluation_only_control
            mean, lo, hi = task_bootstrap(pair, 'difference')
            contrasts.append({'arm': a, 'control': b, 'requested_kl': budget, 'tasks': len(pair),
                'matched_kl_gain_difference': mean, 'descriptive_task_bootstrap_lo': lo,
                'descriptive_task_bootstrap_hi': hi})
    pd.DataFrame(contrasts).to_csv(output/'contrasts.csv', index=False)
    lineage = {'status': lock['status'], 'protocol_sha256': sha256(protocol_path),
        'bank_sha256': sha256(bank_path), 'original_transfer_manifest_sha256': sha256(original_root/'manifest.json'),
        'code_sha256': sha256(__file__), 'decision_time_evaluation_task_trusted_queries': 0,
        'new_physical_candidate_executions': 0, 'generator_parameter_updates': 0,
        'training_source_count': len(transcript), 'evaluation_tasks': test,
        'policies_persisted_before_evaluation': True, 'shuffled_control_common_radius': False,
        'output_sha256': {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(lineage, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bank', type=Path)
    parser.add_argument('--protocol', type=Path, default=Path('configs/transfer_calibration_diagnostic_v1.json'))
    parser.add_argument('--original', type=Path, default=Path('results/fresh_task_verifier_transfer_v1'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.bank, args.protocol, args.original, args.output)
