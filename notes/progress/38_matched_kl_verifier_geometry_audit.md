# Progress 38 — Matched-KL verifier geometry audit

Recorded 2026-10-07 (Asia/Tokyo). This is a post-hoc audit of the already
inspected development archive. It creates no new policy decisions, uses no new
evaluation-task queries, performs no new candidate execution, and does not open
heldout data.

## Why this audit was necessary

The archived ranking table contradicted an earlier prose statement. The public
representation has zero frozen/refreshed pairwise changes, but the all-feature
representation has 20 strict reversals across three tasks: 11 on task 306, 8 on
task 410, and 1 on task 631. The progress note and README were corrected rather
than silently preserving the stronger claim.

## Main result

At matched KL, frozen and refreshed all-feature policies differ on 14 of 16
development evaluation tasks at every requested budget. Twelve of those fourteen
tasks have no strict ranking reversal. Therefore rank change is neither
necessary nor sufficient to explain the observed policy-path change.

To isolate score geometry, each score vector is canonicalized on baseline
support by subtracting its supported maximum and dividing by its supported
range. This removes positive affine transformations, which generate the same
exponential-tilt path. The resulting weighted RMS displacement is compared
against frozen/refreshed policy total variation.

Across nonzero-KL task rows:

- Spearman(canonical score displacement, policy TV) = 0.7720
- Spearman(strict reversal count, policy TV) = 0.2976
- Spearman(canonical score displacement, |trusted reward delta|) = 0.3847

The mean all-feature frozen/refreshed policy TV rises from 0.00283 at requested
KL 0.02 to 0.00866 at requested KL 0.6. Mean trusted reward deltas remain
negative over the four budgets: -0.0001535, -0.0003473, -0.0006317, and
-0.0008096.

These correlations are descriptive on a small, previously inspected development
archive. They are not a causal identification, a generalization theorem, a
heldout result, or a safety certificate.

## Interpretation

The useful object is more specific than "ranking drift." A verifier refresh can
leave ordering mostly unchanged while changing relative score spacing. Under an
exponential optimizer, those spacing changes alter the path through policy
space even after matching total KL displacement. This is directly consistent
with the broader RVL thesis that verification difficulty depends on policy
movement relative to verifier-error geometry, not on elapsed rounds or any one
scalar shift measure.

The public-only arm is the positive control: frozen and refreshed fits differ
only by positive affine calibration, so matched-KL policies coincide exactly.
The all-feature arm is not affine-equivalent, and its within-ranking geometry
changes measurably.

## Repository changes

- `src/analyze_transfer_geometry.py`: hash-checked archived geometry analysis.
- `tests/test_transfer_geometry.py`: regression tests for affine invariance,
  policy metrics, 3 tasks / 20 reversals, 14/16 policy changes, and 12/16
  no-reversal policy changes.
- `results/transfer_geometry_v1/`: frozen summary, correlations, and scope
  manifest.
- corrected `notes/progress/37_kl_matched_transfer_diagnostic.md` and README.

## Next experiment

The next research-grade increment should leave the development archive and move
to independent evidence: execute the already locked real-model RLVR pilot, then
add an intervention in which verifier/evaluator geometry is deliberately varied
at controlled policy KL. The key prospective prediction is that equal policy KL
with different verifier-error alignment will produce materially different false
progress rates.
