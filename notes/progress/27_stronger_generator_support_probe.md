# Stronger-generator support probe: locked before generation

## Hypothesis and decision rule

The frozen Qwen2.5-Coder-0.5B transfer banks contain zero fully hidden-correct
expressions. A stronger pretrained generator may restore support without changing
the public/hidden split. Before inspecting new outputs, freeze this descriptive
pilot: Qwen/Qwen2.5-Coder-1.5B-Instruct, immutable revision
`2e1fd397ee46e1388853d2af2c993145b0f1098a`, transfer_v1 tasks in their
existing order, 16 samples per task, seed 20260925, temperature 0.8, top-p
0.95, 64 generated tokens, batch size 8, deterministic task evaluator.

Primary diagnostic: number of eight tasks having at least one candidate with
trusted_score = 1. The pragmatic proceed threshold is four tasks; it is a
feasibility decision, not a statistical significance test or a claim about
recursive verification lag. Also record expression validity, public-perfect
hidden-imperfect cases and score dispersion. Keep all generations, including
duplicates, failures and truncated outputs; do not select prompts or seeds
after observing outcomes. A failed threshold remains a reported negative result.

These eight tasks and their evaluation were already inspected using the 0.5B
model. Thus the probe cannot be an untouched validation set. Any later RVL
transfer claims need a newly frozen task family, independent generation seeds,
and preregistered controller/boundary decisions. The expression grammar is
restricted; results cannot stand in for general Python program synthesis.

## Observed result and integrity checks

The complete 128-row bank has SHA256
`35427719673e443eb0091b8788e302122d1b27368f96bd93d5c67381180e5a9c`.
Its completed manifest matches the file hash; independent deterministic
rescoring passed on every row. Generation took 425 seconds on CPU. Exactly 80
of 128 expressions passed the restricted parser, and one completion reached the
64-token cap. There were 18 fully trusted-correct candidate occurrences on
five of eight tasks: divisor_code 1, fold 9, nearest 1, round4 1,
saturating_square 6. Bucket, digits, and signed_remainder had zero. Seven
public-perfect candidates failed trusted evaluation. The predefined four-task
support threshold was met, without implying a new task-family replication.
The two previously frozen 0.5B banks had zero correct candidates on all eight
tasks at 32 rather than 16 samples per task. This is a model-and-seed comparison,
not a controlled model-only causal effect; no seed was chosen after inspection.

## Exploratory recursive-loop reuse

After inspecting support, run the **existing**, unchanged experiment scripts
on this now-observed bank. This is post-support exploratory analysis, not a
prespecified transfer test. Both scripts completed all eight tasks. Exact
paid-draw arm: 4,320 configurations (eight tasks × six strengths × two
representations × five audit seeds × nine designs), 32 draws per run. Early
versus uniform yields −0.833 percentage points failure difference, task-cluster
95% descriptive interval [−2.083, 0], and +0.01861 gain [0.00636, 0.03338].
The average uniform failure rate is 4.375%. Distinct-source replay: 4,800
configurations (two acquisition modes and five designs), six sources per run.
With a **matched source stream**, early versus uniform yields −1.042 percentage
points failure difference [−2.917, 0.625] and +0.04053 gain [0.01480,
0.08106]; uniform failure rate 8.75%. Under policy-driven source acquisition,
early failure difference is −1.667 points [−4.792, 1.042] and gain +0.03424
[−0.00117, 0.07196]. Intervals resample the eight task clusters and are
descriptive; correlated settings and one generation seed remain. The favorable
point estimate under shared labels reverses the adverse sign seen in two 0.5B
new banks, so refresh timing depends on generator-induced candidate support and
verifier geometry. The design does **not** isolate these as causal factors.
Neither early schedule supplies a prospective guarantee. All results include
the three tasks without a correct candidate.

## Decision

The capability gate passes, enabling a separately frozen unseen-task evaluation
with this stronger generator. Preserve the negative 0.5B support result and the
opposite source-timing signs. Do not refit a universal boundary on these outcomes.
The next standard-Python benchmark requires the execution isolation and locked
test protocol in `docs/standard_code_validation_gate.md`; no such experiment has
yet been run. No new theorem is established by this probe.
