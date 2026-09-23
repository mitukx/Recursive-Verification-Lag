# Progress XXIV — Equal paid-cost refresh timing

## Question, hypothesis, and design

Does movement-triggered refresh prevent reward collapse at the same realized audit
cost? The previous pilot matched caps but not realized costs. This exploratory
follow-up uses the same recovered pretrained expression bank, not new model
samples or untouched tasks. Designs were fixed before this follow-up execution,
but after inspecting the earlier pilot; this is not confirmatory preregistration.

Three completed tasks: (1) add explicit schedules and exact-event constraints to
the existing loop; (2) run all nine designs at exactly 32 paid draws; (3) compare
paired failures, reward gains, unique candidate coverage and forced deadlines.
No new theorem is asserted.

Each design has 12 tasks × 2 representations × 6 optimizer/strength settings × 5
audit seeds = 720 runs; total 6,480 runs, 77,760 policy updates. Every run has four
8-draw audit events over 12 updates, including a shared initial event. Soft eta is
0.25/1/4; Best-of-N is 2/4/16. Bank, model, seeds and reward definitions match XXIII.
Initial audit indices and first policy are checked to be identical for every pair.
No initial-failure runs are dropped from the primary analysis.

Fixed schedules audit before updates (1,2,3,4), (1,4,7,10), or (1,10,11,12).
Adaptive KL thresholds are 0.1/0.5, max-log thresholds 0.5/1.5, and geometry
thresholds 10/100. These are the previous pilot thresholds, not newly tuned
winners. An adaptive design must spend remaining audit events when the number of
remaining rounds equals its unspent events. Thus it is a movement-triggered rule
with mandatory deadlines, not the original unconstrained cost-saving controller.
Updates continue after the budget is exhausted. Audit targets follow current
policy, with replacement; all labels, including repeats, count as paid draws.

## Results

All designs spend exactly 32 draws. Means below weight tasks equally. Gain is final
trusted reward minus its initial value. Failure means any update below the initial
reward, not deployment failure probability. No design is selected on these outcomes
and then presented as held-out performance.

| Design | Failure rate | Mean gain | Distinct candidate labels |
| --- | ---: | ---: | ---: |
| Early fixed | 9.03% | 0.37114 | 12.17 |
| Uniform fixed | 9.58% | 0.36046 | 11.04 |
| Late fixed | 9.58% | 0.32458 | 10.63 |
| KL 0.1 | 8.47% | 0.34729 | 12.09 |
| KL 0.5 | 8.19% | 0.34259 | 11.68 |
| Max-log 0.5 | 8.75% | 0.34446 | 11.97 |
| Max-log 1.5 | 8.47% | 0.34071 | 11.25 |
| Geometry 10 | 9.03% | 0.35444 | 11.63 |
| Geometry 100 | 9.72% | 0.34149 | 10.99 |

Paired task-bootstrap differences versus uniform, with descriptive 95% percentile
intervals (12 tasks, 1,000 resamples, no multiplicity adjustment):
- KL 0.5: failure −1.389 percentage points [−3.056, −0.278]; gain −0.01787
  [−0.04276, −0.00037]. Fewer failures accompany less progress.
- Early fixed: failure −0.556 points [−1.250, 0.000]; gain +0.01068
  [+0.00192, +0.02306]. Earlier verification also sees 1.136 more distinct
  candidate indices on average. This is consistent with a coverage explanation,
  not proof of the mechanism.
- Late fixed: gain −0.03588 [−0.06310, −0.01281], without lower mean failure.

Adaptive designs average 1.26–2.56 forced deadline events out of three post-initial
events. These interventions materially affect their behavior and preclude attributing
all performance differences to the movement trigger alone.

A common initially-safe post-hoc cohort retains 689 paired configurations per
design. Task-weighted failure is 6.00% for uniform and 4.44% for KL 0.5, but its
mean-gain difference remains negative (−0.02301). The cohort uses offline trusted
outcomes and is not an observable deployment filter. Full results for both cohorts
are retained in summary.csv; no significance-based row filtering was used.

## Falsification and next decision

H1 (timing is irrelevant at fixed draw cost) is challenged by different gains, but
causal attribution specifically to error geometry remains unresolved: policy
concentration changes what is audited. H2 (adaptive refresh dominates fixed
schedules in safety and useful progress) is not supported by this sweep. H3 (late
verification loses informative coverage) is consistent with distinct-index counts,
but these counts do not measure distinct source programs or information directly.

Exactly equal paid draws are now established. Equal informative labeling cost is
NOT: repeated deterministic labels can be cached, duplicate programs have separate
sample indices, and adaptive sampling changes coverage. The next intervention
should separate cached labels from new oracle calls and compare audit acquisition
rules (current-policy versus coverage-seeking) under a shared initial audit. Then
replicate on independent generation seeds and untouched task/model families.
Do not promote this small, reused development bank to a universal refresh law.

## Reproduction and checks

```bash
python -m src.exact_cost_pilot data/recovered_qwen05b_bank.jsonl --output data/exact_cost_new
python -m unittest discover -s tests -q
```

22 tests passed. Tests check exact cost/event count, shared initial state, explicit
schedule realization, never-trigger deadline behavior and invalid designs. All
6,480 runs retain audit indices, refresh rounds and all 12 rewards in local
`data/exact_cost_qwen05b/runs.jsonl.gz`; the manifest includes source, bank and output
SHA256 hashes. The compressed JSONL can be read with gzip.open or pandas.read_json
(lines=True). Full policy vectors are reproducible, not included in this archive.

The initial GitHub upload was not executed because automatic approval review
reached its usage limit. Persistence was recovered subsequently: compact run
records are committed in e254285cc16f907ca11ea0049845f01aff0fb596. Execution-source
snapshots retain the source versions hashed in the original manifest.
