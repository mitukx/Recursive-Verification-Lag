# Progress XXVI — Prospective transfer exposes a timing reversal

## Outcome first

Two newly generated banks replicate the early-fixed advantage at 32 accounted
audit draws, but **reverse its safety advantage under the same six-source replay
control that had favored early verification on development tasks**. This weakens
a general early-refresh claim. No consistently best scalar predictor emerges.
All 512 newly generated candidates fail at least one hidden input, severely
limiting capability-improvement claims even where fractional reward improves.

## Three tasks completed

1. Freeze new specifications, two independent generation seeds, and the comparison
   grid before generation. Persist the protocol in commit
   5e8d44bc4d2bc5e7bb0acbfc07efa19123fe19cd. Freeze development-only predictor
   thresholds before evaluating transfer outcomes, in commit
   91a6301761a37a45c364845ee1af90017c3ef3e2. The coordinate analysis was a declared
   secondary amendment during generation, before new outcome inspection.
2. Generate 256 candidates per seed (20260923 and 20260924), 32 per each of eight
   new specifications, from Qwen2.5-Coder-0.5B-Instruct at the same immutable
   revision. Model file SHA256 matches
   f9523886352217ded3aeeef552b381af79d568c6d49a4b9e423288cea56b0a44.
   Preserve both raw banks and manifests, without outcome-based regeneration.
3. Execute 8,640 exact-paid-draw runs and 9,600 distinct-source runs; keep all
   18,240 runs, or 218,880 policy updates. These are correlated configurations,
   not independent model replications. Test frozen scalar predictors on the
   new banks without refitting. Only two generation seeds and eight tasks exist.

Protocol: `docs/transfer_v1_protocol.md`. Specifications use new combinations
of familiar operators in the same restricted grammar. They are not a pristine
external benchmark or an independent model family. Authors knew development
findings before designing the tasks; this is prospective transfer, not an
externally registered confirmatory trial. The initially failed short-timeout
weight download produced no candidate bank and was retried with longer network
timeouts; sampling settings were unchanged.

## H1: early timing helps at equal paid-draw cost

Primary contrast: early versus uniform, each with 32 accounted oracle reveals.
All settings share initial audit identities and first policy. Subsequent sampled
labels can differ. Primary analysis retains initial failures and every task.

| Contrast | Failure difference, seed 23 | Failure difference, seed 24 | Gain difference, seed 23 | Gain difference, seed 24 |
| --- | ---: | ---: | ---: | ---: |
| Early − uniform | −2.708 pp | −2.500 pp | +0.01159 | +0.00991 |
| KL 0.5 − uniform | −2.500 pp | −2.292 pp | +0.00599 | −0.00126 |

Task-cluster bootstrap after retaining both banks within each task gives early
failure difference −2.604 pp [−4.167, −1.042] and gain +0.01075
[+0.00461, +0.01759]. These are descriptive 95% intervals from 1,000 resamples,
eight task clusters, without multiplicity correction. Early's desired directions
replicate; KL's joint safety/progress criterion does not. The other seven designs
remain in the complete tables, not selected as post-hoc primary winners.
The task-level pattern is heterogeneous: digits, saturating_square, and
signed_remainder have zero early-versus-uniform failure difference in both
banks. The sign of the gain contrast changes across banks for bucket, nearest,
and round4. Thus favorable pooled means do not represent a uniform task effect.

Paid draws count repeated queries. They are not equal distinct-program coverage
or actual wall-clock costs. Trusted scores are precomputed for offline evaluation
and revealed to the loop under the counted audit interface; see
`docs/evidence_and_cost_units.md`.

## H2: early timing helps with exactly the same acquired information

Secondary controlled replay uses six distinct source identities, in the same
order and three batches of two, independent of the policy. Every task satisfies
the predefined six-source feasibility condition. Within a matched comparison,
final labels and final fitted verifier are identical; only reveal timing changes.
This is the same source-level budget/control used in Progress XXV.

| Bank | Early failure | Uniform failure | Early − uniform failure | Early − uniform gain |
| --- | ---: | ---: | ---: | ---: |
| Seed 23 | 20.21% | 19.17% | +1.042 pp | +0.00977 |
| Seed 24 | 18.75% | 14.79% | +3.958 pp | +0.00005 |

The desired safety direction fails in both new banks. Development had a negative
failure difference under this control; transfer is positive. Early exposure to an
intermediate verifier can be harmful; refreshing more promptly is not itself a
guarantee that the new score orders candidates better. This is a counterexample
to a broad empirical extrapolation, not a proof identifying the sole mechanism.
The 32-draw and six-source protocols also differ in initial audit size and total
information; their contrast is not a clean causal decomposition of those factors.
The pooled stream failure difference is +2.500 pp with a descriptive task-bootstrap
interval [−0.521, +7.083] pp. Thus the prespecified point-estimate replication
criterion fails, but these eight tasks do not establish population-level worsening
with a confidence interval excluding zero. The reversal must not be overstated.

All 2,400 soft/stream endpoint rewards match the standard composed-score-exposure
identity within 1.75e-15. That identity remains explanatory algebra, not novel
theory or a prospective safety certificate. Adaptive forced deadlines and
policy-driven acquisition results are retained separately.

## H3: frozen observable coordinates transfer as a universal predictor

Thresholds and directions use only original-development fixed-controller cap-32
trajectories before the first failure. Test rows are new-bank fixed early/uniform/
late schedules, also stopped at the first failure. Thus this tests task plus
schedule shift. The threshold is never adjusted using new outcomes.

Mean balanced accuracy across the four optimizer/representation strata:

| Coordinate | Seed 23 | Seed 24 |
| --- | ---: | ---: |
| KL | 0.6026 | 0.5912 |
| Max log-density ratio | 0.6039 | 0.6199 |
| Restricted feature geometry | 0.6006 | 0.5952 |
| Geometry / squared proxy margin | 0.5768 | 0.6830 |

Eta-times-age applies only to soft selection (two strata), with means 0.4402 and
0.4196; it is not directly comparable to the four-stratum means. No universal
winner is supported: geometry/margin changes rank across generation banks.
These are row-level balanced accuracies, not time-to-warning guarantees or
confidence intervals. One threshold family and a small development suite do not
exhaust possible predictors. All strata and counts are in coordinate_transfer.csv.

## Bank-support audit: a major limitation, not a filter

Both banks have zero perfectly correct candidates on every task. Source diversity
is at least 14 (seed 23) and 10 (seed 24); diversity does not ensure correctness.
Valid-expression fractions are 152/256 and 153/256. Only 3 and 2 candidates,
respectively, hit the 64-token cap; truncation alone does not explain failures.
Unsupported power syntax, other disallowed syntax/calls, and valid but incorrect
expressions all occur. For seed 24, saturating_square has zero trusted reward
for every candidate. It remains in the analysis as predeclared, not silently removed.

The quantitative results measure optimization among imperfect programs under
fractional finite-domain reward. They do not establish correct-program discovery,
capability creation, general coding-agent reliability, or general deployment risk.

## Decision ledger and next experiment

| Hypothesis | Result | Next decision |
| --- | --- | --- |
| Early timing helps at 32 draws | Desired directions replicate in both banks | Retain as protocol-specific evidence; not a universal rule |
| Early timing improves safety with shared labels | Reversed in both banks | Reject generalization; model intermediate-verifier errors explicitly |
| A single shift/geometry scalar wins | Ranking changes across banks | Retain failed predictors; no universal law |
| This small LM provides a capable candidate support | No fully correct candidates | Change generator capability in a separately specified study, not resample until success |

Next priority is a stronger, independently specified pretrained code model and
standard code tasks with a secure execution harness and adequate reference tests.
Keep this transfer suite as now-observed data, not a future untouched holdout.
Preregister a new task-family evaluation and include early-fixed plus equal-source
controls. More sweeps on this support cannot repair missing correct programs.
For theory, pursue an assumption-explicit residual-error envelope with useful
progress, not merely a safety rule that abstains. No new theorem is claimed here.

## Reproduction

Generate with `src.generate_candidate_bank --task-suite transfer_v1 --task-limit 8
--samples 32 --seed 20260923` (and 20260924), specifying the immutable revision
and new output paths. Run `src.exact_cost_pilot` with `--record-coordinates` and
`src.source_timing_pilot` separately on each bank. Aggregate using
`src.analyze_transfer_replication` and `src.summarize_source_transfer`; evaluate
the committed frozen JSON with `src.frozen_coordinate_transfer evaluate`.
27 unit tests pass. Full command choices are in the locked protocol and manifests.
Raw generation records are preserved; compact run archives retain all rewards,
audit indices, refresh times, and (for the 32-draw arm) observable coordinates.
