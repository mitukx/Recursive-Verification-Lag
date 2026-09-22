# Progress XXIII — Auditable pretrained expression pilot

Status: completed exploratory pilot, **not** confirmatory real-world agent validation.
Execution: https://github.com/mitukx/Recursive-Verification-Lag/actions/runs/35705123859
Bank SHA256: `012adf92b2032a56e5f74d4e08d89e639a921e26405238e18624276e7575a327`
Model revision: `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`

## Scope and the three tasks

1. Generated 192 sample occurrences from the immutable pretrained
Qwen2.5-Coder-0.5B-Instruct checkpoint on 12 finite-domain tasks. Public inputs are
(0,1,2,3); reward is exact pass fraction on the 29 disjoint hidden inputs. Generated
text is evaluated by an explicit bounded expression interpreter. LM weights are
frozen; recursive selection changes only the bank distribution.
2. Executed 14,400 task/configuration/audit-seed runs and archived all
trajectories. Factors: soft eta (.25,1,4), BoN N (2,4,16), four fixed cadences,
three adaptive families, two representations, two nominal audit caps, five seeds.
These runs share one bank and are not independent model-generation replications.
3. Evaluated transfer, refresh cost/progress, and a shared initially-safe cadence
cohort. Proved only the conditional robust bounds and elementary identities in
`docs/observable_refresh_frontier.md`; no optimal adaptive frontier is claimed.

## H1: does proxy optimization degrade trusted reward?

Across the declared grid, 1,197/14,400
runs (8.31%) crossed below their initial reward.
936 declined by more than .01 and
668 by more than .05. Worst change was
-0.244013; mean final gain across the grid was
0.315918. This grid frequency is not a deployment probability.

620 first crossings occurred in round one;
299 occurred with verifier age > 1.
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

| axis | metric | mean | count |
| --- | --- | --- | --- |
| optimizer | KL | 0.3762 | 2 |
| optimizer | geometry_over_margin2 | 0.6693 | 2 |
| optimizer | max_log_ratio | 0.6076 | 2 |
| optimizer | restricted_geometry | 0.3723 | 2 |
| representation | KL | 0.6919 | 2 |
| representation | eta_times_age | 0.4617 | 2 |
| representation | geometry_over_margin2 | 0.6604 | 2 |
| representation | max_log_ratio | 0.6779 | 2 |
| representation | restricted_geometry | 0.6758 | 2 |
| task_id | KL | 0.6816 | 5 |
| task_id | eta_times_age | 0.4496 | 4 |
| task_id | geometry_over_margin2 | 0.5454 | 5 |
| task_id | max_log_ratio | 0.6384 | 5 |
| task_id | restricted_geometry | 0.6900 | 5 |

Do not select a universal winner from inconsistent task/optimizer/representation
rankings. The feature geometry is an observable heuristic, not a validated bound
on arbitrary misspecification; oracle error contrasts are never controller inputs.

## H3: does adaptive refresh improve the cost/progress frontier?

The table compares leave-one-task-out selected adaptive and fixed schedules.
Differences are adaptive minus fixed. Failure-difference intervals use paired
bootstrap resampling of tasks. Caps are matched; actual draw counts are not.

| budget_cap | failure_difference | failure_difference_lo | failure_difference_hi | gain_difference | cost_difference | adaptive_mean_cost | fixed_mean_cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | -0.0083 | -0.0319 | 0.0056 | -0.0357 | -4.7333 | 19.8778 | 24.6111 |
| 96 | -0.0111 | -0.0306 | 0.0000 | -0.0418 | -36.9778 | 33.4111 | 70.3889 |

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

| stale_cadence | comparisons | fresh_failure | stale_failure | fresh_pair_rate | stale_pair_rate | fresh_safe_stale_fails | fresh_fails_stale_safe | fresh_draws | stale_draws | failure_difference | failure_lo | failure_hi | gain_difference | gain_lo | gain_hi | cost_difference | cost_lo | cost_hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 689 | 0.0412 | 0.0537 | 0.0377 | 0.0493 | 9 | 1 | 96.0000 | 48.0000 | 0.0125 | 0.0016 | 0.0300 | -0.0104 | -0.0224 | -0.0020 | -48.0000 | -48.0000 | -48.0000 |
| 4 | 689 | 0.0412 | 0.0665 | 0.0377 | 0.0610 | 16 | 0 | 96.0000 | 24.0000 | 0.0253 | 0.0030 | 0.0636 | -0.0277 | -0.0547 | -0.0068 | -72.0000 | -72.0000 | -72.0000 |
| 12 | 689 | 0.0412 | 0.0524 | 0.0377 | 0.0479 | 16 | 9 | 96.0000 | 8.0000 | 0.0112 | -0.0253 | 0.0559 | -0.1022 | -0.2176 | -0.0203 | -88.0000 | -88.0000 | -88.0000 |

## Bank quality and falsifiability

Valid-expression fraction: 0.6615.
Tasks with no fully correct candidate: 5/12.
All invalid outputs and duplicates remain in the denominator; no hand-written
exploit was inserted to make the primary bank collapse. These limitations restrict
claims even when a failure is observed.

| task_id | candidates | distinct_sources | valid_fraction | initial_public | initial_hidden | fully_correct_candidates | public_perfect_hidden_imperfect | public_frontier_count | public_frontier_hidden | trusted_score_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abs | 16 | 6 | 0.7500 | 0.5000 | 0.5151 | 8 | 0 | 8 | 1.0000 | 0.5014 |
| ceil3 | 16 | 13 | 0.7500 | 0.2812 | 0.1767 | 0 | 0 | 1 | 0.0000 | 0.2385 |
| clip | 16 | 15 | 0.3125 | 0.0469 | 0.0065 | 0 | 0 | 1 | 0.0000 | 0.0139 |
| distance | 16 | 10 | 0.8750 | 0.2031 | 0.0388 | 0 | 0 | 1 | 0.5517 | 0.1373 |
| multiple | 16 | 10 | 0.8125 | 0.7344 | 0.7371 | 10 | 0 | 10 | 1.0000 | 0.4001 |
| outside | 16 | 14 | 0.8750 | 0.3438 | 0.2134 | 1 | 2 | 3 | 0.3793 | 0.2951 |
| parity | 16 | 10 | 0.8125 | 0.5312 | 0.5323 | 8 | 0 | 8 | 1.0000 | 0.4989 |
| piece | 16 | 8 | 0.1250 | 0.1250 | 0.1250 | 2 | 0 | 2 | 1.0000 | 0.3416 |
| sign | 16 | 15 | 0.5625 | 0.2969 | 0.2478 | 3 | 1 | 4 | 0.8621 | 0.3946 |
| triangle | 16 | 12 | 0.8125 | 0.4844 | 0.3578 | 5 | 0 | 5 | 1.0000 | 0.4614 |
| trunc3 | 16 | 11 | 0.3750 | 0.2500 | 0.1250 | 0 | 3 | 3 | 0.6207 | 0.2470 |
| wrap | 16 | 13 | 0.8750 | 0.0781 | 0.0776 | 0 | 0 | 5 | 0.2000 | 0.1042 |

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

Persistence: the bank, manifests and summary tables are committed. Full trajectories
and per-run/pair tables remain in Actions artifact 10684256908 (90-day retention,
expiry 2026-12-21); hashes and provenance are in
`data/recovered_qwen05b/artifact_manifest.json`. The initial-safe summary was
recomputed after the workflow to label task-weighted and pair-weighted rates
separately; the raw trajectories were unchanged.
