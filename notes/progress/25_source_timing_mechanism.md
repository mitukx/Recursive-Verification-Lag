# Progress XXV — Timing, label identity, and endogenous acquisition

Status: 7,200 completed runs on the already inspected recovered Qwen expression
bank. No new pretrained generations, untouched tasks or model-weight updates.
Code and results are local; GitHub upload is pending a usage-limit recovery.

## Hypotheses and interventions

H1: timing effects survive control of label identity, order and oracle-call count.
H2: movement-triggered refresh improves both safety and progress over fixed timing.
H3: endogenous acquisition changes the refresh outcome even with unique-source
cost fixed. All choices below were set before this follow-up's outcomes, after
observing XXIV; this remains exploratory rather than confirmatory preregistration.

The existing harness now accepts a label-blind acquisition callback. Duplicate
source hashes are cached: a source is queried at most once. Candidate occurrence
weights are unchanged. Each run spends exactly six source-level oracle calls in
three batches of two, with a common first batch before the first update. The
minimum source count across these tasks is six; that support limitation motivates
the small budget, not outcome-based task selection. Source strings are not semantic
equivalence classes: different strings can compute the same function.

The exogenous stream condition draws a uniform random permutation of source groups
and reveals the same prefix to every timing rule. Initial policy and audit indices
are asserted identical, as is the final six-label sequence. Thus final verifiers
are identical within matched stream comparisons. Fixed reveals occur at rounds
(1,2,3), (1,5,9), or (1,11,12). KL 0.5 and geometry 10 choose times subject to a
mandatory deadline that spends the remaining calls by round 12.

The policy-acquisition condition shares the initial batch, then selects unqueried
source groups proportional to current policy mass with 5% uniform exploration.
This affects acquisition only; it does not inject policy support. No hidden rewards
are given to the acquisition callback. This is a separate factor, not a substitute
for the exogenous timing control. Five seeds, both representations, soft eta
0.25/1/4 and BoN 2/4/16 yield 720 paired configurations per design and acquisition.

## Primary results: common exogenous label stream

All costs equal six distinct-source oracle calls. Failure is ever falling below
initial trusted reward; gain is final minus initial reward. Rates are grid averages
on this bank, not deployment probabilities. Task means have equal weights.

| Timing | Failure rate | Mean gain |
| --- | ---: | ---: |
| Early fixed | 12.36% | 0.34183 |
| Uniform fixed | 14.31% | 0.30768 |
| Late fixed | 13.06% | 0.26024 |
| KL 0.5 + deadline | 11.67% | 0.27937 |
| Geometry 10 + deadline | 12.36% | 0.29485 |

Paired differences from uniform, with descriptive task-bootstrap 95% intervals
(12 clusters, 1,000 draws, no multiple-comparison adjustment):
- Early: failure −1.944 percentage points [−4.167, −0.417]; gain +0.03415
  [+0.01703, +0.05431]. Same information, earlier revelation gives better observed
  average safety and progress on this development bank.
- KL: failure −2.639 points [−7.361, +0.556]; gain −0.02831
  [−0.06743, +0.00844]. No dominance is established.
- Geometry: failure −1.944 points [−4.167, −0.417]; gain −0.01283
  [−0.05081, +0.02239]. Same failure rate as early timing, lower point-estimate gain.
- Late: gain −0.04744 [−0.08140, −0.02045]. Failure is not monotonic in delay.

The common initially-safe post-hoc subset retains 660 of 720 configurations. Early
versus uniform still has failure difference −2.433 points and gain +0.02457.
This subset uses offline outcomes and is not a deployable selection rule. It does
not replace the complete-cohort primary results.

## Acquisition contrast

At early timing, policy-based acquisition versus the fixed stream increases mean
failure by 2.778 points [0.278, 5.698], while gain difference is +0.01745
[−0.01279, +0.05693]. Under uniform timing, the corresponding failure difference
is +1.528 points [−1.389, +5.000]. These are exploratory interactions, not evidence
that one acquisition method universally dominates. All timing and acquisition
rows, including contrary results, remain in the archived tables.

The adaptive rules average about 1.27–1.45 forced deadlines out of two post-initial
audit events in the stream condition. They are heavily affected by the deadline
constraint, so their results cannot be attributed solely to movement thresholds.

## Mechanism check and falsification

For all 1,800 soft-selection stream runs, the endpoint reward reconstructed using
p_T proportional to p_0 exp(eta sum_k L_k v_k) agrees with the recorded endpoint
within 1.34e-15. The final verifier is shared but intermediate score exposure
changes with revelation time. This standard identity is an alternative explanation
that the RVL story must incorporate, not conceal. It does not itself predict which
unobserved errors matter or when a new label is necessary.

H1 is supported within this restricted experiment. H2 remains unsupported as a
joint safety/progress claim. H3 has suggestive task-dependent evidence requiring
replication. No universal movement scalar or novel minimax frontier follows.

The focused prior-art check found HackProbe (arXiv:2609.04665v1) directly addressing
reward-hacking detection/intervention in self-evolving models, alongside reward
model overoptimization and adversarial reward auditing. See
`docs/refresh_timing_identification.md` for primary sources and claim boundaries.
A generic detection-and-refresh narrative is insufficient novelty.

## Next decision and rejection criteria

Freeze the present heuristics. The next substantive test must use new task families
and independent pretrained generation seeds, with adequate correct/incorrect
candidate support declared as diagnostics rather than a post-hoc inclusion filter.
Run the same exogenous revelation control and source-cost accounting there. Reject
a general timing advantage if it reverses or disappears; retain the reversal.

For theory, require a learnable residual envelope and nontrivial progress target,
then bound the information needed for proposal-relevant error contrasts. A safety
rule that simply abstains, or the existing exponential-composition identity,
is not the required new theorem. Compare against early fixed verification, not
only the weaker uniform baseline. Benchmark interfaces for standard code tasks
remain outstanding; these restricted expressions cannot stand in for them.

## Reproduction and archive

```bash
python -m src.source_timing_pilot data/recovered_qwen05b_bank.jsonl --output data/source_timing_new
python -m src.check_score_exposure data/recovered_qwen05b_bank.jsonl data/source_timing_new
python -m unittest discover -s tests -q
```

24 tests passed. Every run archives refresh rounds, source-representative candidate
indices and all 12 rewards in `data/source_timing_qwen05b/runs.jsonl.gz`. The
manifest records bank, execution-source and output hashes. Full policy vectors
are reproducible but not in this compact archive. `summary.csv` includes both
cohorts; `acquisition_effect.csv` records within-timing acquisition contrasts.
No GitHub persistence is claimed for this follow-up: the prior write failed when
automatic approval review could not complete because the usage limit was reached.
