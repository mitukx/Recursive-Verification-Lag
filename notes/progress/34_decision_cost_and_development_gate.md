# Progress 34 — A decision-cost frontier, a negative planner result, and development execution

This continuation uses PR #3's latest source (`16641457d5b3629313b618b07415e57711cd6537`).
The already scored MBPP+ pilot was available when this acquisition study was
designed; every new number here is **retrospective**, not an independent test.
The fixed gain floor is .01. The existing frozen 64-task split is untouched.

## Completed result

The sharp box interval yields a realized unit-cost certificate-size formula:
sort lower-bound improvements to accept, upper-bound improvements to reject.
The evaluator computes this hindsight lower bound; it never reaches query
selection. The exact binary planning reference maximizes decision resolution
probability under a deliberately uncalibrated independent .5 prior, conditional
on a frozen candidate. Its recursion is a standard stochastic threshold
evaluation reduction, not new optimal-refresh theory.

On all 480 eight-task fixed-comparison settings, the planner and impact order
have the same resolution and accepted gain. Their cost is equal except at
the one-additional-label cap: the planner abstains on impossible decisions,
using .27708 versus .46875 calls at the same 72.71% resolution. Both
resolve 470/480 at the two-additional-label cap, versus 343/480 for the source
stream. At cap six every method resolves; mean additional labels are .8000
for planner/impact versus 1.36042 for stream, with hindsight minimum .73125.
All three begin with the same two paid initial source labels per setting.

In twelve-round loops with a total cap of four source labels, impact order has
mean final gain .128905, planner .120140 and source stream .054650. Actual
mean labels are 2.8104, 2.7875 and 2.8833, so caps do not match actual cost. At cap six
the figures are .135978, .132497 and .122221. Exact current-decision optimality
under a fictional prior does not optimize future adaptive verifier refits.
The complex planner's recursive progress superiority is unsupported;
preserve both its abstention savings and this negative progress result.

## Reproducibility and implementation boundary

Full rows/transcripts, per-task tables, paired task-bootstrap descriptive
contrasts and artifact/code hashes are archived under
`results/decision_audit_mbppplus_v1/`. The three new modules implement
decision audit, frozen-proposal recursive gates and full retrospective
evaluation. Exhaustive tests cover binary completions, fractional-reward
witness minimality, all three query strategies, prior mismatch, exact paid
cost, duplicates and unqueried-outcome noninterference. These tests establish
implementation behavior, not novel scientific results.

Two correctness repairs protect the next run: source timing skips and reports
tasks with fewer than six unique texts without replacing them, and Docker
infrastructure/malformed-reply errors cannot become zero trusted labels.
Declared candidate resource/time-limit failures remain counted and recorded.

## Next execution and decision

The development-only workflow generates the prelocked 32 tasks × 16 samples
using Qwen2.5-Coder-1.5B, preserves an unscored bank, then scores in isolated
containers and runs the existing fixed-cost/fixed-information timing analyses.
It adds onset-focused all-task summaries and an explicit development triage.
No heldout outcomes are opened. Submitting the workflow is not a result; model
download, generation or scorer failures must be reported with their logs.

The research priority is to establish a transferable endogenous failure regime,
then evaluate **fresh-support progress** at matched generation and verification
costs. Elaborating one finite-bank acquisition rule has little headroom in the
observed pilot. Full rationale and commands:
`docs/frontier_research_upgrade.md`. No production-scale research claim
is made by this increment.
