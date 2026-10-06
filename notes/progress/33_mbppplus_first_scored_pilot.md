# Progress 33 — First isolated scored MBPP+ program pilot

## Hypothesis and locked design

The original Qwen2.5-Coder-1.5B-Instruct 64-program bank was generated and
committed unscored. The exact-cost and source-timing early/uniform comparisons
in `docs/mbppplus_recursive_loop_lock.md` were fixed before candidate
scoring. The recursive optimizer reweights **one frozen bank**; the language
model's parameters are not updated. We asked whether earlier verification
reduces loss at matched paid label cost. Safe projection was a separately
marked secondary amendment fixed before MBPP+ candidate outcomes were read.

## Isolated execution and provenance

[GitHub Actions run 35868084266](https://github.com/mitukx/Recursive-Verification-Lag/actions/runs/35868084266)
verified all eight official reference solutions against the original public
assertions and released EvalPlus additional test programs **before** scoring
any model candidates. It then used one disposable, unprivileged `linux/amd64`
Docker container per program, without network or host mounts and with CPU,
memory, process, output and wall limits. All 64 candidates returned normal
scoring statuses; zero infrastructure failures were recorded. The downloaded
ZIP SHA256 matched GitHub's artifact digest. The image ID, dataset hash,
input and scored-bank hashes, and scorer/worker hashes are recorded in
`data/mbppplus_qwen15b_pilot_v1_scored.score_manifest.json` and were checked
locally. Additional tests are **released**, not a secret pretraining holdout.

Public assertions yield a positive score for 37/64 candidates; 28/64 pass
the additional test program. Three of eight tasks have zero trusted-success
support. Task 754 has 7/8 candidates passing at least one public assertion
but 0/8 passing additional tests: this is a public/trusted gap, though no
policy on this bank can fall below its zero trusted-reward baseline. See
`results/mbppplus_scored_task_support_v1.csv` for all task counts.

## Locked results and limitations

| Matched oracle schedule | Runs | Failures: early / uniform | Early minus uniform final gain | Descriptive task-bootstrap 95% | Cost |
| --- | ---: | ---: | ---: | --- | --- |
| Four 8-draw batches | 480 | 19 / 18 | +.00415 | [−.00502,+.01986] | 32 paid draws |
| Three 2-source batches, same ordered stream | 480 | 20 / 19 | +.01527 | [+.00024,+.04058] | 6 distinct sources |

In the draw-based arm, 18/480 failures occurred in round one in **both**
timings, before the timing intervention can explain them. Its only additional
early failure is task 733, Best-of-16, all features, seed 4. In the matched
six-source stream, early has two failures after round one and uniform has
one. These correlated runs comprise only eight selected tasks; bootstrap
intervals are descriptive, not population inference. Early refresh does not
improve failure incidence here; the bank does **not** exhibit a substantial
timing-driven collapse.

The six-source certified controller uses 160 matched settings, averages
2.825 paid distinct-source labels, has mean final gain .14480 and zero
baseline failures by its finite-bank certificate. **Projection has exactly
the same trajectories and outcomes:** there are zero budget-exhausted
uncertifiable proposals, so it never fires. The matched fixed six-source
uniform schedule has mean gain .14409 and 6/160 baseline failures while
spending all six labels. Actual label costs and later label identities
differ, so these figures do not establish a fair equal-cost superiority.

The public-only residual metric has contradictory zero-distance pairs in
2/8 tasks; the public+syntactic representation has no exact collision in
this bank, but fitting a Lipschitz constant from these outcomes would be
post-hoc. See `results/mbppplus_residual_metric_collisions_v1.csv`.

## Decision and falsifiability

This is a **negative transfer result** for the strong expression-bank lag
and the expression-bank projection advantage. It is real pretrained-model
code on a standard released benchmark, but the eight tasks and eight
candidate occurrences per task provide little room for a robust boundary;
three tasks have no successful candidate at all. No learned capability
creation, general safety certificate, or Oral/Spotlight-level result follows.

Freeze a larger task list and independent candidate draws *before further
scoring*, retain all tasks including all-zero ones, and report support
strata without outcome-based exclusion. Test matched actual paid labels and
distinct-source costs separately. If the expanded bank remains similarly
benign, narrow RVL to a conditional theory/negative result rather than
presenting the expression-bank transition as universal.

## Reproduction

The scored bank and manifest are committed. Run the locked scripts on
`data/mbppplus_qwen15b_pilot_v1_scored.jsonl`:
`python -m src.exact_cost_pilot ... --record-coordinates`,
`python -m src.source_timing_pilot ...`, and
`python -m src.certified_refresh ... --fallback abstain/project`.
Their complete command lines and parameters are in
`docs/mbppplus_recursive_loop_lock.md` and the scripts' CLI help.
