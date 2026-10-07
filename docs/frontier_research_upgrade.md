# Research upgrade: identify the binding bottleneck before claiming progress

This continuation starts from draft PR #3 (`research/rvl-transfer-v1`), not
the older `main` branch. It adds an exact decision-cost diagnostic and a
development-only standard-code experiment. It does not claim a finished
production-scale result or external adoption.

**Latest increment:** `notes/progress/35_endogenous_auditing_and_fresh_task_lock.md`
and `notes/theory/endogenous_information_and_progress.md` add an endogenous
learning-trap proof, an exact small future-aware control reference, and strong
one-query-exploration / direct-known-positive baselines. On the full eight-task
pilot, the single-query baseline reproduces 99.36% of the primary lookahead's
mean gain increment, and direct selection wins at lower mean actual label cost.
This negative comparison motivates a pre-outcome 16-training / 16-evaluation
task verifier-transfer lock on the pending development bank. The older
fixed-comparison results below remain valid; neither retrospective study is
independent transfer or learned generator improvement.

## What would be a substantive contribution?

The candidate research question is:

> Under a fixed trusted-verification and generation budget, can a decision
> rule predict when verifier reuse ceases to support useful policy progress,
> and spend verification so that progress transfers to newly generated support?

Three separable claims need distinct evidence:

| Claim | Current support | Required evidence |
| --- | --- | --- |
| Gain cannot be identified from a transcript that leaves relevant reward directions unresolved. | Exact finite-bank box bound and certificate-size characterization. | Endogenous-trajectory lower/upper bounds beyond a classical single-decision reduction. |
| An intervention spends trusted verification more effectively. | Retrospective acquisition comparison on eight previously scored tasks. | New task/model banks, matched caps and actual costs, calibrated development-frozen rules. |
| Verifier/policy coevolution improves capability. | None. Existing experiments only redistribute probability within a frozen bank. | A separately locked new-generation or parameter-update experiment, evaluated on independent fresh support. |

## Concrete addition and honest novelty boundary

`src/decision_audit.py` exposes an evaluator-only minimum certificate size:
it measures how many additional source labels are sufficient with hindsight.
An executable Bellman reference chooses queries that maximize the planning
prior probability of deciding a **frozen** proposal under a remaining budget.
Its independent binary prior is fixed at .5 and uncalibrated. Exact planning
is capped at ten source groups. The safety certificate uses no prior and no
unpaid reward. `src/decision_refresh.py` freezes the candidate throughout
each audit episode, refitting the verifier only between decisions.

The exact witness formula and Bellman recursion are elementary; stochastic
threshold evaluation and baseline-safe improvement have established prior
literature. See `notes/theory/decision_directed_auditing.md`. The contribution
here is a reproducible way to distinguish information shortage from acquisition
inefficiency, and to test whether a more complicated intervention is warranted.
Do not sell the reference planner as a new optimal recursive-refresh algorithm.

## What the completed experiment actually says

Eight tasks, six optimizer strengths, two representations and five audit seeds
produce **480 correlated comparison settings**, not 480 independent tasks.
Every task, including three zero-success tasks, remains included. Two paid
source labels fit the initial verifier. Each frozen candidate then composes
twelve stale updates. The decision target is true gain >= .01; rejecting a
candidate below this floor does not necessarily mean it has negative gain.

| Completed retrospective result | Impact order | Exact decision planner | Random source stream |
| --- | ---: | ---: | ---: |
| Resolve fixed gain decision, two additional-label cap | 97.92% | 97.92% | 71.46% |
| Additional calls to resolve every fixed comparison, six-label cap | .8000 | .8000 | 1.3604 |
| Hindsight minimum additional calls, same fixed comparisons | .7313 | .7313 | .7313 |
| Recursive final mean gain, total cap four labels | .12890 | .12014 | .05465 |
| Actual mean source labels, same four-label-cap loop | 2.8104 | 2.7875 | 2.8833 |
| Recursive final mean gain, total cap six labels | .13598 | .13250 | .12222 |

The exact planner has **no fixed-comparison resolution or accepted-gain
advantage** over impact order here and has lower recursive mean gain. At the
one-additional-label cap it avoids queries that cannot possibly resolve the
comparison, using .27708 versus .46875 additional calls at equal resolution;
at the other displayed caps it has the same cost as impact order. At full resolution,
impact order uses only .06875 more additional labels than the hindsight
witness average. This pilot therefore leaves little average acquisition
headroom in that setting; it does not establish that planner complexity
improves recursive progress at matched actual label cost.
Different later source identities change future verifier fits, so the recursive
comparison is an acquisition intervention, not an isolated refresh-timing effect.
Equal caps also do not mean equal actual label costs.

All accepted recursive policies satisfy the paid-label finite-bank baseline
gate; this conditional safety is by construction, not empirical evidence of
general deployment safety. Query transcripts, complete comparison/round rows,
per-task summaries, paired task-bootstrap descriptive intervals and hashes are
in `results/decision_audit_mbppplus_v1/`. Those intervals do not convert eight
selected tasks into a population claim. No programs were generated or newly
executed for this retrospective experiment.

Reproduce from the repository root:

```sh
python -m unittest discover -s tests -v
python -m src.evaluate_decision_audit data/mbppplus_qwen15b_pilot_v1_scored.jsonl --output results/decision_audit_reproduction
```

## Execute the next empirical gate before designing another controller

`.github/workflows/rvl-development-v2.yml` runs the already locked **32
development tasks × 16 model completions = 512 new candidate occurrences**.
It uses the pinned model and dataset, CPU generation, original prompt/decoding,
and the existing pre-outcome task manifest. It uploads the unscored bank before
scoring, then validates every reference and evaluates each program in its own
restricted Docker container. It never generates or scores the heldout split.
This is a pilot-informed development expansion, not confirmatory transfer.
The execution environment pins the official CPU-only PyTorch 2.6.0 wheel
before generation; all runtime versions are recorded in the generator manifest.

The pipeline runs the original all-task 32-paid-draw arm and identical ordered
six-source arm. A task with fewer than six distinct texts is reported as
source-timing-unidentified and stays in the draw arm. Infrastructure failures
and malformed scoring replies now abort instead of creating assumed zero
labels; declared candidate CPU/memory/output and wall limits remain recorded
candidate failures. The development gate retains round-one failures separately
and counts later crossings only after an initially safe first update.

The new five-task **development triage** rule is fixed before this run: fewer
than five task IDs with a later-onset crossing means the boundary remains
underidentified and heldout data must not be opened to search for a favorable
effect. Passing this triage merely permits freezing a predictor/cost rule;
it does not satisfy the separate original five-task heldout criterion.
The .5-prior decision planner is not run on sixteen-source development banks:
its exponential reference algorithm would exceed the declared ten-source cap.

If a development boundary is identifiable, freeze a predictor, cost target and
acquisition rule using development data alone; then perform the predeclared
heldout comparison. If the boundary remains absent, retain the negative result
and redesign as a different study before seeing further outcomes. A workflow
being submitted or running is not completed evidence; inspect its actual
status and download/verify its artifacts before updating any results table.

## Capability-creation experiment, conditional on transfer

The next separately locked study should update a policy or verifier using only
paid training labels, then draw **new completions** from the updated generator
and score them with evaluation-only tests. Generated occurrences used for
training, rule calibration and final evaluation must be distinct. Include a
generation-only baseline, fixed-refresh baseline, trust-region baseline and
the development-frozen intervention. Match both generation tokens and actual
trusted calls; plot the full cost/progress frontier and task-level effects.
Report changes in fresh-sample pass probability, not success after choosing
among previously audited programs. Keep failed/unsolvable tasks in the main
denominator and report concentration/support loss.

Only positive independently evaluated fresh-support progress would support
the advertised policy/verifier coevolution story. The finite-bank safety proof
does not transport to this experiment automatically. This separation is the
main protection against mistaking benchmark selection for capability creation.
