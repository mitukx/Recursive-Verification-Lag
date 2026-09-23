# MBPP+ recursive-loop analysis lock, before trusted scoring

## Status and decision rule

The unscored 64-program MBPP+ bank and this analysis lock were committed
before any model candidate was run on public or extra tests. The dataset's
released test programs, reference solutions, and task prompts were inspected
to build the scorer; model-generated program outcomes were not inspected.
This is a preregistered *within-project pilot*, not a clean external study.
Execution must first pass all eight reference programs inside a pinned Docker
container, recording the image ID, code hashes and limits. If any reference
fails, stop and document the failure; no assumed zero labels or silent fallback.
Keep every model candidate, including failures, duplicates, and timeouts.

## Primary descriptive contrasts

Run the existing `src.exact_cost_pilot` with `--record-coordinates`, unchanged
nine designs, six optimizer-strength settings, two feature representations,
five audit seeds, twelve rounds, and 32 accounted draws per configuration.
Run `src.source_timing_pilot` with unchanged five designs, exactly six distinct
source labels, the same configurations, and its identical-source-order stream
arm. Both have eight task clusters. Primary contrasts are early minus uniform
for (i) any round below initial trusted reward and (ii) final trusted gain,
separately for exact paid draws and the identical-source stream. Preserve
initial failures. The source acquisition arm that depends on current policy,
the alternative strengths and the adaptive trigger designs are secondary.

Task-cluster intervals are descriptive only; eight deliberately selected
benchmark tasks do not license population inference. The binary EvalPlus
trusted reward may make failures rare or unidentified in a stratum. Report
each task and stratum, including constant-zero rewards, without selecting a
subset after seeing outcomes. 32 draws with eight sampled occurrences can
heavily resample duplicates; report both paid draws and distinct labels.
Likewise, six distinct-source labels can approach full coverage. A null result
would not falsify RVL across larger candidate banks. No model weights are
updated; these runs are recursive selection on one frozen bank.

## Secondary safety/progress comparison

Evaluate `src.certified_refresh` on all eight tasks with its existing six
distinct-source budget, two initial sources, soft strength 1 and Best-of-4,
both representations, five seeds and twelve rounds. Compare to early,
uniform and late stream schedules on *the same optimizer/seed/representation
cells*. Report failures, final gain, actual distinct label cost, acceptance
fraction and how often a proposed policy is uncertifiable. The controller
has a conditional finite-support no-collapse certificate; a zero measured
failure count is not evidence of an empirical advantage independent of that
condition. Different later source acquisition means the comparison is not an
isolated timing effect. `fallback=interpolate` is post-v1/v2 exploratory and
must be separately labeled if included.

**Amendment recorded before MBPP+ candidate scoring.** The safe-region
projection controller was designed after inspecting both expression banks,
but the MBPP+ candidates remain unscored. Run `fallback=project` as an
explicitly **secondary** outcome on all eight MBPP+ tasks with exactly the
same 160 settings, two initial source audits, six-source maximum and 12
rounds as the abstaining controller. Compare paired final gain, any failure,
actual source labels, audited-policy mass and effective source support.
Keep all tasks including all-zero reward tasks, and report projection onto
audited sources even if final gain rises. This amendment must not be called
an independent replication: the algorithm, benchmark selection and
stopping rules were informed by the two expression banks, and the MBPP+
reference and tests are publicly available. No thresholds are fitted from
the forthcoming MBPP+ outcomes.

## Theory/empirical boundary

The box-identification bound and its boundary derivative are exact for
source-identical deterministic rewards in [0,1]. They do not imply that the
learnt verifier's error is structured or that a six-source budget is sufficient
for useful progress. No scalar shift metric is a prospective predictor unless
its threshold and direction are fixed on separate development tasks. There is
no such MBPP+ threshold in this eight-task pilot; report coordinate traces,
not a fitted cross-benchmark winner. No Oral/Spotlight claim follows from
these eight publicly released benchmark tasks.
