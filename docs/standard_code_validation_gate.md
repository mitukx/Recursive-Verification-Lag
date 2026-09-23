# Standard Python code validation gate (design and first scored pilot)

The first pinned eight-task Qwen2.5-Coder-1.5B-Instruct MBPP+ bank was scored
in isolated Docker and the pre-scoring-locked comparisons were run. Its
timing and projection results are largely negative; see
`notes/progress/33_mbppplus_first_scored_pilot.md`. The gate below remains
the design requirement for a larger independent follow-up, rather than a
claim that eight tasks establish benchmark-level transfer.

The restricted-expression banks test a narrow, finite-domain programming task.
They do not establish general program synthesis. This gate defines the next
evaluation family before any standard-benchmark run is interpreted as evidence.

## Evidence and execution requirements

Use the publicly released MBPP+ tasks and EvalPlus additional tests at a pinned
dataset revision. Select task IDs and the public/test partition from dataset
metadata before generating programs. Freeze the model revision, prompts, seeds,
sample counts, stopping rules and decoding settings in a manifest. If the model
may have trained on benchmark material, report this contamination limitation.
Neither released tests nor EvalPlus augmentations constitute secret held-out
data merely because they are withheld from the *optimization loop*; call them
evaluation-only trusted tests. The benchmark's reference implementation, when
available, should be used to validate each test partition before generation.

Each completion must run in a disposable isolated container with disabled
network, non-root identity, read-only base filesystem, no host mounts,
time/memory/process caps and a fresh instance per candidate. Record runtime
errors, timeouts and rejected cases as failures. Do not execute arbitrary model
code in the host Python interpreter; this environment currently lacks Docker,
so no standard-program benchmark claims are made in this workspace.

## RVL comparison to preregister at that gate

Treat each sampled program occurrence as frozen support, retain duplicates and
invalid completions. On each problem, public tests form the cheap verifier and
evaluation-only additional tests score trusted reward. The trusted label budget
counts *distinct* program/source identities as well as draws. Independently
sweep Best-of-N and soft selection, their strength, and refresh interval; use
the same bank, public tests, and initial label stream in paired comparisons.
Include both identical-source-order timing and equal-paid-draw protocols. The
adaptive rule is frozen on a development split and evaluated once on unseen
task IDs and generation seeds. Report correct-program discovery, fractional
pass rates, initial failures, label cost, and every task, including those with
zero correct candidates. Contrast KL, maximum log-density ratio, and a
verifier-error geometry statistic using frozen development thresholds and
time-to-warning analysis. If strong-generator support remains absent, treat
algorithmic comparisons as conditional on inadequate support.

The hypotheses are falsifiable: early refresh can worsen outcomes at equal
source information; a scalar threshold may fail to transfer; adaptation may
increase verifier cost or harm gain even when it prevents collapse. Report such
results without replacing task IDs, seeds or thresholds. The observable-frontier
theorem is a separate mathematical goal and does not follow from experiments.
