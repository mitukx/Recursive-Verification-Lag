# Transfer v2 protocol, fixed before seeing candidate outputs

## Question and limits

Does the change in generator capability restore correct-program support on a
newly authored task suite, and does refresh timing have the same sign as on the
observed v1 1.5B bank? This is a prospective **within-family** replication.
The researcher designed the suite after seeing v1 results; it is not an
external benchmark or an independent representation shift. This protocol is
fixed before v2 generation, but not registered outside the project.

## Frozen generation and integrity

Eight task IDs and their references in `src/transfer_code_tasks_v2.py`, in file
order; evaluator and public/hidden partition in `src/finite_code_tasks.py`.
Model: Qwen/Qwen2.5-Coder-1.5B-Instruct at
`2e1fd397ee46e1388853d2af2c993145b0f1098a`. Seed 20260926; 16
occurrences per task; temperature 0.8; top-p 0.95; maximum 64 new tokens;
batch size 8. Freeze the generator and task-source hashes in a manifest.
Discard no outputs, duplicates, failures or truncated completions. Report
the number of tasks with a fully hidden-correct candidate, total correct
occurrences, validity, and public-perfect/hidden-imperfect count. The
predefined feasibility decision is at least 4/8 tasks with a correct
candidate; this is a design gate, not a hypothesis test.

## Frozen recursive comparisons

Reuse `src.exact_cost_pilot` and `src.source_timing_pilot` without changing
their nine and five controller designs, respectively, five audit seeds,
strengths, representations, or 12 rounds. The two primary descriptive
contrasts are early minus uniform failure rate and final gain, separately
at 32 paid draws and at six distinct sources with identical stream labels.
Failure is any policy reward below its initial reward, including first-round
failure. Retain all tasks. Task-cluster bootstrap intervals are descriptive
and do not justify a population claim from eight hand-authored tasks.
Compare the direction with v1 1.5B; a sign reversal is a negative result.
Policy-driven source acquisition and adaptive controllers are secondary.
Do not choose a stronger schedule from v2 outcomes.

The public tests are the cheap score available to generation; the disjoint
hidden finite domain is the offline trusted reward, selectively revealed under
the charged audit interface. All candidates are frozen, so this is recursive
**selection/reweighting**, not model weight updates or a real coding agent.
No theorem follows from any empirical comparison. For scalar-warning
evaluation, use only previously frozen development thresholds and report
unidentified strata rather than retune thresholds on v2.
