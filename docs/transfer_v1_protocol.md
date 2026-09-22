# Prospective transfer v1: locked before generation

Question: do earlier verification and its safety/progress tradeoff transfer to
unseen task specifications and independent generation seeds of the same model?
This is not a second model family or general-purpose Python benchmark. Task
authoring was informed by development findings; no external preregistration is
claimed. Existing operations recur in new combinations, so family independence
is limited. Public inputs and hidden target remain unchanged.

Generate all eight tasks in src/transfer_code_tasks.py, 32 occurrences per task,
seeds 20260923 and 20260924, immutable Qwen revision
ea3f2471cf1b1f0db85067f1ef93848e38e88c25. Temperature .8, top-p .95, 64 tokens,
batch 8, CPU threads 4. Preserve all invalid outputs and duplicates. Do not retry
because correctness or failure rate is unfavorable. A technical failure must be
recorded and may be resumed only with a new output identity.

Primary: run the unchanged exact_cost_pilot nine-design grid separately on each
bank, with 32 paid draws, common initial audit, five audit seeds, 12 rounds,
both representations, soft eta .25/1/4 and BoN 2/4/16. Primary contrasts are
early minus uniform and KL .5 minus uniform; retain all nine designs. Report
failure (any reward below initial) and final gain jointly. Desired-direction
replication requires the point-estimate signs to agree in both generation banks;
failure to do so is a negative result, not a reason to retune.

Use task-cluster bootstrap on generation-seed-averaged contrasts, keeping all
audit seeds/optimizers/representations together within tasks. Report each bank
separately. Two generation seeds and eight tasks are insufficient for broad
significance claims. Do not merge these tasks with development and relabel them
held out. No multiple-testing-adjusted claim is planned.

Secondary: exact-source stream/policy acquisition comparison only if each task
has at least six distinct sources, as required by the already specified budget.
If this fails, report that source-count feasibility failure; do not silently drop
tasks or regenerate. Primary paid-draw analysis remains defined for every task.
Bank quality is a diagnostic, not a post-outcome inclusion criterion.

Falsifiers: gain/failure direction reverses across banks; adaptive safety is
bought with lower gain; effects are confined to syntax failures or a single task;
or limited candidate support prevents the source-level comparison. Preserve each.
Next decision: revise the proposed mechanism, not thresholds on these new outcomes.
