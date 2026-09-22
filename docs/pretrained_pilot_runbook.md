# Pretrained expression pilot: reproduction

Use a clean Python 3.11/3.12 environment. Generation needs the optional model
requirements; selection/analysis do not. No generated code is executed.

```bash
pip install -r requirements-model.txt
python -m src.generate_candidate_bank \
  --revision ea3f2471cf1b1f0db85067f1ef93848e38e88c25 \
  --samples 16 --output data/pilot_qwen05b_bank.jsonl
python -m src.run_candidate_sweep data/pilot_qwen05b_bank.jsonl \
  --output data/pilot_qwen05b
python -m src.analyze_candidate_sweep data/pilot_qwen05b
python -m src.compare_refresh data/pilot_qwen05b
python -m unittest discover -s tests -v
```

Existing banks/results are never overwritten. To reproduce from scratch choose
new output paths. The frozen bank hash, model revision, prompts, batch seeds,
decoding parameters, versions, token IDs and sampling-law log probabilities are
archived. Exact generation can still vary across hardware/software versions.
Run `--device mps` on Apple Silicon or `--device cuda` where available; that is a
new bank and must receive its own identifier. No API credentials or paid services
are required by this pipeline.

Pilot generation: Qwen/Qwen2.5-Coder-0.5B-Instruct (official model card:
https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct), 12 finite integer tasks,
16 independent sample occurrences per task, temperature .8, top-p .95,
64-token cap, batch size 8, seed 20260922. This is pretrained-model evidence in a
restricted expression language, not a general Python, agent, or weight-training
experiment. Independent generation-seed replication is still required.

## Data and decisions

- `pilot_qwen05b_bank.jsonl`: raw outputs and both public and offline hidden scores.
- `.manifest.json`: immutable input metadata, bank hash and completion flag.
- `runs.csv`: every task/factor/audit-seed result, including negative outcomes.
- `trajectories.csv.gz`: per-step pre-refresh proposals, committed movement,
  paid audit counts and outcomes. `oracle_*` columns are diagnostic only.
- `transfer.csv`: threshold transfer, trained without the held group; excludes
  post-collapse rows and refuses single-class balanced-accuracy conclusions.
- `controller_summary.csv`: task-clustered bootstrap intervals over the declared
  sweep. These are conditional on the development task set and one model bank.
- `crossfit_refresh*.csv`: exploratory leave-one-task-out fixed/adaptive selection.
  It records actual costs and held-task cost-ceiling violations. It is not a final
  confirmatory test, and equal caps are not equal realized cost.

The generation pilot contains **only development tasks**. Do not describe their
cross-validation folds as a pristine final holdout. The 100-task/128-sample target
and second-family evaluation in the original protocol remain outstanding. Avoid
adding hand-written exploits to the primary model bank. If a separate intervention
bank is used, mark it synthetic/edited and report it independently.

## Predeclared falsification and next decisions

H1: stale reuse can reduce hidden-domain reward. If the frozen pilot has no
baseline crossings, report that and inspect candidate diversity / public-frontier
quality before enlarging the bank. Do not regenerate it until a collapse appears.

H2: geometry transfers better than raw movement. Compare held-task, held-optimizer,
and held-representation scores. If no coordinate wins consistently, retain the
negative result. A post-outcome error contrast cannot be called an observable
predictor. Scores from different targets/folds are not directly interchangeable.

H3: adaptive refresh improves cost/progress. Select schedules without held-task
outcomes, report failures, gains and actual costs. If gains are bought with more
labels or abstention, do not call it cost-matched prevention. The heuristic refit
has no guarantee of fixing misspecification. Report zero-mass candidates/support
concentration to identify selection saturation and numerical limitations.

Theoretical scope and the remaining completeness/optimality gap are explicit in
`observable_refresh_frontier.md`. No new universal law is asserted.
