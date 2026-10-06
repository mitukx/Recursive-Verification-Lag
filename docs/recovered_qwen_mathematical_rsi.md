# Recovered-Qwen mathematical RSI mechanism replay

This is the CPU-available bridge between the synthetic mathematical RSI
controller and the still-blocked learned-verifier GPU chain.

The input is the committed `data/recovered_qwen05b_bank.jsonl`: actual
generations from `Qwen/Qwen2.5-Coder-0.5B-Instruct` on 12
restricted-expression tasks. The bank is a previously inspected development
pilot, so every result here is **exploratory mechanism evidence**, never
confirmatory evidence.

## Frozen proposals

For each `eta in {0.25,1,4}`, the replay follows 12 rounds of unguarded
public-score reweighting. This produces 36 frozen proposal comparisons without
reading hidden trusted scores. Each proposal records proxy gain and, after the
controller action is frozen, evaluator-only trusted gain.

The controller sees `trusted_score` only for candidates selected by its paid
audit rule.

## OAI-math-derived control layers

**#136 — coded verification.** The 12 task identities are local obligations.
The aggregate trusted-gain box certificate is critical. Up to 25% of task
obligations may be certified locally harmful before the coded controller
refuses to allow the proposal. This is an engineering analogue, not a
PCP-for-PPAD reduction.

**#116 — fixed probe coverage.** `coded_fixed_probe` allocates at least one
trusted query to every one of the 12 task strata before spending remaining
budget on high-impact candidate occurrences. `coded_random` provides the
coverage ablation and refreshes if any task receives no probe. The task basis is
fixed before trusted labels are read; it is not a proved universal hitting set.

**#140 — information budget.** With 12 task strata and target error 0.1, the
locked proxy requirement is `ceil(12 * log(10)) = 28` trusted labels.
Therefore the 12- and 24-label `coded_fixed_probe_info` arms must refresh;
36 and 48 labels may decide. This is a heuristic research budget, not an LLM
lower bound.

**#229 — not invoked as a theorem.** This bank is a flat candidate set, not a
broadcast tree, so reconstruction criticality is deliberately excluded from
this replay. It remains in the separate real-learned-verifier diagnostic.

## Compared controllers

- `proxy_only`: public reward only, zero trusted labels;
- `global_impact`: exact finite-bank global box certificate, label-blind
  highest-|policy-mass-change| acquisition;
- `global_random`: global certificate with random unique candidate audits;
- `coded_random`: coded local obligations with random acquisition plus full
  task-coverage requirement;
- `coded_fixed_probe`: coded obligations with guaranteed task coverage;
- `coded_fixed_probe_info`: fixed probes plus the information-budget gate.

Any non-proxy `allow` requires a nonnegative distribution-free global lower
bound under only the audited labels and the assumption that every unseen
trusted score lies in [0,1]. The independent validator rejects an output where
an allowed finite-bank update is evaluator-known harmful.

## What to look for

The replay does **not** need a positive result to be useful. The informative
questions are whether public-score optimization creates false progress, how
much trusted budget exact certification needs, whether fixed task coverage
changes resolution, whether coded verification catches concentrated task
regressions, and how much conservatism the information-budget gate adds.

The GitHub workflow runs the complete replay on the committed bank,
independently validates it, and retains raw trial rows regardless of result.
