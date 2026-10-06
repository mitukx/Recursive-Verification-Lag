# Real-model mathematical RSI replay

This stage moves the OpenAI-Math-inspired RSI controller from synthetic
champion/challenger tests onto retained **real Qwen candidate evidence**.

The input is a completed `qwen_learned_verifier_bridge_v1` artifact. For each
real-model seed it contains 64 current-policy candidate occurrences, exact
trusted rewards, four verifier score fields (`oracle`, `fresh`, `stale`,
`shuffled`), and the independently measured terminal preference shift caused
by optimizing each verifier arm.

No model response is regenerated in this stage. The experiment asks a narrower
question: given exactly the same real candidate bank and trusted-label budget,
does the OAI-math-inspired controller make better intervention decisions when
the verifier is fresh, stale, or deliberately misaligned?

## Locked controllers

At each trusted-label budget the replay compares five methods:

1. `passive_fixed`: the repository's existing passive covariance audit.
2. `active_minimax`: the existing geometry-aware active audit.
3. `coded_random`: #136-inspired local obligations, but trusted queries are
   sampled globally and may miss local probe strata.
4. `coded_fixed_probe`: coded verification plus #116-inspired predeclared
   fixed probe coverage. Candidate IDs are sorted and assigned round-robin to
   four strata before trusted labels are inspected.
5. `coded_fixed_probe_info`: the fixed-probe controller plus the #140-inspired
   budget gate `B >= ceil(2 d log(1/epsilon))`.

All methods are charged the requested trusted-label budget even if the
information gate refuses to decide. This makes the primary 32-label comparison
conservative and directly comparable.

The coded controller keeps a global covariance confidence interval as a
critical obligation. It also evaluates four local stratum obligations and
permits at most one local obligation to be non-positive/inconclusive
(`delta=0.25`). This is an **engineering analogue** of robust local
verification; it is not a PCP-for-PPAD reduction.

## #229 diagnostic

For each real verifier arm the replay also reports

`d * lambda_hat^2`, where `lambda_hat = 2a - 1` and `a` is thresholded
verifier/trusted agreement, for `d in {2,4,8,16}`.

The report always records `assumptions_met=false`: this candidate bank is not
an independent homogeneous broadcast tree, so the threshold is not used as a
promotion theorem.

## Primary rule

The locked primary comparison is at 32 charged trusted labels. The directional
criterion for `coded_fixed_probe_info` is:

- on stale/shuffled arms, harmful-update allow rate must be no larger than
  `passive_fixed`; and
- wrong-sign decisive rate must not increase.

Higher refresh/inconclusive rate and beneficial blocking are explicit costs and
must remain visible. A negative or underpowered result is retained.

This replay is retrospective with respect to the already-designed upstream
candidate bank. It is therefore **not** confirmatory evidence for a theorem or
for unrestricted RSI. Its value is to determine whether the mathematical
control primitives survive contact with retained real-model verifier geometry
before spending more GPU budget.

## Execution

After a completed learned-verifier run:

```bash
python scripts/run_real_llm_mathematical_rsi.py \
  --protocol configs/real_llm_mathematical_rsi_v1.json \
  --input-root /path/to/qwen_learned_verifier_bridge_v1 \
  --output results/real_llm_mathematical_rsi_v1

python scripts/validate_real_llm_mathematical_rsi_evidence.py \
  results/real_llm_mathematical_rsi_v1 \
  --protocol configs/real_llm_mathematical_rsi_v1.json \
  --input-root /path/to/qwen_learned_verifier_bridge_v1
```

The GitHub workflow `rvl-real-llm-mathematical-rsi` performs upstream semantic
revalidation, checks the earlier promotion receipt, binds a new evidence
receipt to the exact downstream source/protocol/run, executes the replay,
independently recomputes the summaries, and uploads immutable artifacts.
