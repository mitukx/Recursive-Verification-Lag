# Post-training capability campaign v1

This is the repository's hiring-facing post-training track. The goal is not to
accumulate more infrastructure features. The goal is to answer two empirical
questions on a real causal language model:

1. **Can verified RL produce a held-out capability gain?**
2. **When reward comes from a learned verifier, does keeping that verifier fresh
   preserve trusted capability progress better than stale or shuffled reward?**

The campaign intentionally reuses the existing Qwen/GSM8K RLVR runner and the
prospectively locked learned-verifier bridge. Systems components are supporting
infrastructure, not the primary result.

## Evidence tracks

### A. Oracle RLVR capability gain

Run `src/run_qwen_rlvr_experiment.py` on a pinned model/data split. Retain
before/after greedy held-out predictions, optimizer history, telemetry, exact
configuration, and model/checkpoint provenance. The primary number is held-out
exact-match accuracy delta. Zero and negative deltas are first-class results.

### B. Learned reward-model robustness

Run the four matched-update arms `oracle / fresh / stale / shuffled`. The primary
trusted endpoint is terminal preference shift on evaluation candidates whose
trusted labels are withheld until arm updates and LR selection are fixed.
Verifier quality and GRPO-aligned geometry are retained separately so a higher
proxy score cannot be mistaken for a capability improvement.

## Scorecard

After evidence exists, build one machine-readable summary:

```bash
python scripts/summarize_posttraining_campaign.py \
  --rlvr-benchmark artifacts/qwen-rlvr/benchmark.json \
  --learned-verifier-summary results/qwen_learned_verifier_bridge_v1/summary.json \
  --output artifacts/posttraining-capability-scorecard.json
```

The scorecard reports held-out capability delta, learned-verifier evidence
sufficiency, fresh-vs-stale trusted preference shift, fresh-vs-shuffled trusted
preference shift, and the gap from the oracle arm. It never turns missing or
underpowered evidence into a positive claim.

## Hiring claim boundary

The repository should be described as a post-training research/engineering
project only to the level supported by retained raw runs. The target signal is:
design the reward, run real RL, detect reward-model exploitation or staleness,
intervene, and show the effect on independent trusted evaluation. Production
scale and frontier-model gains require separate evidence.
