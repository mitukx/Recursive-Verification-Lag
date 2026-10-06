# Verification-Aware Async RL

This subsystem turns the causal-LM mini-lab from an asynchronous actor/learner loop into an explicit **actor -> verifier -> learner** pipeline.

## Research question

The systems question is not only whether rollout can run asynchronously from learning. It is whether learning remains correct when policy updates and verification updates proceed at different rates.

Each replay group therefore carries two independent provenance clocks:

- **behavior policy version**: the policy that generated the immutable tokens/logprobs;
- **verifier version**: the reward/verifier state that admitted the group to learning.

The learner applies independent lag bounds to both clocks.

## Durable replay state machine

```text
generation
   |
   v
pending_verification
   |
   | claim + lease
   v
verifying --------------------.
   |                           |
   | success                   | failure / lease expiry
   v                           |
ready <--- reverify stale -----'
   |
   | learner admission:
   | policy_lag <= bound
   | verifier_lag <= bound
   v
consumed

terminal side states: stale, quarantined
```

Behavior data are immutable across verification and re-verification. A verifier may update reward metadata, but it may not change prompt/response tokens or behavior logprobs.

## Fail-closed semantics

A group in `pending_verification` or `verifying` is invisible to the learner. Verification exceptions are retried under a bounded attempt count and then quarantined; they are not converted to zero-reward training examples. Expired verification leases are reclaimable after restart.

When a learned verifier is refit, any ready group outside the configured verifier-version lag bound is returned to `pending_verification` and rescored from its original behavior trajectory. This preserves off-policy provenance while preventing stale reward admission.

## Version consistency

Verification and verifier refits share a runtime lock. A single replay group cannot contain scores from multiple verifier versions. Trusted audited rewards are treated as valid under the current verifier admission clock after refit because their reward is evaluator-owned rather than proxy-owned.

Learner metrics now retain:

- behavior policy version and policy lag;
- verifier version at admission and verifier lag;
- verification queue-to-ready latency;
- verification age at learner admission;
- verification attempts and backlog.

## Verification debt backpressure

The runtime now treats unverified work as a bounded systems liability rather than only a queue length. `VerificationDebtController` scores observable replay state from pending/verifying work, stale ready rewards, policy lag, verifier lag, and age of the oldest unverified group. The scalar is used only for rollout backpressure and telemetry; it is not a safety or quality certificate.

Three control regions are explicit:

- `admit_generation`: verification debt is below the soft limit;
- `throttle_generation`: rollout waits for verification/learner progress once debt is elevated;
- `pause_generation`: the same fail-closed wait is enforced above the hard limit until debt drains.

The controller configuration is part of the durable verification resume identity, so a restarted run cannot silently change debt weights or thresholds.

`src/benchmark_verification_debt.py` provides a deterministic three-arm queue simulation: synchronous inline verification, naive async with policy-lag control only, and verification-aware async admission. It is intended to lock the expected control-plane behavior before real GPU measurements. Its output is explicitly synthetic evidence only.

## Current evidence

CPU/tiny-model acceptance tests cover:

1. pending verification is not learnable;
2. fresh verification admits a group;
3. verifier refresh requeues stale ready data for re-verification;
4. injected verifier failure is retried, quarantined, and produces no policy update;
5. existing token-exact behavior replay remains compatible.

This is systems-contract evidence, not a throughput claim. Real GPU measurements of verifier throughput, queue growth, policy/verifier staleness distributions, and end-to-end training efficiency remain to be executed.

## Next experiments

The next evidence tier should sweep verifier service capacity and injected latency independently from rollout capacity. Report policy lag, verifier lag, verification debt, learner idle fraction, reward freshness, tokens/s, verifier calls/s, and held-out task quality. The central comparison should be:

- synchronous verification baseline;
- async verification with policy-lag control only;
- verification-aware async admission with both policy and verifier lag bounds.

Negative and throughput-regressing results should be retained.
