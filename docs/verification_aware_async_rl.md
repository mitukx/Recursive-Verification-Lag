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

## Verifier fleet concurrency

The runtime accepts `verification_workers=N`. Workers lease distinct replay groups concurrently and score them under a shared reader lease. Learned-verifier refit/audit intervention obtains an exclusive writer lease, so a version transition cannot occur in the middle of any in-flight group. The gate is writer-preferring to prevent an unbounded stream of scoring work from starving a pending refit.

Tiny-model acceptance includes a slow verifier fixture and requires observed verification concurrency greater than one with two workers. This demonstrates scheduling semantics only; real verifier scaling efficiency remains part of Issue #66.

## Version consistency

Verification and verifier refits share a runtime lock. A single replay group cannot contain scores from multiple verifier versions. Trusted audited rewards are treated as valid under the current verifier admission clock after refit because their reward is evaluator-owned rather than proxy-owned.

Learner metrics now retain:

- behavior policy version and policy lag;
- verifier version at admission and verifier lag;
- verification queue-to-ready latency;
- verification age at learner admission;
- verification attempts and backlog.

## Locked three-arm runtime semantics

The Issue #66 comparison now runs through one `AsyncHFLab` implementation with
an explicit `arm` switch, reducing implementation-level confounds:

- `sync_inline`: generation waits until the previous verification backlog is
  drained before admitting another rollout group; verifier freshness is enforced.
- `async_policy_only`: generation is not throttled by verification debt and the
  learner does not require a current verifier version. Old verifier rewards may
  therefore remain learner-admissible by design; policy-lag bounds still apply.
- `verification_aware_async`: generation/verification/learning remain decoupled,
  while verifier freshness and verification-debt backpressure are both enforced.

The replay layer has an explicit `enforce_verifier_freshness` switch so the
policy-only control does not accidentally inherit stale-reward requeue behavior
from the treatment arm. Contract tests exercise both paths.

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
