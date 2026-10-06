# Upstream PR draft: workload-aware rollout routing

**Target:** `verl-project/verl`  
**Pinned base:** `8718ca30a3f002f93b7c4fd99b9b2506718681bc`

## Proposed title

`[Rollout] Add workload-aware routing for heterogeneous request lengths`

## Motivation

The default global rollout router balances concurrent request counts. For agent/RL traffic, request cost can vary substantially with prompt length and generation budget, so equal request counts can still create large per-replica work imbalance and tail latency.

This change adds an opt-in built-in router strategy that balances **admitted predicted token work** rather than request count:

`work = prefill_weight * prompt_tokens + decode_weight * decode_budget`

It is intentionally orthogonal to KV-cache-aware routing: this strategy does not inspect prefix-cache state and can be evaluated independently.

## Changes

- add `WorkloadAwareRequestLoadBalancer`;
- derive only two scalar hints in `LLMServerClient`: `prompt_tokens` and `decode_budget`;
- preserve the existing router field-declaration protocol, so full prompt IDs and sampling dictionaries do not cross the Ray routing RPC;
- retain sticky sessions;
- keep exact non-evicting per-request work accounting until release;
- fail closed on duplicate in-flight request IDs and release/server mismatches;
- fall back to configured rollout `response_length` when a request does not explicitly provide a generation budget;
- add CPU tests for accounting, plugin YAML loading, scalar wire fields, dynamic server handling, and fallback behavior.

Example router config:

```yaml
router_class: verl.workers.rollout.router.WorkloadAwareRequestLoadBalancer
prefill_weight: 1.0
decode_weight: 1.0
default_decode_tokens: 1
```

## Evidence policy

The committed deterministic trace simulation is hypothesis validation only and is not reported as serving performance.

A real endpoint benchmark is prepared with:
- pinned vLLM 0.29.0 and pinned model revision;
- two independent GPU replicas;
- homogeneous control and heterogeneous long-tail workloads;
- identical request traces/seeds;
- alternating arm order across repeats;
- prefix caching disabled in the primary routing-only comparison;
- request throughput, completion-token throughput, p50/p95/p99 latency, per-replica predicted-work imbalance, raw per-request records, output hash parity, and GPU telemetry.

The PR should only claim a performance gain after those real-GPU measurements complete. Negative/null results are retained.

## Relationship to existing work

This does not duplicate the KV-cache-aware routing work discussed in #6940. Cache-aware routing optimizes locality; this proposal addresses heterogeneous admitted request cost. A future composite policy could combine both signals, but this PR keeps the mechanism isolated and measurable.
