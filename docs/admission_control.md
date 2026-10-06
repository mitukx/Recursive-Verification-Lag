# Workload-aware admission control

The rollout scheduler can optionally bound work in weighted units rather than
only counting requests. This matters for RL inference because a request with
many samples or a large expected generation length can consume far more
serving capacity than a small evaluation request.

## Work units

`RolloutRequest.estimated_tokens` is an optional scheduling hint representing
the estimated total generated-token work for the request. If omitted, the
scheduler falls back to `samples`, preserving existing behavior.

`RolloutRequest.workload_id` identifies the independent workload competing
for capacity, such as a training rollout stream, evaluation stream, or
monitoring stream.

## Admission invariants

`FairWorkloadAdmission` enforces:

1. global in-flight work never exceeds `capacity_units`;
2. optional per-workload in-flight caps preserve headroom for other streams;
3. queued work is bounded and excess work is shed immediately with
   `OverloadedError`;
4. active workload queues receive round-robin service so one stream cannot
   continuously monopolize newly freed capacity;
5. timeout and task cancellation refund both queued and granted capacity;
6. speculative work cannot bypass normal queued work.

The scheduler uses the same controller for hedged requests. A hedge is launched
only if an additional speculative lease can be acquired immediately, preventing
tail-latency protection from silently violating the global work budget.

## Scope

These units are scheduling estimates, not measured GPU memory reservations.
Real vLLM/SGLang experiments should calibrate estimated token work against
prefill/decode cost and GPU memory pressure. The CPU tests establish accounting,
fairness, overload, and cancellation correctness.
