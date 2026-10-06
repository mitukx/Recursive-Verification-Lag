# Prefill/decode and KV-aware scheduling

LLM inference requests are not homogeneous. A long prompt is prefill-heavy,
while a long sampled completion is decode-heavy. The scheduler therefore
accepts optional request hints and worker capacity profiles.

## Request hints

`RolloutRequest` supports:

- `prompt_tokens_estimate`: estimated prompt length;
- `decode_tokens_estimate`: expected generated tokens per sample;
- `samples`: number of sampled continuations.

The control plane conservatively reserves KV capacity for every sampled
sequence. This intentionally does not assume prefix-cache sharing because that
behavior depends on the serving backend.

## Worker profile

`WorkerSlot` can expose:

- `prefill_tokens_per_s`;
- `decode_tokens_per_s`;
- `max_kv_tokens`.

When these values are provided, routing estimates prefill and decode service
time separately, combines that with the measured EWMA latency and current
in-flight load, and penalizes high KV pressure. Workers without enough KV
headroom are excluded before dispatch.

Reservations are released on success, failure, timeout, and cancellation.
The existing admission controller still provides a global workload budget;
KV reservation is an additional per-worker serving constraint.

These are scheduling estimates, not a replacement for vLLM's internal block
manager. Real GPU runs should calibrate the configured rates and KV capacity
against measured TTFT/TBT and the server's `/metrics` output.
