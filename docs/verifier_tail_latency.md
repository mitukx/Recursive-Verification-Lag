# Verifier tail-latency routing

The distributed verifier service now includes an adaptive scheduler for
heterogeneous workers. It routes by a predicted-completion heuristic based on:

- worker capacity;
- locally reserved in-flight work;
- remotely reported in-flight work;
- EWMA end-to-end verifier RPC service time.

The scheduler preserves strict verifier-version fencing. A request deadline can
bound an unhealthy/slow worker; failures are quarantined and the same immutable
generation is retried on another compatible verifier.

The repository includes a loopback RPC benchmark comparing the prior naive
round-robin fleet against adaptive predicted-completion routing with deliberately
heterogeneous verifier latency/capacity. It retains per-request raw latency,
worker assignment, p50/p95, throughput, and final scheduler state.

This benchmark is intentionally labeled control-plane CPU evidence. It should
not be presented as a real-GPU or multi-host performance claim. The next tier is
Issue #66: run the same routing and verification-debt controls against real
vLLM/SGLang-backed rollout + verifier services under independently swept
capacity and injected latency.
