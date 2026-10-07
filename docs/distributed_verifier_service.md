# Distributed verifier service

The verification-aware asynchronous RL path can now move verification out of the
actor/learner process. `VerifierWorkerServer` exposes a bounded newline-delimited
JSON RPC service with two operations:

- `ping`: worker identity, verifier version, health, capacity, and in-flight load.
- `verify`: immutable generation payload + required verifier version.

The server fails closed on unhealthy workers, version mismatch, request timeout,
or verifier version changes during an in-flight request. The client independently
checks request identity, verifier version, and exact generation-payload equality
before accepting a reward.

`DistributedVerifierFleet` adds a small coordinator-side failover layer. Health
refresh quarantines stale/unhealthy workers, verification retries compatible
workers, and publishing a new expected verifier version fences the old fleet
until workers report the new version.

This is control-plane correctness evidence. It does not yet establish real-GPU
throughput, multi-host transport efficiency, or learned-verifier quality. Those
remain part of the real serving phase diagram tracked in Issue #66.
