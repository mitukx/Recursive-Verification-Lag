# verl workload-aware rollout routing upstream prototype

Target: verl-project/verl Q3 roadmap item "workload awareness scheduling".

This prototype is intentionally orthogonal to draft KV-cache-aware PR #6940.
The current default verl router balances request counts. This proposal tracks
predicted outstanding token work per replica using prompt length plus requested
decode budget and chooses the replica with the least predicted work.

The prototype mirrors verl's current acquire/release field declaration contract:
- acquire consumes prompt_ids and sampling_params;
- release consumes only request_id;
- admitted work is retained locally so prompt_ids are not serialized again;
- sticky sessions and deterministic routing remain explicit semantics.

The included benchmark is a deterministic queue/service model and is **not**
evidence of GPU speedup. It only validates the scheduling hypothesis and
accounting. Any upstream performance claim requires a pinned real vLLM/SGLang
rollout benchmark with identical prompts/seeds and throughput + p50/p95/p99
latency + per-replica utilization/cache metrics.

Upstream write blocker: the connected GitHub account currently exposes no
mitukx fork of verl, so this branch stages the patch logic until a writable fork
exists.
