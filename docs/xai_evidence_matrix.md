# xAI-oriented engineering evidence matrix

This file deliberately separates implemented evidence from claims that still
require real GPU measurements.

| Frontier RL / inference capability | Evidence in this repository | Status |
|---|---|---|
| Async rollout orchestration | `AsyncRolloutEngine` | implemented + CI |
| High-throughput multi-worker scheduling | predicted-completion routing from EWMA service time, bounded queue, per-worker inflight limits | implemented + CI |
| SLO / tail-latency control | end-to-end request deadlines, queue/capacity deadline accounting, optional distinct-worker hedging + loser cancellation | implemented + synthetic CI artifact; real GPU evidence pending |
| Workload admission / overload control | weighted in-flight work budget, bounded queued work, immediate shedding, per-workload caps, round-robin fairness, cancellation-safe leases | implemented + CI artifact |
| Dynamic micro-batching | compatibility-keyed `DynamicBatcher` + vLLM batch-chat adapter | implemented + contract CI; GPU throughput pending |
| Backpressure / timeout handling | scheduler counters and timeout paths | implemented + CI |
| Worker failure handling | cross-worker failover retries, circuit breaker, cooldown, single half-open probe, explicit recovery, deterministic chaos benchmark | implemented + CI artifact |
| Remote worker transport | asyncio TCP RPC server/client with request IDs and policy-version checks | implemented + integration CI |
| Remote inference integration | OpenAI-compatible `VLLMHTTPBackend` | implemented; real GPU benchmark pending |
| Local real-model rollout | `HFLocalBackend` with CUDA/MPS/CPU selection | real Transformers smoke CI |
| RLVR / GRPO update | grouped advantages + clipped token-level causal-LM trainer | real Transformers update smoke CI; quality benchmark pending |
| PyTorch distributed execution | 2-process torchrun, broadcast, all-reduce, DDP gradient sync + optimizer step | real CI |
| RL numerics safety | fp32 log-softmax, finite checks, clipped ratios, grad non-finite guard, clip diagnostics | implemented + CI |
| Low-precision policy | auto/fp32/fp16/bf16 model-load policy with CUDA bf16 capability guard | implemented; GPU numerical comparison pending |
| Verifier integration | exact and functional verifier interfaces | implemented + CI |
| Weight-version correctness | stale rollout rejection | implemented + CI |
| Trainer-to-worker synchronization control plane | immutable manifests + checksums + worker acknowledgements | implemented + CI |
| Two-phase policy deployment | publish -> all-worker ack -> activate; real TCP-worker version transition test | implemented + integration CI |
| Recovery | atomic reference checkpoint save/restore; remote retries | implemented + CI |
| Observability | counters, p50/p95/max latency, tokens/s | implemented + CI |
| Profiling | Chrome trace recorder and benchmark reports | implemented + CI |
| Deterministic failure replay | semantic scheduler event log, invariant validation, concurrency-stable SHA-256 trace digest | implemented + CI artifact |
| Regression / deploy gates | machine-readable absolute and relative benchmark rules fail CI on regressions | implemented + CI |
| Artifact lineage | benchmark/replay manifest with path, size, SHA-256 and immediate verification | implemented + CI artifact |
| Cluster throughput benchmark | `benchmark_vllm_cluster.py` | runnable; GPU endpoints required |
| FSDP / NCCL multi-GPU | not yet validated | gap |
| Custom CUDA/Triton kernels | not yet implemented | gap |
| BF16/FP16 empirical stability study | harness supported | measurement gap |

## Persistent asynchronous training and RVL integration

See [the mini-lab runbook](mini_frontier_lab.md) for executable commands and
explicit acceptance boundaries.

| Added capability | Implementation/evidence | Remaining boundary |
|---|---|---|
| Concurrent actor/learner training | CPU actors and independent real causal-LM model copies | multi-node online learner service |
| Durable experience | WAL replay, expiring fenced leases, immutable behavior tokens, lag filtering | distributed database/retention |
| RVL interventions | trusted audit acquisition, residual fitting, reward re-evaluation, equal-cost cadence sweeps | heuristic controller, not a certificate |
| Policy/verifier co-evolution | bandit attacks, trusted labels, critic fitting, fresh failure-task replay | open-ended learned red teaming |
| Model-driven coding agents | JSON tools, pinned episode weights, per-step journals, terminal-return RL | pretrained capability gain and multi-hour completion |
| Executable reward security | evaluator-owned I/O comparisons outside candidate Python, isolated Docker, attack tests | broad adversarial robustness |
| Verifier ensembles | executable public tests, trained residual critic, JSON remote judge adapter | real stronger-model calibration |
| Distributed token-level RL | actual two-rank DDP optimizer update; FSDP/DCP entry point | NCCL/FSDP GPU validation |
| Deployment | health-probed blue/green serving, drain old episode leases | real GPU reload/uptime evidence |
| Token-exact serving data | server-owned token IDs/logprobs, strict schema/model checks | real vLLM/SGLang GPU acceptance |
| Metrics | p50/p95/p99, LM tokens/s, utilization sampler, explicit-input MFU estimator | measured sustained GPU scaling |

## Evidence bar before using this as an xAI portfolio centerpiece

The repository now has real model and real multi-process CI evidence. Do not
advertise it as a high-performance GPU RL system until the following are
published with immutable configs and raw results:

1. a pinned Qwen model and serving-stack version;
2. one-GPU and multi-GPU rollout throughput;
3. p50/p95 latency at multiple concurrency levels;
4. GPU utilization / memory traces;
5. sync-vs-async speedup;
6. one real RLVR training curve on a public benchmark;
7. a real GPU worker-failure experiment demonstrating recovery (CPU deterministic chaos evidence is already CI-validated);
8. policy-weight synchronization overhead;
9. fp32 vs bf16/fp16 throughput and numerical-stability comparison;
10. NCCL/FSDP multi-GPU validation.

The codebase should make these experiments easy; the measurements are the
actual hiring evidence.
