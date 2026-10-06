# xAI-oriented engineering evidence matrix

This file deliberately separates implemented evidence from claims that still
require real GPU measurements.

| Frontier RL / inference capability | Evidence in this repository | Status |
|---|---|---|
| Async rollout orchestration | `AsyncRolloutEngine` | implemented + CI |
| High-throughput multi-worker scheduling | `LeastLoadedScheduler`, bounded queue, per-worker inflight limits | implemented + CI |
| Backpressure / timeout handling | scheduler counters and timeout paths | implemented + CI |
| Worker failure handling | health streaks, quarantine, fail-fast when no healthy worker exists | implemented + CI |
| Remote inference integration | OpenAI-compatible `VLLMHTTPBackend` | implemented; real GPU benchmark pending |
| Local real-model rollout | `HFLocalBackend` with CUDA/MPS/CPU selection | implemented; hardware smoke run required |
| RLVR / GRPO update | group advantages + clipped token-level causal-LM trainer | implemented; model-quality benchmark pending |
| RL numerics safety | fp32 log-softmax, finite checks, clipped ratios, grad non-finite guard, clip diagnostics | implemented + syntax/logic tests |
| Low-precision policy | auto/fp32/fp16/bf16 model-load policy with CUDA bf16 capability guard | implemented; numerical comparison pending |
| Verifier integration | exact and functional verifier interfaces | implemented + CI |
| Weight-version correctness | stale rollout rejection | implemented + CI |
| Trainer-to-worker synchronization control plane | immutable manifests + checksums + worker acknowledgements | implemented + CI |
| Recovery | atomic reference checkpoint save/restore; remote retries | implemented + CI |
| Observability | counters, p50/p95/max latency, tokens/s | implemented + CI |
| Profiling | Chrome trace recorder and benchmark reports | implemented + CI |
| Cluster throughput benchmark | `benchmark_vllm_cluster.py` | runnable; GPU endpoints required |
| FSDP / DeepSpeed scale-out | not yet implemented | gap |
| NCCL / multi-node collectives | not yet implemented | gap |
| Custom CUDA/Triton kernels | not yet implemented | gap |
| BF16/FP16 empirical stability study | harness supported | measurement gap |

## Evidence bar before using this as an xAI portfolio centerpiece

Do not advertise this as a high-performance RL system until the following are
published with immutable configs and raw results:

1. a pinned Qwen model and serving-stack version;
2. one-GPU and multi-GPU rollout throughput;
3. p50/p95 latency at multiple concurrency levels;
4. GPU utilization / memory traces;
5. sync-vs-async speedup;
6. one real RLVR training curve on a public benchmark;
7. a worker-failure experiment demonstrating recovery;
8. policy-weight synchronization overhead;
9. fp32 vs bf16/fp16 throughput and numerical-stability comparison.

The codebase should make these experiments easy; the measurements are the
actual hiring evidence.
