# xAI-oriented engineering evidence matrix

This file deliberately separates implemented evidence from claims that still
require real GPU measurements.

| Frontier RL / inference capability | Evidence in this repository | Status |
|---|---|---|
| Async rollout orchestration | `AsyncRolloutEngine` | implemented + CI |
| High-throughput multi-worker scheduling | `LeastLoadedScheduler`, bounded queue, per-worker inflight limits | implemented + CI |
| Dynamic micro-batching | compatibility-keyed `DynamicBatcher` + vLLM batch-chat adapter | implemented + contract CI; GPU throughput pending |
| Backpressure / timeout handling | scheduler counters and timeout paths | implemented + CI |
| Worker failure handling | health streaks, quarantine, fail-fast when no healthy worker exists | implemented + CI |
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
| Recovery | atomic reference checkpoint save/restore; remote retries | implemented + CI |
| Observability | counters, p50/p95/max latency, tokens/s | implemented + CI |
| Profiling | Chrome trace recorder and benchmark reports | implemented + CI |
| Cluster throughput benchmark | `benchmark_vllm_cluster.py` | runnable; GPU endpoints required |
| FSDP / NCCL multi-GPU | not yet validated | gap |
| Custom CUDA/Triton kernels | not yet implemented | gap |
| BF16/FP16 empirical stability study | harness supported | measurement gap |

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
7. a worker-failure experiment demonstrating recovery;
8. policy-weight synchronization overhead;
9. fp32 vs bf16/fp16 throughput and numerical-stability comparison;
10. NCCL/FSDP multi-GPU validation.

The codebase should make these experiments easy; the measurements are the
actual hiring evidence.
