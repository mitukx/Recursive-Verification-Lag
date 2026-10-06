# Frontier GPU evidence bundle

This repository has separate acceptance paths for real-model RLVR, FSDP/NCCL
training, and vLLM serving. The `frontier-gpu-evidence` workflow combines them
into one auditable two-GPU run so the portfolio can point to raw evidence rather
than disconnected implementation claims.

## What one run measures

1. **Token-exact real-model replay.** A small Qwen model generates multiple
   GSM8K candidates. Response token IDs and behavior log-probabilities are saved
   with independently computed verifiable rewards.
2. **Low-precision GRPO parity.** The same FP32-generated behavior replay is
   consumed by FP32, BF16, and FP16 one-step GRPO runs. Loss, gradient norm,
   behavior-KL, tokens/s, peak GPU memory, and non-finite failures are recorded.
3. **Stage-aware GRPO profiling.** A warmed GRPO step is captured with PyTorch
   CPU/CUDA profiling and named ranges for advantage computation, model forward,
   backward, gradient clipping, and optimizer step. The workflow retains a Chrome
   trace plus the top self-device/self-CPU operators, memory, and tokens/s.
4. **Transactional RLVR.** Held-out Qwen evaluation runs before/after candidate
   updates. Rejected updates restore model, AdamW optimizer, and RNG state.
   Promotion decisions are retained in a hash-chained ledger.
5. **1 -> 2 GPU FSDP scaling.** The same real-model replay is trained with one
   and two CUDA ranks, then a two-rank job resumes from the sharded checkpoint.
   Throughput, scaling efficiency, precision, peak memory, and per-rank GPU
   telemetry are retained.
6. **vLLM serving sweep.** Streaming inference records requests/s, tokens/s,
   p50/p95/p99 latency, TTFT, TBT, Prometheus snapshots, KV-cache usage when
   available, server logs, and second-level GPU telemetry.
7. **Real process failure.** Two vLLM workers are launched on separate GPUs,
   one receives SIGTERM during load, and the scheduler must recover all requests
   through cross-worker failover.
8. **Fail-closed aggregation.** `summarize_gpu_evidence.py` checks that all
   required evidence categories are present and mechanically valid, then writes
   one `gpu-evidence-summary.json` with SHA-256 lineage for each raw source.

## Run

Use GitHub Actions -> **frontier-gpu-evidence** -> **Run workflow** on a
self-hosted Linux runner with at least two CUDA GPUs. The default model is
`Qwen/Qwen2.5-0.5B-Instruct`.

Equivalent local sequence:

```bash
python -m pip install -r requirements-systems.txt -r requirements-gpu.txt

python -m src.generate_rlvr_replay \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --tasks 4 --samples 4 \
  --output artifacts/frontier-gpu-evidence/replay.jsonl

MODEL=Qwen/Qwen2.5-0.5B-Instruct \
OUT=artifacts/frontier-gpu-evidence/qwen \
bash scripts/run_qwen_rlvr_gpu.sh

MODEL=Qwen/Qwen2.5-0.5B-Instruct \
REPLAY=artifacts/frontier-gpu-evidence/replay.jsonl \
OUT=artifacts/frontier-gpu-evidence/fsdp \
bash scripts/run_fsdp_scaling_gpu.sh

MODEL=Qwen/Qwen2.5-0.5B-Instruct \
OUT=artifacts/frontier-gpu-evidence/vllm \
bash scripts/run_vllm_free_gpu.sh

MODEL=Qwen/Qwen2.5-0.5B-Instruct GPU0=0 GPU1=1 \
OUT=artifacts/frontier-gpu-evidence/failover \
bash scripts/run_vllm_failover_demo.sh
```

## Acceptance semantics

A complete bundle requires all of the following:

- FP32/BF16/FP16 GRPO runs all completed with finite measured metrics;
- the GRPO profiler emitted named training stages and a non-empty Chrome trace;
- transactional Qwen RLVR produced at least one promotion decision;
- measured multi-rank FSDP throughput exists and has positive scaling efficiency;
- the two-rank FSDP job successfully resumed from the distributed checkpoint;
- the vLLM serving artifact contains at least three positive-throughput
  concurrency points;
- the injected worker failure caused at least one scheduler failure and at least
  one successful failover while preserving 100% request completion.

These are mechanical evidence checks, not thresholds for "frontier-scale"
performance. A slow result is still useful evidence if it is real, reproducible,
and accompanied by profiler data. Performance claims should be made only from
the raw run that produced the bundle.
