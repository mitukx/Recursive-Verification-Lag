# Free-GPU vLLM evidence runbook

This runbook turns the control-plane project into measured GPU evidence. It is
designed for a temporary single-GPU notebook/VM such as a free GPU environment;
the repository itself does not assume a specific provider.

## 1. Install

Use a Linux CUDA environment and install the GPU requirements:

```bash
git clone https://github.com/mitukx/Recursive-Verification-Lag.git
cd Recursive-Verification-Lag
python -m pip install -U pip
pip install -r requirements-gpu.txt
```

The launcher records the installed vLLM package metadata and `nvidia-smi`
output in the artifact directory, so the exact environment is retained with
the result.

## 2. Single-GPU serving sweep

```bash
MODEL=Qwen/Qwen2.5-0.5B-Instruct \
REQUESTS=24 \
CONCURRENCY=1,2,4,8 \
MAX_TOKENS=64 \
bash scripts/run_vllm_free_gpu.sh
```

The script launches the current documented `vllm serve` OpenAI-compatible
server, waits for `/v1/models`, samples GPU utilization every second, then
runs streaming requests. It captures client-side TTFT, end-to-end latency,
approximate inter-token time, requests/s and tokens/s. It also snapshots the
server's Prometheus-compatible `/metrics` endpoint, including generation-token
counters and KV-cache usage when exposed.

Outputs are written below `artifacts/vllm-free-gpu/` and checksummed into a
manifest.

## 3. Real process-failure experiment

A stronger test starts two small vLLM servers and kills one during rollout:

```bash
MODEL=Qwen/Qwen2.5-0.5B-Instruct \
GPU_MEMORY_UTILIZATION=0.40 \
bash scripts/run_vllm_failover_demo.sh
```

This intentionally sends SIGTERM to one locally launched serving process. The
scheduler must quarantine the failed endpoint and retry affected requests on
the second endpoint. The benchmark fails unless every request completes and at
least one cross-endpoint failover succeeds.

Two servers on one free GPU may not fit on every accelerator. If they do not,
run the serving sweep first and reserve failover for a larger-memory or
multi-GPU session.

## 4. Evidence to publish

Publish the raw JSON reports, GPU telemetry CSV, vLLM package metadata, server
logs, manifest, model name, GPU model, concurrency sweep and exact command.
Do not claim production-scale performance from a free-GPU run; use it as direct
evidence that the same scheduler and failure logic execute against a real GPU
serving stack.
