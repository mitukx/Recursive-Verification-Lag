# RLVR Systems Track

This extension turns the RVL research repository into a small end-to-end
post-training systems project. The goal is to make engineering ability
observable, not to claim frontier-scale performance from a laptop.

## What is implemented

- bounded-concurrency asynchronous rollout scheduling;
- latency-aware multi-worker scheduling with distinct-worker failover retries, end-to-end deadlines, optional hedging, circuit-breaker quarantine, and half-open recovery;
- a backend protocol plus a trainable CPU toy policy;
- an OpenAI-compatible HTTP adapter for vLLM/SGLang servers;
- a local Hugging Face causal-LM backend with token-level log-prob capture;
- deterministic and functional reward verification;
- grouped reward normalization and clipped GRPO objectives;
- a one-device PyTorch causal-LM GRPO trainer;
- policy-movement/staleness verifier refresh;
- atomic checkpoint/restore for the reference trainable backend;
- throughput/latency/reward telemetry and async-vs-serial benchmarks;
- Mac/CPU demos, tests, and dedicated CI.

The lightweight backend exercises the same rollout/verifier/control-plane
interfaces used by real-model backends. Torch and Transformers are optional so
the research CI remains fast.

## Fast CPU/Mac validation

```bash
python -m unittest tests.test_rlvr_systems tests.test_rlvr_reliability tests.test_rlvr_objectives -v
python -m src.run_rlvr_systems_demo --rounds 12 --samples 32
python -m src.benchmark_rollout_engine --requests 16 --samples 8 --latency-ms 5 --concurrency 8
python -m src.benchmark_scheduler_chaos --requests 32 --recovery-requests 12
python -m src.benchmark_scheduler_slo --trials 12 --primary-ms 40 --backup-ms 2 --hedge-after-ms 5 --deadline-ms 20
```

## Real local model on Apple Silicon, CUDA, or CPU

Install the optional dependencies:

```bash
pip install -r requirements-systems.txt
```

Run a one-device real-model rollout/verification/update smoke test:

```bash
python -m src.run_hf_grpo_smoke \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --steps 1 \
  --samples 2
```

`HFLocalBackend` captures generated token ids and normalized transition
log-probabilities. `HFCausalLMGRPOTrainer` then recomputes differentiable
token log-probabilities and applies a clipped token-level GRPO/PPO-style
surrogate. This is intentionally a small single-device trainer; large-model
training should replace the optimizer layer with FSDP/DeepSpeed while keeping
the rollout/verifier interfaces.

## Point at a GPU serving stack

`VLLMHTTPBackend` targets the standard `/v1/completions` endpoint exposed by
vLLM and compatible servers and records returned token log-probabilities:

```python
from src.rvl_systems.backends import VLLMHTTPBackend

backend = VLLMHTTPBackend(
    endpoint="http://127.0.0.1:8000",
    model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
)
```

For real GPU evidence, use `docs/free_gpu_runbook.md`. The GPU harness launches
`vllm serve`, captures streaming TTFT/TBT and tokens/s, snapshots the vLLM
Prometheus endpoint, records `nvidia-smi` telemetry, and includes a two-server
SIGTERM failure-injection path.

## Engineering benchmark plan

Report concrete systems numbers instead of repository size:

1. synchronous vs asynchronous rollout throughput;
2. p50/p95 rollout and verifier latency;
3. tokens/s and GPU utilization from a remote vLLM/SGLang server;
4. cross-worker failover, circuit-breaker recovery, deadline miss rate, and hedged-vs-unhedged p95 latency;
5. checkpoint/resume correctness under worker interruption;
6. verifier-lag phase sweeps with fixed compute budgets;
7. comparison of fixed-cadence vs movement-triggered refresh;
8. policy-weight synchronization overhead between trainer and rollout workers.

A strong public result should include hardware, model revision, serving
version, seeds, profiler traces, and a reproducible command.

See `docs/systems_split_plan.md` for the criteria for extracting this package
into a standalone systems repository.
