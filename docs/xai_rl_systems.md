# RLVR Systems Track

This extension turns the RVL research repository into a small end-to-end
post-training systems project. The goal is to make engineering ability
observable, not to claim frontier-scale performance from a laptop.

## What is implemented

- bounded-concurrency asynchronous rollout scheduling;
- a backend protocol plus a trainable CPU toy policy;
- an OpenAI-compatible HTTP adapter for vLLM/SGLang servers;
- deterministic reward verification;
- grouped reward normalization and a GRPO-style policy-gradient update;
- a policy-movement/staleness refresh controller aligned with RVL;
- throughput/latency/reward telemetry;
- an end-to-end Mac/CPU demo and unit tests.

The CPU backend is deliberately small enough for CI while exercising the exact
control plane used by a remote inference backend. This separates systems logic
from GPU availability.

## Run on a Mac

```bash
python -m unittest tests.test_rlvr_systems -v
python -m src.run_rlvr_systems_demo --rounds 12 --samples 32
```

## Point at a GPU serving stack

`VLLMHTTPBackend` targets the standard `/v1/completions` endpoint exposed by
vLLM and compatible servers. Start the GPU server separately, then construct:

```python
from src.rvl_systems.backends import VLLMHTTPBackend

backend = VLLMHTTPBackend(
    endpoint="http://127.0.0.1:8000",
    model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
)
```

The next production step is a model-training adapter that consumes
`TrainRecord` objects with PyTorch/DeepSpeed/FSDP, plus batched token-level
logprobs from the serving backend. The core rollout/verifier/refresh/telemetry
interfaces should remain unchanged.

## Engineering benchmark plan

Report concrete systems numbers instead of repository size:

1. synchronous vs asynchronous rollout throughput;
2. p50/p95 rollout and verifier latency;
3. tokens/s and GPU utilization from a remote vLLM server;
4. checkpoint/resume correctness under worker interruption;
5. verifier-lag phase sweeps with fixed compute budgets;
6. comparison of fixed-cadence vs movement-triggered refresh.

A strong public result should include the hardware, model revision, serving
version, seeds, profiler traces, and a reproducible command.
