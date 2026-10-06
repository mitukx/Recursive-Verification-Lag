# Fused Triton GRPO surrogate

The training stack now contains an optional Triton backend for the tokenwise
clipped GRPO/PPO surrogate used after current token log-probabilities have been
computed.

## Fused work

One Triton forward kernel computes, per token:

- clamped log importance ratio;
- exponentiated importance ratio;
- clipped ratio;
- minimum clipped surrogate;
- clip indicator;
- absolute log-ratio diagnostic;
- behavior-KL diagnostic;
- local derivative with respect to the current log-probability.

The custom autograd backward uses a second Triton kernel to multiply upstream
gradients by the saved local derivative. Old behavior log-probabilities and
advantages remain non-differentiable inputs.

This does not fuse the transformer log-softmax/gather itself, so it should be
described as an elementwise GRPO hot-path fusion rather than an end-to-end RL
kernel.

## Correctness

`benchmark_triton_grpo.py` compares both forward surrogate values and gradients
against a PyTorch eager reference before reporting timing. The benchmark aborts
if either maximum absolute error exceeds the configured hard-coded acceptance
threshold.

## Running on a GPU

```bash
pip install -r requirements-gpu.txt
bash scripts/run_triton_grpo_gpu.sh
```

The result captures the GPU model, PyTorch/Triton package versions, eager and
Triton latency at multiple token counts, speedup, numerical errors and an
artifact checksum manifest.

No speedup is claimed in the repository until a real CUDA run is committed as
evidence. Small tensors may be slower because launch overhead dominates; the
portfolio-relevant result is the measured crossover and large-token regime.
