# Numerical parity gates

The repository separates algorithmic numerical checks from GPU-kernel claims.

`precision_parity.py` emulates IEEE FP16 and BF16 rounding on CPU and compares
importance-ratio calculations against FP32 reference values. CI gates maximum
relative ratio error on a fixed representative grid.

This catches accidental changes in clipping/log-probability arithmetic and
provides a stable report schema. It is not evidence that a CUDA/Triton kernel
matches PyTorch. When GPU kernels are added, the same report should be populated
from actual eager/compiled/kernel outputs and tighter, hardware-scoped
tolerances.


## Low-precision RL and INT8 contracts

The GPU evidence workflow also runs the same token-exact Qwen replay through one
GRPO backward pass in FP32, BF16, and FP16. The report records loss, gradient
norm, behavior-KL estimate, max log-ratio, tokens/s, and peak allocated GPU
memory for each dtype. Non-finite metrics fail the evidence bundle. This is a
single-step numerical/performance probe, not a stability claim for long runs.

Separately, `quantization.py` provides a dependency-free reference contract
for per-output-channel symmetric INT8 weight-only quantization. CI measures
linear-output error against FP32 and enforces broad numerical regression gates.
The reference implementation dequantizes before the matrix multiply and
therefore makes **no INT8 speedup claim**; its purpose is to make scale, clipping,
zero-row, storage, and error semantics explicit before an optimized kernel or
engine backend is introduced.
