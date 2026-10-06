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
