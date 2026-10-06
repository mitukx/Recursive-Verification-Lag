#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-artifacts/triton-grpo}"
SIZES="${SIZES:-4096,16384,65536,262144,1048576}"
WARMUP="${WARMUP:-25}"
ITERATIONS="${ITERATIONS:-100}"

mkdir -p "$OUT"
nvidia-smi > "$OUT/nvidia-smi.txt"
python -m pip show torch > "$OUT/torch-package.txt"
python -m pip show triton > "$OUT/triton-package.txt"

python -m src.benchmark_triton_grpo   --sizes "$SIZES"   --warmup "$WARMUP"   --iterations "$ITERATIONS"   --output "$OUT/benchmark.json"

python -m src.manifest_artifacts   --output "$OUT/manifest.json"   "$OUT/nvidia-smi.txt"   "$OUT/torch-package.txt"   "$OUT/triton-package.txt"   "$OUT/benchmark.json"

python -m src.manifest_artifacts --verify "$OUT/manifest.json"
echo "Triton GRPO benchmark complete: $OUT"
