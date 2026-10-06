#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-artifacts/fsdp-scaling}"
mkdir -p "$OUT"

python - <<'PY'
import torch
assert torch.cuda.device_count() >= 2, "at least two CUDA GPUs are required"
print("GPUs:", torch.cuda.device_count())
PY

nvidia-smi > "$OUT/nvidia-smi.txt"
python -m pip show torch > "$OUT/torch-package.txt"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1   -m src.run_distributed_mini_lab   --mode fsdp   --output "$OUT/one-gpu"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2   -m src.run_distributed_mini_lab   --mode fsdp   --output "$OUT/two-gpu"

python -m src.summarize_distributed_scaling   --single "$OUT/one-gpu/distributed-report.json"   --multi "$OUT/two-gpu/distributed-report.json"   --output "$OUT/scaling-summary.json"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2   -m src.run_distributed_mini_lab   --mode fsdp   --resume-from "$OUT/two-gpu/sharded-checkpoint"   --output "$OUT/resumed-two-gpu"

python - "$OUT/resumed-two-gpu/distributed-report.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report.get("resumed_from_checkpoint") is True, report
print("resume acceptance passed")
PY

python -m src.manifest_artifacts   --output "$OUT/manifest.json"   "$OUT/nvidia-smi.txt"   "$OUT/torch-package.txt"   "$OUT/one-gpu/distributed-report.json"   "$OUT/two-gpu/distributed-report.json"   "$OUT/resumed-two-gpu/distributed-report.json"   "$OUT/scaling-summary.json"

python -m src.manifest_artifacts --verify "$OUT/manifest.json"
echo "FSDP scaling acceptance complete: $OUT"
