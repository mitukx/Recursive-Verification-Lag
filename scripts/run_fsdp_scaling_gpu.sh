#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-artifacts/fsdp-scaling}"
MODEL="${MODEL:-}"
REPLAY="${REPLAY:-}"
mkdir -p "$OUT"

python - <<'PY'
import torch
assert torch.cuda.device_count() >= 2, "at least two CUDA GPUs are required"
print("GPUs:", torch.cuda.device_count())
PY

nvidia-smi > "$OUT/nvidia-smi.txt"
python -m pip show torch > "$OUT/torch-package.txt"

EXTRA_ARGS=()
if [[ -n "$MODEL" ]]; then
  if [[ -z "$REPLAY" ]]; then
    echo "REPLAY is required when MODEL is set" >&2
    exit 2
  fi
  EXTRA_ARGS+=(--model "$MODEL" --replay "$REPLAY")
fi

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1   -m src.run_distributed_mini_lab   --mode fsdp   "${EXTRA_ARGS[@]}"   --output "$OUT/one-gpu"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2   -m src.run_distributed_mini_lab   --mode fsdp   "${EXTRA_ARGS[@]}"   --output "$OUT/two-gpu"

python -m src.summarize_distributed_scaling   --single "$OUT/one-gpu/distributed-report.json"   --multi "$OUT/two-gpu/distributed-report.json"   --output "$OUT/scaling-summary.json"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2   -m src.run_distributed_mini_lab   --mode fsdp   "${EXTRA_ARGS[@]}"   --resume-from "$OUT/two-gpu/sharded-checkpoint"   --output "$OUT/resumed-two-gpu"

python - "$OUT/resumed-two-gpu/distributed-report.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report.get("resumed_from_checkpoint") is True, report
print("resume acceptance passed")
PY

ARTIFACTS=(
  "$OUT/nvidia-smi.txt"
  "$OUT/torch-package.txt"
  "$OUT/one-gpu/distributed-report.json"
  "$OUT/two-gpu/distributed-report.json"
  "$OUT/resumed-two-gpu/distributed-report.json"
  "$OUT/scaling-summary.json"
)
if [[ -n "$REPLAY" ]]; then
  ARTIFACTS+=("$REPLAY")
  if [[ -f "$REPLAY.meta.json" ]]; then
    ARTIFACTS+=("$REPLAY.meta.json")
  fi
fi
mapfile -t GPU_REPORTS < <(find "$OUT" -name 'gpu-rank-*.json' -type f | sort)
ARTIFACTS+=("${GPU_REPORTS[@]}")
python -m src.manifest_artifacts --output "$OUT/manifest.json" "${ARTIFACTS[@]}"

python -m src.manifest_artifacts --verify "$OUT/manifest.json"
echo "FSDP scaling acceptance complete: $OUT"
