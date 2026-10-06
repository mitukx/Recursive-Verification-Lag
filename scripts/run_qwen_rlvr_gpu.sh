#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
TRAIN_LIMIT="${TRAIN_LIMIT:-32}"
EVAL_LIMIT="${EVAL_LIMIT:-32}"
STEPS="${STEPS:-8}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-4}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
PRECISION="${PRECISION:-auto}"
OBJECTIVE_BACKEND="${OBJECTIVE_BACKEND:-torch}"
OUT="${OUT:-artifacts/qwen-rlvr}"

mkdir -p "$OUT"
nvidia-smi > "$OUT/nvidia-smi.txt"
python -m pip show torch > "$OUT/torch-package.txt"
python -m pip show transformers > "$OUT/transformers-package.txt"
python -m pip show datasets > "$OUT/datasets-package.txt"

python -m src.run_qwen_rlvr_experiment   --model "$MODEL"   --dataset gsm8k   --train-limit "$TRAIN_LIMIT"   --eval-limit "$EVAL_LIMIT"   --steps "$STEPS"   --prompts-per-step "$PROMPTS_PER_STEP"   --samples-per-prompt "$SAMPLES_PER_PROMPT"   --max-new-tokens "$MAX_NEW_TOKENS"   --precision "$PRECISION"   --objective-backend "$OBJECTIVE_BACKEND"   --output-dir "$OUT/run"

python -m src.manifest_artifacts   --output "$OUT/manifest.json"   "$OUT/nvidia-smi.txt"   "$OUT/torch-package.txt"   "$OUT/transformers-package.txt"   "$OUT/datasets-package.txt"   "$OUT/run/benchmark.json"   "$OUT/run/history.json"   "$OUT/run/telemetry.json"   "$OUT/run/predictions-before.jsonl"   "$OUT/run/predictions-after.jsonl"

python -m src.manifest_artifacts --verify "$OUT/manifest.json"
echo "Qwen RLVR experiment complete: $OUT"
