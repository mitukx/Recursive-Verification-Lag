#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-7ae557604adf67be50417f59c2c2f167def9a775}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
TASKS="${TASKS:-8}"
SAMPLES="${SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
TEMPERATURE="${TEMPERATURE:-0.8}"
OUT="${OUT:-artifacts/vllm-token-exact-bridge}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"

mkdir -p "$OUT"
python -m pip show vllm > "$OUT/vllm-package.txt"
python -m pip show torch > "$OUT/torch-package.txt"
python -m pip show transformers > "$OUT/transformers-package.txt"
python -m pip show datasets > "$OUT/datasets-package.txt"
python -m pip freeze > "$OUT/dependencies.txt"
nvidia-smi > "$OUT/nvidia-smi-before.txt"

MODEL_PATH="$(
  MODEL_REPO="$MODEL_REPO" MODEL_REVISION="$MODEL_REVISION" python - <<'PY'
import os
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    revision=os.environ["MODEL_REVISION"],
))
PY
)"
printf '%s\n' "$MODEL_REPO" > "$OUT/model-repo.txt"
printf '%s\n' "$MODEL_REVISION" > "$OUT/model-revision.txt"
printf '%s\n' "$MODEL_PATH" > "$OUT/model-snapshot-path.txt"

SERVER_PID=""
MONITOR_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL_PATH"   --host 127.0.0.1   --port "$PORT"   --served-model-name "$MODEL_REPO"   --dtype float32   --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"   --max-model-len "$MAX_MODEL_LEN"   > "$OUT/vllm-server.log" 2>&1 &
SERVER_PID=$!

python - "$PORT" <<'PY'
import sys, time, urllib.request
port = sys.argv[1]
url = f"http://127.0.0.1:{port}/v1/models"
deadline = time.time() + 300
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            if response.status == 200:
                raise SystemExit(0)
    except Exception:
        time.sleep(2)
raise SystemExit("vLLM did not become ready within 300 seconds")
PY

nvidia-smi   --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu   --format=csv,noheader,nounits   --loop-ms=500 > "$OUT/gpu-serving.csv" &
MONITOR_PID=$!

python -m src.generate_vllm_token_exact_replay   --endpoint "http://127.0.0.1:$PORT"   --model "$MODEL_REPO"   --tasks "$TASKS"   --samples "$SAMPLES"   --max-new-tokens "$MAX_NEW_TOKENS"   --temperature "$TEMPERATURE"   --output "$OUT/replay.jsonl"

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
MONITOR_PID=""
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""

# Ensure the server process has released its CUDA context before loading the learner.
for _ in $(seq 1 30); do
  if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q .; then
    break
  fi
  sleep 1
done
nvidia-smi > "$OUT/nvidia-smi-between.txt"

nvidia-smi   --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu   --format=csv,noheader,nounits   --loop-ms=500 > "$OUT/gpu-training.csv" &
MONITOR_PID=$!

CUDA_VISIBLE_DEVICES="$GPU" python -m src.train_vllm_token_exact_replay   --model "$MODEL_PATH"   --replay "$OUT/replay.jsonl"   --precision fp32   --learning-rate 1e-6   --max-preupdate-log-ratio 0.20   --output "$OUT/train-report.json"

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
MONITOR_PID=""
nvidia-smi > "$OUT/nvidia-smi-after.txt"

python -m src.validate_vllm_token_exact_bridge   --replay "$OUT/replay.jsonl"   --metadata "$OUT/replay.jsonl.meta.json"   --train-report "$OUT/train-report.json"   --model-repo "$MODEL_REPO"   --model-revision "$MODEL_REVISION"   --output "$OUT/validation.json"

python -m src.manifest_artifacts   --output "$OUT/manifest.json"   "$OUT/vllm-package.txt"   "$OUT/torch-package.txt"   "$OUT/transformers-package.txt"   "$OUT/datasets-package.txt"   "$OUT/dependencies.txt"   "$OUT/nvidia-smi-before.txt"   "$OUT/nvidia-smi-between.txt"   "$OUT/nvidia-smi-after.txt"   "$OUT/model-repo.txt"   "$OUT/model-revision.txt"   "$OUT/model-snapshot-path.txt"   "$OUT/vllm-server.log"   "$OUT/gpu-serving.csv"   "$OUT/gpu-training.csv"   "$OUT/replay.jsonl"   "$OUT/replay.jsonl.meta.json"   "$OUT/train-report.json"   "$OUT/validation.json"

python -m src.manifest_artifacts --verify "$OUT/manifest.json"
echo "Token-exact vLLM -> GRPO bridge evidence complete: $OUT"
