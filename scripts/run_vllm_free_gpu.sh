#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
CONCURRENCY="${CONCURRENCY:-1,2,4,8}"
REQUESTS="${REQUESTS:-24}"
MAX_TOKENS="${MAX_TOKENS:-64}"
OUT="${OUT:-artifacts/vllm-free-gpu}"

mkdir -p "$OUT"
python -m pip show vllm > "$OUT/vllm-package.txt" || {
  echo "vLLM is not installed. Run: pip install -r requirements-gpu.txt" >&2
  exit 2
}
nvidia-smi > "$OUT/nvidia-smi.txt"

vllm serve "$MODEL"   --host 127.0.0.1   --port "$PORT"   --dtype auto   --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"   --max-model-len "$MAX_MODEL_LEN"   > "$OUT/vllm-server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

python - "$PORT" <<'PY'
import sys, time, urllib.request
port = sys.argv[1]
url = f"http://127.0.0.1:{port}/v1/models"
deadline = time.time() + 300
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            if r.status == 200:
                print("vLLM ready")
                raise SystemExit(0)
    except Exception:
        time.sleep(2)
raise SystemExit("vLLM did not become ready within 300s")
PY

nvidia-smi   --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu   --format=csv,noheader,nounits   --loop=1 > "$OUT/gpu-telemetry.csv" &
MONITOR_PID=$!

python -m src.benchmark_vllm_serving   --endpoint "http://127.0.0.1:$PORT"   --model "$MODEL"   --requests "$REQUESTS"   --concurrency "$CONCURRENCY"   --max-tokens "$MAX_TOKENS"   --output-dir "$OUT/benchmark"

python -m src.manifest_artifacts   --output "$OUT/manifest.json"   "$OUT/nvidia-smi.txt"   "$OUT/vllm-package.txt"   "$OUT/gpu-telemetry.csv"   "$OUT/benchmark/summary.json"   "$OUT/benchmark/gpu-inventory.json"

python -m src.manifest_artifacts --verify "$OUT/manifest.json"
echo "GPU benchmark complete: $OUT"
