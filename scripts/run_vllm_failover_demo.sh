#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT0="${PORT0:-8000}"
PORT1="${PORT1:-8001}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.40}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
REQUESTS="${REQUESTS:-64}"
MAX_TOKENS="${MAX_TOKENS:-128}"
OUT="${OUT:-artifacts/vllm-failover}"

mkdir -p "$OUT"

vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT0"   --dtype auto --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"   --max-model-len "$MAX_MODEL_LEN" > "$OUT/server-0.log" 2>&1 &
PID0=$!
vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT1"   --dtype auto --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"   --max-model-len "$MAX_MODEL_LEN" > "$OUT/server-1.log" 2>&1 &
PID1=$!

cleanup() {
  kill "$PID0" "$PID1" 2>/dev/null || true
}
trap cleanup EXIT

python - "$PORT0" "$PORT1" <<'PY'
import sys, time, urllib.request
ports = sys.argv[1:]
deadline = time.time() + 300
pending = set(ports)
while pending and time.time() < deadline:
    for port in list(pending):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
                if r.status == 200:
                    pending.remove(port)
        except Exception:
            pass
    time.sleep(2)
if pending:
    raise SystemExit(f"servers not ready: {sorted(pending)}")
print("both vLLM servers ready")
PY

python -m src.benchmark_vllm_failure   --endpoint "http://127.0.0.1:$PORT0"   --endpoint "http://127.0.0.1:$PORT1"   --model "$MODEL"   --kill-pid "$PID0"   --requests "$REQUESTS"   --max-tokens "$MAX_TOKENS"   --output "$OUT/failover.json"

echo "Failover benchmark complete: $OUT/failover.json"
