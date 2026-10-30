#!/usr/bin/env bash
# Smoke test, run on the remote from anywhere. Stops at the first failing stage (exit != 0).
set -euo pipefail

SKIP_IPC=0 SKIP_VLM=0
for a in "$@"; do
  case "$a" in
    --skip-ipc) SKIP_IPC=1 ;;
    --skip-vlm) SKIP_VLM=1 ;;
    -h|--help) echo "usage: $0 [--skip-ipc] [--skip-vlm]"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PATH="$HOME/.local/bin:$HOME/.local/ollama/bin:$PATH"
LOCK=/tmp/house_robot-gpu.lock
gpu() { if command -v flock >/dev/null; then flock "$LOCK" "$@"; else "$@"; fi; }

stage() {  # stage <n> <title> <cmd...>
  local n=$1 title=$2; shift 2
  local t0=$SECONDS
  echo "--- stage $n: $title ---"
  if "$@"; then
    echo "[PASS] stage $n: $title ($((SECONDS - t0)) s)"
  else
    echo "[FAIL] stage $n: $title ($((SECONDS - t0)) s)" >&2
    exit 1
  fi
}

stage 1 "GPU + render (check_env.py)" gpu uv run python scripts/check_env.py
stage 2 "render speed / backend (EGL vs OSMesa)" gpu uv run python scripts/remote/render_bench.py

if [ "$SKIP_IPC" = 1 ]; then
  echo "[SKIP] stage 3: IPC"
else
  # Shortest existing condition: stop after the cloth has settled in the basket, no video.
  stage 3 "IPC short run (cloth_ipc_washer_test.py --stop-after settle)" \
    gpu .venv-ipc/bin/python scripts/cloth_ipc_washer_test.py --stop-after settle --no-video \
      --json outputs/remote_smoke_ipc.json
fi

if [ "$SKIP_VLM" = 1 ]; then
  echo "[SKIP] stage 4: VLM"
else
  stage 4 "VLM ping (ollama /api/chat)" uv run python scripts/remote/vlm_ping.py
fi
echo "SMOKE OK"
