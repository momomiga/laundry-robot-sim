#!/usr/bin/env bash
# Set up a fresh GPU box (vast.ai etc.) in the repo root. Idempotent. Run on the remote from the repo root.
set -euo pipefail

SKIP_SYSTEM=0 SKIP_OLLAMA=0 SKIP_IPC=0
for a in "$@"; do
  case "$a" in
    --skip-system) SKIP_SYSTEM=1 ;;
    --skip-ollama) SKIP_OLLAMA=1 ;;
    --skip-ipc) SKIP_IPC=1 ;;
    -h|--help) echo "usage: $0 [--skip-system] [--skip-ollama] [--skip-ipc]"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PATH="$HOME/.local/bin:$HOME/.local/ollama/bin:$PATH"
step() { printf '\n== %s ==\n' "$*"; }
SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo"

# ---- 1. system packages ----
if [ "$SKIP_SYSTEM" = 0 ]; then
  step "apt packages"
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -qq
  # libosmesa6 = CPU rendering fallback only
  $SUDO apt-get install -y -qq ffmpeg libegl1 libgl1 libosmesa6 rsync curl ca-certificates
fi

# ---- 2. uv ----
if ! command -v uv >/dev/null; then
  step "install uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# ---- 3. EGL diagnosis ----
step "EGL diagnosis"
egl_ok=1
if [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  echo "ok: /usr/share/glvnd/egl_vendor.d/10_nvidia.json present"
else
  echo "missing: /usr/share/glvnd/egl_vendor.d/10_nvidia.json"; egl_ok=0
fi
if ldconfig -p 2>/dev/null | grep -q libEGL_nvidia; then
  echo "ok: libEGL_nvidia found"
else
  echo "missing: libEGL_nvidia"; egl_ok=0
fi
echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-<unset>}"
case "${NVIDIA_DRIVER_CAPABILITIES:-}" in
  *all*|*graphics*) ;;
  *) echo "note: NVIDIA_DRIVER_CAPABILITIES lacks graphics/all (harmless on a bare host, fatal for EGL in a container)" ;;
esac
if [ "$egl_ok" = 0 ]; then
  cat >&2 <<'WARN'
WARNING: the NVIDIA EGL vendor is not visible. Rendering will fall back to OSMesa (CPU) and be very slow.
  Fix: create the instance with the docker option  -e NVIDIA_DRIVER_CAPABILITIES=all
  (it cannot be changed on a running container; recreate the instance).
WARN
fi

# ---- 4. python envs ----
step "uv sync (.venv)"
uv sync

if [ "$SKIP_IPC" = 0 ]; then
  step ".venv-ipc"
  [ -x .venv-ipc/bin/python ] || uv venv .venv-ipc --python 3.12
  TORCH_PIN=$(grep -oE 'torch==[0-9][^ ]*' requirements-ipc.txt | head -n 1)
  # torch first and alone (see header of requirements-ipc.txt)
  uv pip install --python .venv-ipc/bin/python --no-deps \
    --index-url https://download.pytorch.org/whl/cu128 "${TORCH_PIN:-torch==2.11.0+cu128}"
  uv pip install --python .venv-ipc/bin/python -r requirements-ipc.txt
fi

# ---- 5. ollama ----
if [ "$SKIP_OLLAMA" = 0 ]; then
  step "ollama"
  command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
  if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    for _ in $(seq 1 30); do
      curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break
      sleep 1
    done
    curl -sf http://127.0.0.1:11434/api/tags >/dev/null || { echo "ollama did not start; see /tmp/ollama.log" >&2; exit 1; }
  fi
  echo "ollama serve up (log: /tmp/ollama.log)"
  ollama pull qwen3-vl:8b-instruct
fi

# ---- 6. smoke ----
step "smoke"
SMOKE_ARGS=()
[ "$SKIP_IPC" = 1 ] && SMOKE_ARGS+=(--skip-ipc)
[ "$SKIP_OLLAMA" = 1 ] && SMOKE_ARGS+=(--skip-vlm)
bash scripts/remote/smoke.sh "${SMOKE_ARGS[@]}"
