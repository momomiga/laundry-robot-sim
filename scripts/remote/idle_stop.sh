#!/usr/bin/env bash
# Stop this vast.ai instance after a long idle period. Runs ON the remote.
#   nohup bash scripts/remote/idle_stop.sh > /workspace/jobs/idle_stop.log 2>&1 &
# --status prints "idle" / "busy: <reasons>" (exit 0 / 1) and is reused by collect_and_stop.sh.
# Never prints CONTAINER_API_KEY (CLI output is redacted).
set -euo pipefail

usage() {
  cat <<USAGE
usage: $0 [--status] [--interval SEC=60] [--idle-min M=90] [--max-hours H] [--dry-run]
          [--no-gpu-check] [--lock PATH=/tmp/house_robot-gpu.lock] [--jobs-dir DIR=/workspace/jobs]

Busy (any): GPU lock held; nvidia-smi lists compute apps; a queue process
  (\$JOBS_DIR/q_*.sh) is alive; \$JOBS_DIR/KEEPALIVE exists.
Watch mode: stops the instance after M consecutive idle minutes. --max-hours H is a hard cap
  since start (stops even if busy; KEEPALIVE does not override it). --dry-run logs
  "WOULD STOP (<reason>)" and exits 0.
Exit codes: 0 idle / done, 1 busy (--status), 2 usage/missing tool, 4 missing CONTAINER_ID/CONTAINER_API_KEY.
USAGE
  exit "${1:-2}"
}

MODE=watch INTERVAL=60 IDLE_MIN=90 MAX_HOURS="" DRY=0 GPU_CHECK=1
LOCK=/tmp/house_robot-gpu.lock JOBS_DIR=/workspace/jobs
while [ "$#" -gt 0 ]; do
  case "$1" in
    --status) MODE=status ;;
    --interval) INTERVAL=${2:?}; shift ;;
    --idle-min) IDLE_MIN=${2:?}; shift ;;
    --max-hours) MAX_HOURS=${2:?}; shift ;;
    --dry-run) DRY=1 ;;
    --no-gpu-check) GPU_CHECK=0 ;;
    --lock) LOCK=${2:?}; shift ;;
    --jobs-dir) JOBS_DIR=${2:?}; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1" >&2; usage ;;
  esac
  shift
done
num() { [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "not a number: $1" >&2; exit 2; }; }
num "$INTERVAL"; num "$IDLE_MIN"; [ -z "$MAX_HOURS" ] || num "$MAX_HOURS"
IDLE_S=$(awk -v m="$IDLE_MIN" 'BEGIN{print int(m*60)}')
MAX_S=""; [ -z "$MAX_HOURS" ] || MAX_S=$(awk -v h="$MAX_HOURS" 'BEGIN{print int(h*3600)}')

if [ "$GPU_CHECK" = 1 ] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi not found (use --no-gpu-check to skip the GPU process check)" >&2
  exit 2
fi

REASONS=""
# Sets REASONS; returns 0 if idle, 1 if busy.
check_status() {
  local r=() apps
  flock -n "$LOCK" true 2>/dev/null || r+=("gpu lock held")
  if [ "$GPU_CHECK" = 1 ]; then
    if apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); then
      [ -z "$apps" ] || r+=("gpu compute apps: $(echo "$apps" | tr '\n' ' ' | sed 's/ *$//')")
    else
      r+=("nvidia-smi failed")
    fi
  fi
  pgrep -f "$JOBS_DIR/q_[^/ ]*\.sh" >/dev/null 2>&1 && r+=("queue process alive")
  [ ! -e "$JOBS_DIR/KEEPALIVE" ] || r+=("KEEPALIVE present")
  REASONS=""
  if [ "${#r[@]}" -gt 0 ]; then
    REASONS=$(printf '%s, ' "${r[@]}"); REASONS=${REASONS%, }
    return 1
  fi
  return 0
}

if [ "$MODE" = status ]; then
  if check_status; then echo idle; exit 0; else echo "busy: $REASONS"; exit 1; fi
fi

log() { echo "$(date '+%F %T') $*"; }

redact_run() { # run a command, print its output without the API key
  local out rc=0
  out=$("$@" 2>&1) || rc=$?
  printf '%s\n' "${out//"$CONTAINER_API_KEY"/<redacted>}"
  return "$rc"
}

need_env() {
  if [ -z "${CONTAINER_ID:-}" ] || [ -z "${CONTAINER_API_KEY:-}" ]; then
    echo "error: CONTAINER_ID / CONTAINER_API_KEY not set; cannot stop the instance" >&2
    exit 4
  fi
}
[ "$DRY" = 1 ] || need_env

do_stop() { # $1 = reason
  log "decision: stop ($1)"
  sync || true
  if [ "$DRY" = 1 ]; then log "WOULD STOP ($1)"; exit 0; fi
  local cli
  cli=$(command -v vastai || echo /opt/instance-tools/bin/vastai)
  if redact_run "$cli" stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"; then
    log "stop command issued"; exit 0
  fi
  log "stop command failed; will retry"
}

log "watching: idle-min=$IDLE_MIN interval=$INTERVAL max-hours=${MAX_HOURS:-off} lock=$LOCK jobs-dir=$JOBS_DIR dry-run=$DRY"
START=$SECONDS IDLE_SINCE="" LAST=""
while true; do
  if check_status; then
    STATE=idle
    [ -n "$IDLE_SINCE" ] || IDLE_SINCE=$SECONDS
  else
    STATE="busy: $REASONS"; IDLE_SINCE=""
  fi
  [ "$STATE" = "$LAST" ] || { log "$STATE"; LAST=$STATE; }
  if [ -n "$MAX_S" ] && [ $((SECONDS - START)) -ge "$MAX_S" ]; then
    do_stop "max-hours $MAX_HOURS reached"
  elif [ -n "$IDLE_SINCE" ] && [ $((SECONDS - IDLE_SINCE)) -ge "$IDLE_S" ]; then
    do_stop "idle for $IDLE_MIN min"
  fi
  sleep "$INTERVAL"
done
