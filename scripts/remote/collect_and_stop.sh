#!/usr/bin/env bash
# Wait for remote GPU queue(s) to finish, collect + verify the outputs, optionally stop the instance. Run locally.
# Never prints CONTAINER_API_KEY (it only exists in the remote environment; CLI output is redacted there).
# shellcheck disable=SC2016,SC2001
# (the remote scripts are intentionally single-quoted)
set -euo pipefail

usage() {
  cat <<USAGE
usage: $0 [--stop] [--auth-check] [--poll SEC=300] [--max-ssh-fail N=6] [--remote-dir DIR=/workspace/house_robot]
          [--jobs-dir DIR=/workspace/jobs] [--dest DIR=outputs/remote] [--lock PATH]
          <user@host|local> <port> <queue_log>...

1. poll until every queue_log has a QUEUE_DONE / QUEUE_ABORT line and idle_stop.sh --status says idle
2. sync_from.sh <remote-dir>/outputs -> DEST, and <jobs-dir>/*.log|*.sh -> DEST/jobs (idle_stop*.log excluded)
3. verify every remote file exists locally with the same size (mismatch: exit 1, no stop)
4. --stop: read-only show-instance check, 'vastai stop' (on the remote), then wait until ssh is unreachable
--auth-check: only the read-only show-instance check. "local" runs everything on this machine
(--stop / --auth-check refuse in local mode). --lock is forwarded to idle_stop.sh --status.
Exit codes: 0 ok, 1 verify failed, 2 usage/broken status, 3 unreachable (instance may already be stopped),
4 auth/stop failed, 5 busy again before the stop (not stopped).
USAGE
  exit "${1:-2}"
}

STOP=0 AUTH=0 POLL=300 MAXFAIL=6 REMOTE_DIR=/workspace/house_robot JOBS_DIR=/workspace/jobs DEST="" LOCK="" POS=()
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
while [ "$#" -gt 0 ]; do
  case "$1" in
    --stop) STOP=1 ;;
    --auth-check) AUTH=1 ;;
    --poll) POLL=${2:?}; shift ;;
    --max-ssh-fail) MAXFAIL=${2:?}; shift ;;
    --remote-dir) REMOTE_DIR=${2:?}; shift ;;
    --jobs-dir) JOBS_DIR=${2:?}; shift ;;
    --dest) DEST=${2:?}; shift ;;
    --lock) LOCK=${2:?}; shift ;;
    -h|--help) usage 0 ;;
    -*) echo "unknown option: $1" >&2; usage ;;
    *) POS+=("$1") ;;
  esac
  shift
done
[ "${#POS[@]}" -ge 2 ] || usage
HOST=${POS[0]} PORT=${POS[1]} LOGS=("${POS[@]:2}")
[ -n "$DEST" ] || DEST=$ROOT/outputs/remote
[[ "$POLL" =~ ^[0-9]+$ && "$MAXFAIL" =~ ^[0-9]+$ ]] || { echo "--poll/--max-ssh-fail must be integers" >&2; exit 2; }
if [ "$HOST" = local ] && { [ "$STOP" = 1 ] || [ "$AUTH" = 1 ]; }; then
  echo "--stop/--auth-check are not allowed in local mode" >&2; exit 2
fi
[ "$AUTH" = 1 ] || [ "${#LOGS[@]}" -ge 1 ] || { echo "at least one queue_log is required" >&2; usage; }

SSH_OPTS=(-p "$PORT" -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new)
ts() { date '+%F %T'; }
say() { echo "$(ts) $*"; }

# POSIX single-quote escaping (the remote login shell may not be bash).
sq() { local q="'"; printf "'%s'" "${1//$q/$q\\$q$q}"; }

# rsh <script> [args...]: run a bash script on the remote (or locally), 90 s limit; args are quoted.
rsh() {
  local script=$1 cmd a; shift
  cmd="bash -c $(sq "$script") _"
  for a in "$@"; do cmd+=" $(sq "$a")"; done
  if [ "$HOST" = local ]; then
    timeout 90 bash -c "$cmd"
  else
    timeout 90 ssh "${SSH_OPTS[@]}" "$HOST" "$cmd"
  fi
}

# Read-only show-instance (status fields only). Runs on the remote, where the credentials live.
SHOW_SCRIPT='
[ -n "${CONTAINER_ID:-}" ] && [ -n "${CONTAINER_API_KEY:-}" ] || { echo "CONTAINER_ID/CONTAINER_API_KEY not set" >&2; exit 4; }
cli=$(command -v vastai || echo /opt/instance-tools/bin/vastai)
rc=0; out=$("$cli" show instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" --raw 2>/dev/null) || rc=$?
out=${out//"$CONTAINER_API_KEY"/<redacted>}
[ "$rc" -eq 0 ] || { echo "vastai show instance failed (rc=$rc)" >&2; exit 4; }
f=$(printf "%s\n" "$out" | grep -oE "\"(actual_status|intended_status|cur_state|status_msg)\": *(\"[^\"]*\"|null)" || true)
[ -n "$f" ] || { echo "show instance: no status fields in the response" >&2; exit 4; }
printf "%s\n" "$f"
'
show_check() {
  local out
  if out=$(rsh "$SHOW_SCRIPT"); then
    say "show instance: $(echo "$out" | tr '\n' ' ')"
  else
    say "show-instance check failed (auth/credentials/ssh)"; return 1
  fi
}

if [ "$AUTH" = 1 ]; then
  show_check || exit 4
  exit 0
fi

# ---- 1. poll ----
LOG_SCRIPT='
for f in "$@"; do
  if [ -f "$f" ]; then
    m=$(grep -E "^(QUEUE_DONE$|QUEUE_ABORT)" "$f" | tail -n 1 || true)
    s=$(grep "^=== " "$f" | tail -n 1 || true)
  else m=""; s="(log missing)"; fi
  printf "LOG|%s|%s|%s\n" "$f" "$m" "$s"
done
st=$(bash "$STATUS_SH" --status --jobs-dir "$JOBS" ${LOCK:+--lock "$LOCK"} 2>&1 | tail -n 1) || true
printf "STATUS|%s\n" "${st:-no status output}"
'
STATUS_SCRIPT='export STATUS_SH="$1" JOBS="$2" LOCK="$3"; shift 3; '"$LOG_SCRIPT"
STATUS_SH="$REMOTE_DIR/scripts/remote/idle_stop.sh"
LAST="" FAILS=0 OUT=""
while true; do
  if OUT=$(rsh "$STATUS_SCRIPT" "$STATUS_SH" "$JOBS_DIR" "$LOCK" "${LOGS[@]}"); then
    FAILS=0
    PENDING=0 STATUS=""
    while IFS='|' read -r kind a b _; do
      case "$kind" in
        LOG) [ -n "$b" ] || PENDING=1 ;;
        STATUS) STATUS=$a ;;
      esac
    done <<<"$OUT"
    case "$STATUS" in
      idle|busy*) ;;
      *) echo "error: idle_stop.sh --status gave '$STATUS' (is the code synced to $REMOTE_DIR?)" >&2; exit 2 ;;
    esac
    # status text without volatile timestamps: LOG lines and the status
    CUR=$(grep -v '^$' <<<"$OUT" | sed 's/^LOG|\([^|]*\)|\([^|]*\)|/\1: [\2] /; s/^STATUS|/status: /')
    if [ "$CUR" != "$LAST" ]; then say "state:"; echo "$CUR" | sed 's/^/  /'; LAST=$CUR; fi
    if [ "$PENDING" = 0 ] && [ "$STATUS" = idle ]; then break; fi
  else
    FAILS=$((FAILS + 1))
    say "ssh/poll failure $FAILS/$MAXFAIL"
    if [ "$FAILS" -ge "$MAXFAIL" ]; then
      echo "error: $MAXFAIL consecutive failures; the instance may already be stopped (nothing else done)" >&2
      exit 3
    fi
  fi
  sleep "$POLL"
done
say "queue(s) finished and idle"

# ---- 2. collect ----
mkdir -p "$DEST/jobs"
"$ROOT/scripts/remote/sync_from.sh" "$HOST" "$PORT" "$REMOTE_DIR" "$DEST"
JOB_RSYNC=(rsync -a --exclude='idle_stop*.log' --include='*.log' --include='*.sh' --exclude='*')
if [ "$HOST" = local ]; then
  "${JOB_RSYNC[@]}" "$JOBS_DIR/" "$DEST/jobs/"
else
  "${JOB_RSYNC[@]}" -z -e "ssh -p $PORT -o BatchMode=yes -o ConnectTimeout=20" "$HOST:$JOBS_DIR/" "$DEST/jobs/"
fi

# ---- 3. verify ----
BAD=0 NFILES=0 NBYTES=0
verify_list() { # $1 = label, $2 = local dir, stdin = "relpath<TAB>size" lines of the remote side
  local p s ls
  while IFS=$'\t' read -r p s; do
    [ -n "$p" ] || continue
    NFILES=$((NFILES + 1)); NBYTES=$((NBYTES + s))
    if [ ! -f "$2/$p" ]; then echo "MISSING  $1/$p (remote $s bytes)"; BAD=$((BAD + 1)); continue; fi
    ls=$(stat -c %s "$2/$p")
    if [ "$ls" != "$s" ]; then echo "MISMATCH $1/$p remote=$s local=$ls"; BAD=$((BAD + 1)); fi
  done
}
OUT_LIST=$(rsh 'cd "$1/outputs" && find . -type f -printf "%P\t%s\n" | sort' "$REMOTE_DIR") || { echo "error: cannot list remote outputs" >&2; exit 3; }
JOB_LIST=$(rsh 'cd "$1" && find . -maxdepth 1 -type f \( -name "*.log" -o -name "*.sh" \) ! -name "idle_stop*.log" -printf "%P\t%s\n" | sort' "$JOBS_DIR") || { echo "error: cannot list remote jobs" >&2; exit 3; }
verify_list outputs "$DEST" <<<"$OUT_LIST"
verify_list jobs "$DEST/jobs" <<<"$JOB_LIST"
if [ "$BAD" -gt 0 ]; then
  echo "verify FAILED: $BAD file(s) missing or with a different size; not stopping" >&2
  exit 1
fi
say "verify OK: $NFILES files, $NBYTES bytes under $DEST"
for l in "${LOGS[@]}"; do
  m=$(grep -F "LOG|$l|" <<<"$OUT" | head -n 1 | cut -d'|' -f3 || true)
  case "$m" in
    QUEUE_DONE) echo "  $l: DONE" ;;
    QUEUE_ABORT*) echo "  $l: ABORT ${m#QUEUE_ABORT}" ;;
    *) echo "  $l: (no marker)" ;;
  esac
done
[ "$STOP" = 1 ] || { say "done (no --stop: instance left running)"; exit 0; }

# ---- 4. stop ----
# A new queue may have started while collecting: re-check right before stopping.
ST=$(rsh 'bash "$1" --status --jobs-dir "$2" ${3:+--lock "$3"} 2>&1 | tail -n 1' "$STATUS_SH" "$JOBS_DIR" "$LOCK") || true
if [ "$ST" != idle ]; then say "not stopping: remote is '${ST:-unreachable}' now"; exit 5; fi
show_check || exit 4
STOP_SCRIPT='
[ -n "${CONTAINER_ID:-}" ] && [ -n "${CONTAINER_API_KEY:-}" ] || { echo "CONTAINER_ID/CONTAINER_API_KEY not set" >&2; exit 4; }
cli=$(command -v vastai || echo /opt/instance-tools/bin/vastai)
sync || true
rc=0; out=$("$cli" stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" 2>&1) || rc=$?
printf "%s\n" "${out//"$CONTAINER_API_KEY"/<redacted>}"
exit "$rc"
'
say "stopping the instance"
# The connection can drop while the container goes down, so the outcome is judged by reachability below.
rsh "$STOP_SCRIPT" || say "stop command returned non-zero (or the connection dropped); checking reachability"
DEADLINE=$((SECONDS + 300)) DOWN=0
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  if rsh true >/dev/null 2>&1; then DOWN=0; else DOWN=$((DOWN + 1)); fi
  if [ "$DOWN" -ge 3 ]; then say "instance unreachable: stopped"; exit 0; fi
  sleep 20
done
echo "error: still reachable 5 min after the stop command" >&2
show_check || true
exit 4
