#!/usr/bin/env bash
# Send the working tree to a remote host (e.g. a vast.ai instance) with rsync over ssh. Run locally.
set -euo pipefail

usage() {
  cat <<USAGE
usage: $0 [--check] [--force-git] <user@host|local> <port> [remote_dir=/workspace/house_robot]

  <user@host>  e.g. root@ssh5.vast.ai ; the literal "local" copies into a local directory
               (port is ignored; remote_dir is then required and used as the destination).
  --check      rsync --dry-run only: list what would be sent and abort (exit 1) if anything
               forbidden (private/, outputs/, .venv*, symlinks to agent-ops state) is in the list.
  --force-git  replace the remote .git even if it already has one (discards remote-only commits).

Never sent: private/ outputs/ .venv*/ __pycache__/ AGENTS.md CLAUDE.md docs/knowledge.
A forbidden-path check on the rsync dry run always runs before the real transfer.
.git is a worktree pointer file here, so the shared git dir is copied as a real .git
(first transfer only) and HEAD is set to the current branch.
USAGE
  exit "${1:-2}"
}

CHECK=0 FORCE_GIT=0 POS=()
for a in "$@"; do
  case "$a" in
    --check) CHECK=1 ;;
    --force-git) FORCE_GIT=1 ;;
    -h|--help) usage 0 ;;
    *) POS+=("$a") ;;
  esac
done
[ "${#POS[@]}" -ge 2 ] || usage
HOST=${POS[0]} PORT=${POS[1]} DEST=${POS[2]:-/workspace/house_robot}
LOCAL_MODE=0
[ "$HOST" = local ] && LOCAL_MODE=1
[ "$LOCAL_MODE" = 1 ] && [ "${#POS[@]}" -ge 3 ] || [ "$LOCAL_MODE" = 0 ] || { echo "local mode needs a destination dir" >&2; exit 2; }

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

EXCLUDES=(
  --exclude=/.git --exclude=/private/ --exclude=/outputs/ --exclude='/.venv*/'
  --exclude=__pycache__/ --exclude='*.pyc' --exclude=.ruff_cache/ --exclude=.pytest_cache/
  --exclude=/AGENTS.md --exclude=/CLAUDE.md --exclude=/docs/knowledge --exclude=/.orca/
)
# Anything matching this in the dry-run list is a bug in the excludes -> abort.
FORBIDDEN='(^|/)private(/|$)|^outputs(/|$)|^\.venv|^AGENTS\.md$|^CLAUDE\.md$|^docs/knowledge|^\.orca(/|$)|\.heic$|\.jpe?g$'

if [ "$LOCAL_MODE" = 1 ]; then
  [ "$CHECK" = 1 ] || mkdir -p "$DEST"
  TARGET="$DEST/"; SSH_OPTS=(); RSYNC_PATH=()
else
  SSH_CMD="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
  SSH_OPTS=(-e "$SSH_CMD")
  RSYNC_PATH=(--rsync-path="mkdir -p '$DEST' && rsync")
  TARGET="$HOST:$DEST/"
fi
RSYNC=(rsync -a --no-owner --no-group --delete --itemize-changes "${SSH_OPTS[@]}" "${RSYNC_PATH[@]}" "${EXCLUDES[@]}")

if [ "$LOCAL_MODE" = 0 ]; then
  # Bare CUDA images lack rsync/git; rsync needs itself on both ends, even for --dry-run.
  $SSH_CMD "$HOST" 'command -v rsync >/dev/null && command -v git >/dev/null ||
    { apt-get update -qq && apt-get install -y -qq rsync git >/dev/null; }'
fi

echo "== dry run =="
# --dry-run does not run the remote mkdir, so list against a path that may not exist yet (rsync tolerates it).
LIST=$("${RSYNC[@]}" --dry-run ./ "$TARGET" | sed -E 's/^[^ ]+ //; s/ -> .*//' | grep -v '^$' || true)
echo "$LIST" | head -n 40
echo "($(echo "$LIST" | grep -c . || true) entries)"
if echo "$LIST" | grep -Eq "$FORBIDDEN"; then
  echo "ABORT: forbidden paths would be sent:" >&2
  echo "$LIST" | grep -E "$FORBIDDEN" >&2
  exit 1
fi
echo "check OK: no private/outputs/.venv*/symlink paths in the transfer list"
[ "$CHECK" = 1 ] && exit 0

echo "== transfer =="
"${RSYNC[@]}" ./ "$TARGET" | grep -v '^$' | tail -n 5

# --- .git ---
if [ -f .git ]; then GITCOMMON=$(cd "$(git rev-parse --git-common-dir)" && pwd); else GITCOMMON=$ROOT/.git; fi
BRANCH=$(git branch --show-current)
SHA=$(git rev-parse HEAD)

remote_sh() { if [ "$LOCAL_MODE" = 1 ]; then bash -c "$1"; else ssh -p "$PORT" "$HOST" "$1"; fi; }
HAVE_GIT=$(remote_sh "test -e '$DEST/.git' && echo yes || echo no")
if [ "$HAVE_GIT" = yes ] && [ "$FORCE_GIT" = 0 ]; then
  echo ".git already exists on the target: left untouched (use --force-git to replace it)"
else
  echo "== .git ($GITCOMMON -> target) =="
  GIT_TARGET="$TARGET.git/"
  [ "$LOCAL_MODE" = 1 ] || RSYNC_PATH=(--rsync-path="mkdir -p '$DEST/.git' && rsync")
  rsync -a --no-owner --no-group --delete "${SSH_OPTS[@]}" "${RSYNC_PATH[@]}" --exclude=/worktrees/ --exclude=/index "$GITCOMMON/" "$GIT_TARGET"
  if [ -n "$BRANCH" ]; then
    FIX="git -C '$DEST' symbolic-ref HEAD 'refs/heads/$BRANCH'"
  else
    FIX="git -C '$DEST' update-ref --no-deref HEAD '$SHA'"
  fi
  remote_sh "unset GIT_DIR GIT_WORK_TREE; $FIX && git -C '$DEST' reset -q && echo 'git HEAD: '\$(git -C '$DEST' rev-parse --short HEAD)"
fi
echo "done. next: ssh -p $PORT $HOST  ->  cd $DEST && bash scripts/remote/setup.sh"
