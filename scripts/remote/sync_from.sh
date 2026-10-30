#!/usr/bin/env bash
# Bring the remote outputs/ back to ./outputs/remote/. Run locally.
set -euo pipefail

usage() {
  cat <<USAGE
usage: $0 <user@host|local> <port> [remote_dir=/workspace/house_robot] [local_dest=<repo>/outputs/remote]

Copies <remote_dir>/outputs/ into local_dest (nothing is deleted locally).

Code changes made on the remote (commit there first) can be fetched with:
  git fetch ssh://<user@host>:<port>/<remote_dir> <branch>:refs/remotes/vast/<branch>
  e.g. git fetch ssh://root@ssh5.vast.ai:12345/workspace/house_robot main:refs/remotes/vast/main
  then: git log vast/main ; git merge vast/main   (or cherry-pick)
(Uncommitted remote edits: rsync them manually, or commit on the remote first. The remote git author may need
 git config user.name/user.email set there.)
USAGE
  exit "${1:-2}"
}
case "${1:-}" in -h|--help) usage 0 ;; esac
[ "$#" -ge 2 ] || usage
HOST=$1 PORT=$2 SRC=${3:-/workspace/house_robot}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DEST=${4:-$ROOT/outputs/remote}
mkdir -p "$DEST"
if [ "$HOST" = local ]; then
  rsync -av "$SRC/outputs/" "$DEST/"
else
  rsync -avz -e "ssh -p $PORT -o StrictHostKeyChecking=accept-new" "$HOST:$SRC/outputs/" "$DEST/"
fi
echo "-> $DEST/"
