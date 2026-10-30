# Running on a vast.ai GPU instance

Code goes over with rsync (not GitHub), visuals come back as PNG/MP4 files (headless rendering, no RDP).

## Choosing an instance
- On-demand (not interruptible).
- GPU VRAM >= 16 GB (Genesis ~1-2 GB, IPC ~2.3 GB, VLM qwen3-vl:8b ~7.4 GB). Blackwell needs CUDA 12.8+ drivers.
- Disk >= 60 GB (.venv + .venv-ipc with torch, ollama model ~6 GB).
- Docker options: `-e NVIDIA_DRIVER_CAPABILITIES=all` (required for the NVIDIA EGL; without it rendering
  falls back to OSMesa on the CPU and is very slow). It cannot be added later; recreate the instance.
- Image example: `nvidia/cuda:12.8.1-devel-ubuntu22.04` or a vast.ai `pytorch`/cuda 12.8+ base image; SSH launch mode, root.

## Flow
```sh
# local
scripts/remote/sync_to.sh --check root@HOST PORT      # dry run + forbidden-path check
scripts/remote/sync_to.sh root@HOST PORT              # -> /workspace/house_robot
ssh -p PORT root@HOST
# remote
cd /workspace/house_robot && bash scripts/remote/setup.sh   # idempotent; --skip-system --skip-ollama --skip-ipc
flock /tmp/house_robot-gpu.lock uv run python scripts/demo_video.py   # work; GPU jobs through the lock
# local
scripts/remote/sync_from.sh root@HOST PORT            # remote outputs/ -> outputs/remote/
```
Re-running `sync_to.sh` updates files; the remote `.git` is only copied the first time (`--force-git` replaces it).
It uses `rsync --delete`: the local tree is the source of truth, so files created or edited only on the remote
(outside `outputs/` and the excluded paths) are overwritten or deleted by the next `sync_to.sh`. Edit code locally;
if you must edit on the remote, commit and fetch it back (below) before syncing again.
Never sent: `private/`, `outputs/`, `.venv*/`, AGENTS.md/CLAUDE.md/docs/knowledge. The dry-run check aborts if any appear.

Code changed on the remote: commit there (set `git config user.name/user.email`), then locally
`git fetch ssh://root@HOST:PORT/workspace/house_robot BRANCH:refs/remotes/vast/BRANCH`.
Do not push the remote to GitHub: the git author contains personal info.

Smoke only: `bash scripts/remote/smoke.sh [--skip-ipc] [--skip-vlm]`. ollama log: `/tmp/ollama.log`
(restart after an instance restart: `nohup ollama serve > /tmp/ollama.log 2>&1 &`).

## Unattended runs
Queue scripts (`/workspace/jobs/q_*.sh`, started with nohup, log to `q_*.log`, GPU jobs under the lock) end by
printing `QUEUE_DONE` (or `QUEUE_ABORT <reason>`). Two layers of defense stop the instance afterwards:
```sh
# remote safety net, launched when a queue is left unattended (an idle instance is stopped after 90 min,
# also while you work interactively without the lock: touch KEEPALIVE); --max-hours H = hard cap
nohup bash scripts/remote/idle_stop.sh > /workspace/jobs/idle_stop.log 2>&1 &
bash scripts/remote/idle_stop.sh --status      # "idle" / "busy: <reasons>" (lock, GPU apps, queue process, KEEPALIVE)
touch /workspace/jobs/KEEPALIVE                # keeps it busy (rm to release); --max-hours still wins

# local watcher (first line of defense): wait for the queue(s), collect, verify, then stop
scripts/remote/collect_and_stop.sh --auth-check root@HOST PORT            # read-only credential check first
nohup setsid scripts/remote/collect_and_stop.sh --stop root@HOST PORT /workspace/jobs/q_x.log \
  > outputs/remote/collect_NAME.log 2>&1 &
```
Order: the local watcher normally collects and stops the instance; if the local machine is off, the remote idle
stop (90 min by default) is the fallback, so results stay on the instance's disk (`/workspace` survives stop/start)
until the next `sync_from.sh`. Nothing is stopped if verification (file sizes vs. remote) fails. Exit codes:
1 verify failed, 3 unreachable (instance may already be stopped), 4 auth/stop failed,
5 busy again right before the stop (not stopped).
The direct ssh port can change after a restart; pass the new PORT. `local` as HOST runs everything locally (testing;
`--stop` is refused). The instance's API key stays in the remote environment and is never printed.

## Costs and pitfalls
- A stopped instance still bills for storage (disk is kept).
- After stopping, resuming may fail or wait because the GPU may be taken by someone else. Copy results out
  (`sync_from.sh`) before stopping, and destroy the instance when finished.
