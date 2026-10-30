"""Batch runner for the lift-only basket pinch trials of scripts/cloth_ipc_washer_test.py (IPC, friction only).

Runs `cloth_ipc_washer_test.py --stop-after lift --no-video` once per seed (one process per trial, each under the
GPU lock so other GPU jobs can interleave), collects the per-trial JSON and appends one summary row per condition
to outputs/cloth_ipc_lift/table.md.

Success (stage 1): after the lift >= 50 % of the cloth vertices are above the basket rim and the nearest vertex is
within 3 cm of the fingertip centre (`lift_success` in the trial JSON). IPC is not deterministic, so a seed only
fixes the drop pose, not the heap.

Usage
  .venv-ipc/bin/python scripts/cloth_ipc_lift_batch.py --tag base --seeds 0-4 -- --depth 0.04
  .venv-ipc/bin/python scripts/cloth_ipc_lift_batch.py --table          # print the summary table only
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "cloth_ipc_lift"
LOCK = "/tmp/house_robot-gpu.lock"
TABLE = OUT / "table.md"
HEADER = ("| tag | args | trials | success | mean lift dz (cloth mean) [m] | between fingers open / closed (mean) "
          "| above-rim frac (mean) | tip-nearest [m] (median) | errors | per-seed (s:ok,attempts) |\n"
          "|---|---|---|---|---|---|---|---|---|---|\n")


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else float("nan")


def median(v):
    v = sorted(x for x in v if x is not None)
    return v[len(v) // 2] if v else float("nan")


def summarize(tag, extra, results):
    n = len(results)
    ok = sum(bool(r.get("lift_success")) for r in results)
    ok1 = sum(bool(r.get("attempts_log", [{}])[0].get("lift_success")) for r in results)
    err = sum("error" in r and r["error"] is not None for r in results)
    per = " ".join(f"{r['seed']}:{'Y' if r.get('lift_success') else 'n'}"
                   + (f",{r['attempts']}" if "attempts" in r else "")
                   + (f",miss{sum(bool(e.get('miss')) for e in r['attempts_log'])}" if r.get("attempts_log") else "")
                   for r in results)
    return (f"| {tag} | `{' '.join(extra)}` | {n} | {ok}/{n} (1st attempt {ok1}/{n}) | {mean([r.get('lift_dzmean') for r in results]):.3f} "
            f"| {mean([r.get('verts_between_fingers_open') for r in results]):.1f} / "
            f"{mean([r.get('verts_between_fingers_closed') for r in results]):.1f} "
            f"| {mean([r.get('lifted_frac_above_rim') for r in results]):.2f} "
            f"| {median([r.get('lifted_tip_to_nearest_vertex') for r in results]):.3f} | {err} | {per} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="condition name (JSON prefix and table row)")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--table", action="store_true", help="only print the table")
    ap.add_argument("--timeout", type=float, default=900.0, help="per-trial timeout [s]")
    ap.add_argument("extra", nargs="*", help="arguments passed to cloth_ipc_washer_test.py (after --)")
    a = ap.parse_args()
    if a.table or a.tag is None:
        print(TABLE.read_text() if TABLE.exists() else "(no table yet)")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    # snapshot of the trial script, so editing it while a batch runs does not change the remaining trials
    snap = ROOT / "scripts" / f"_snap_{a.tag}.py"
    shutil.copy(ROOT / "scripts/cloth_ipc_washer_test.py", snap)
    try:
        results = run_trials(a, snap)
    finally:
        snap.unlink(missing_ok=True)
    row = summarize(a.tag, a.extra, results)
    if not TABLE.exists():
        TABLE.write_text(HEADER)
    with TABLE.open("a") as f:
        f.write(row)
    print(HEADER + row)


def run_trials(a, snap):
    results = []
    for s in parse_seeds(a.seeds):
        js = OUT / f"{a.tag}_s{s}.json"
        cmd = ["flock", LOCK, str(ROOT / ".venv-ipc/bin/python"), str(snap),
               "--seed", str(s), "--stop-after", "lift", "--no-video", "--json", str(js)] + a.extra
        t0 = time.time()
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=a.timeout)
        if p.returncode != 0 or not js.exists():
            print(p.stdout[-2000:], p.stderr[-3000:], file=sys.stderr)
            r = {"seed": s, "error": f"exit {p.returncode}"}
        else:
            r = json.loads(js.read_text())
        results.append(r)
        print(f"[{a.tag}] seed {s}: success={r.get('lift_success')} attempts={r.get('attempts', 1)} "
              f"above_rim={r.get('lifted_frac_above_rim')} tip_near={r.get('lifted_tip_to_nearest_vertex')} "
              f"between open/closed={r.get('verts_between_fingers_open')}/{r.get('verts_between_fingers_closed')} "
              f"touch_closed={r.get('touch_closed')} lost_at_dz={r.get('lift_lost_at_tip_dz')} "
              f"log={[(e.get('miss'), e.get('ipc_squeeze_N_closed'), e.get('lift_success')) for e in r.get('attempts_log', [])]} "
              f"squeeze_closed={r.get('ipc_squeeze_N_closed')} squeeze_lift={r.get('ipc_squeeze_N_lift_min_mean')} "
              f"lag_closed={r.get('ipc_lag_mm_closed')} "
              f"cyl={r.get('cyl_verts')} err={r.get('error')} wall={time.time() - t0:.0f}s", flush=True)
    return results


if __name__ == "__main__":
    main()
