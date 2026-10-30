"""Comparison table of the gripper / arm jitter around the first pinch, for several cloth_ipc_washer_diag.py runs.

One row per (run, segment) for the first attempt: close (gripper closing), hold (0.5 s after closing), lift (lift 1 +
lift 2 while the cloth is still held, i.e. up to `lift_lost_at_tip_dz` of the trial), all (close .. end of lift 2).

Jitter = residual after a Savitzky-Golay fit (cubic, ~0.2 s window) inside the segment. The scripted moves are cubic
smoothsteps in joint space, so the fit removes the commanded motion; the centred moving average used by
cloth_ipc_diag_report.py leaves ~a*T^2/24 (about 1 mm for the 2 s lift) of the commanded acceleration in its residual.
  l6     link6 position jitter (arm): RMS / peak-to-peak [mm], dominant frequency [Hz] (>= 3 Hz)
  q7     finger joint jitter (gripper relative to link6): peak-to-peak [mm] of joint7, sign-flip rate of its velocity
         (1.0 = flips every step = Nyquist chatter)
  sq     IPC squeeze [N] mean / std; cf7 coupling force on link7 [N] max; newton = Newton iterations per step max
  arm    max joint-space tracking error of joints 1-6 [mrad] (cmd - q)

  .venv-ipc/bin/python scripts/cloth_ipc_jitter_table.py outputs/ipc_diag/a_base outputs/ipc_diag/a_gkp2000 [--md]
"""
import argparse
import json

import numpy as np
from scipy.signal import savgol_filter


def sg_res(x, win):
    """x - cubic Savitzky-Golay fit along axis 0 (window shrunk to the segment length)."""
    n = len(x)
    w = min(win, n if n % 2 else n - 1)
    if w < 5:
        return np.zeros_like(x)
    return x - savgol_filter(x, w, 3, axis=0, mode="interp")


def dom_freq(r, dt, fmin=3.0):
    if len(r) < 32:
        return float("nan")
    r = r - r.mean(0)
    f = np.fft.rfftfreq(len(r), dt)
    p = (np.abs(np.fft.rfft(r * np.hanning(len(r))[:, None], axis=0)) ** 2).sum(1)
    p[f < fmin] = 0
    return float(f[np.argmax(p)])


def stats(d, a, b, dt, win):
    s = slice(a, b)
    lp = d["lpos"][s].reshape(-1, 3, 3)
    r6 = sg_res(lp[:, 0], win)
    r7 = sg_res(lp[:, 1], win)
    m6 = np.linalg.norm(r6, axis=1)
    q = d["q"][s]
    rq7 = sg_res(q[:, 10:12], win)
    dq = d["dq"][s]
    sgn = np.sign(dq[:, 10])
    flip = float((sgn[1:] * sgn[:-1] < 0).mean()) if len(sgn) > 1 else float("nan")
    sq = d["squeeze"][s, 0]
    cf = d["cforce"][s].reshape(-1, 4, 6)
    cf7 = np.linalg.norm(cf[:, 2, :3], axis=1)
    out = dict(
        n=b - a,
        l6_rms=float(np.sqrt((m6 ** 2).mean()) * 1e3), l6_p2p=float((r6.max(0) - r6.min(0)).max() * 1e3),
        l6_f=dom_freq(r6, dt), l7_p2p=float((r7.max(0) - r7.min(0)).max() * 1e3), l7_f=dom_freq(r7, dt),
        q7_p2p=float(np.ptp(rq7, axis=0).max() * 1e3), q7_f=dom_freq(rq7, dt), q7_flip=flip,
        sq_mean=float(np.nanmean(sq)), sq_std=float(np.nanstd(sq)), sq_min=float(np.nanmin(sq)),
        cf7_max=float(cf7.max()), cf7_hp=float(np.linalg.norm(sg_res(cf[:, 2, :3], win).std(0))),
        arm_err=float(np.abs(d["cmd"][s, 4:10] - q[:, 4:10]).max() * 1e3),
        newton=int(d["newton"][s].max()) if "newton" in d else -1,
    )
    return out


def segments(m, d, by_line=False):
    """First-attempt segment bounds from the recorded segment starts (cloth_ipc_washer_diag.py `seg`)."""
    seg = m["seg"]
    n = len(d["q"])
    starts = [s[0] for s in seg] + [n]

    # the recorded source text comes from the file on disk at the end of the run; runs of the 932e97b version whose
    # file was edited meanwhile are matched by their line numbers instead
    old_lines = {"gripper(CLOSED": 547, "a.lift1_s": 587, "a.lift2_s": 591}

    def first(pat, after=0):
        for i, s in enumerate(seg):
            if s[0] >= after and s[1] == "attempt" and (s[2] == old_lines[pat] if by_line else pat in s[3]):
                return i
        return None
    ic = first("gripper(CLOSED")
    if ic is None:
        return {}
    il1 = first("a.lift1_s", starts[ic])
    il2 = first("a.lift2_s", starts[ic])
    nxt = ic + 1
    while nxt < len(seg) and seg[nxt][1] != "attempt":   # skip the --switch-test windows
        nxt += 1
    he = next(x for x in starts[nxt + 1:] if x > starts[nxt])   # --switch-test: the first window
    out = {"close": (starts[ic], starts[ic + 1]), "hold": (starts[nxt], he)}
    if il1 is not None and il2 is not None:
        a, b = starts[il1], starts[il2 + 1]
        lost = m["res"].get("lift_lost_at_tip_dz")
        log = m["res"].get("attempts_log") or [{}]
        lost = log[0].get("lift_lost_at_tip_dz", lost)
        e = b
        if lost is not None:
            dz = d["tcp"][a:b, 2] - d["tcp"][a, 2]
            k = np.where(dz >= lost)[0]
            e = a + int(k[0]) if len(k) else b
        out["lift"] = (a, max(e, a + 5))
        out["all"] = (starts[ic], b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bases", nargs="+", help="paths without extension")
    ap.add_argument("--win-s", type=float, default=0.2, help="Savitzky-Golay window [s]")
    ap.add_argument("--segs", default="close,hold,lift")
    ap.add_argument("--md", action="store_true", help="markdown table")
    ap.add_argument("--by-line", default="", help="comma-separated tags whose segments are matched by the 932e97b "
                    "line numbers (runs whose test script was edited on disk while they ran)")
    ap.add_argument("--windows", action="store_true", help="rows = the --switch-test windows instead of segments")
    ap.add_argument("--skip", type=int, default=10, help="--windows: steps skipped after each switch")
    a = ap.parse_args()
    cols = ["cond", "seg", "steps", "l6 rms/p2p mm @Hz", "l7 p2p", "q7 p2p mm @Hz flip", "squeeze mean/std/min N",
            "cf7 max N", "cf7 hp", "arm err mrad", "newton max", "lift ok", "touch closed", "verts open/closed",
            "lost dz m"]
    rows = []
    for base in a.bases:
        d = dict(np.load(base + ".npz"))
        m = json.load(open(base + ".json"))
        dt = m["diag"]["dt"]
        win = int(round(a.win_s / dt)) | 1
        log = (m["res"].get("attempts_log") or [{}])[0]
        lost = log.get("lift_lost_at_tip_dz")
        segs = [(n, se) for n, se in segments(m, d, m["tag"] in a.by_line.split(",")).items() if n in a.segs.split(",")]
        if a.windows and m.get("windows"):   # --switch-test windows, skipping the switching transient
            segs = [("win:" + n, (st + a.skip, en)) for n, st, en in m["windows"]]
        for name, (s, e) in segs:
            r = stats(d, s, e, dt, win)
            rows.append([m["tag"], name, f"{s}-{e}", f"{r['l6_rms']:.2f}/{r['l6_p2p']:.2f} @{r['l6_f']:.0f}",
                         f"{r['l7_p2p']:.2f}", f"{r['q7_p2p']:.2f} @{r['q7_f']:.0f} {r['q7_flip']:.2f}",
                         f"{r['sq_mean']:.1f}/{r['sq_std']:.1f}/{r['sq_min']:.1f}", f"{r['cf7_max']:.0f}",
                         f"{r['cf7_hp']:.1f}", f"{r['arm_err']:.0f}", str(r["newton"]),
                         str(log.get("lift_success")), str(log.get("touch_closed")),
                         f"{log.get('verts_between_fingers_open')}/{log.get('verts_between_fingers_closed')}",
                         str(lost)])
    if a.md:
        print("| " + " | ".join(cols) + " |")
        print("|" + "---|" * len(cols))
        for r in rows:
            print("| " + " | ".join(r) + " |")
    else:
        w = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)] if rows else [len(c) for c in cols]
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
        for r in rows:
            print("  ".join(v.ljust(w[i]) for i, v in enumerate(r)))


if __name__ == "__main__":
    main()
