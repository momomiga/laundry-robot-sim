"""Summarise outputs/ipc_diag/<tag>.npz/.json written by cloth_ipc_washer_diag.py.

Per segment (one move / hold / gripper call of the test script): high-frequency jitter of link6 and the fingers
(residual after a centred 0.2 s moving average: RMS / peak-to-peak [mm], dominant frequency [Hz]), sign-flip rate of
the finger joint velocity (1.0 = flips every step = Nyquist chatter), finger joint torque spread, IPC lag, coupling
force spread, Newton iterations. Then the arm-vs-obstacle penetration table over the whole run.

  .venv-ipc/bin/python scripts/cloth_ipc_diag_report.py outputs/ipc_diag/full_base [--seg-filter attempt]
"""
import argparse
import json

import numpy as np

LINKS = [f"link{i}" for i in range(1, 9)]
GROUPS = ["washer", "mast", "chassis", "basket"]
COUP = ["base_link", "link6", "link7", "link8"]


def hp(x, w):
    """x - centred moving average (edges: shrinking window)."""
    if len(x) < 3:
        return x * 0
    k = np.ones(w) / w
    pad = np.pad(x, ((w // 2, w - 1 - w // 2),) + ((0, 0),) * (x.ndim - 1), mode="edge")
    if x.ndim == 1:
        return x - np.convolve(pad, k, "valid")
    return x - np.stack([np.convolve(pad[:, j], k, "valid") for j in range(x.shape[1])], 1)


def dom_freq(v, dt):
    if len(v) < 32:
        return float("nan")
    v = v - v.mean()
    f = np.fft.rfftfreq(len(v), dt)
    p = np.abs(np.fft.rfft(v * np.hanning(len(v)))) ** 2
    p[f < 2.0] = 0
    return float(f[np.argmax(p)])


def seg_stats(d, a, b, dt, w):
    s = slice(a, b)
    lp = d["lpos"][s].reshape(-1, 3, 3)          # link6, link7, link8
    lv = d["lvel"][s].reshape(-1, 3, 3)
    out = {}
    for j, n in enumerate(["l6", "l7", "l8"]):
        r = hp(lp[:, j], w)
        mag = np.linalg.norm(r, axis=1)
        out[f"{n}_rms_mm"] = float(np.sqrt((mag ** 2).mean()) * 1e3)
        out[f"{n}_p2p_mm"] = float((r.max(0) - r.min(0)).max() * 1e3)
        vr = hp(lv[:, j], w)
        ax = np.argmax(vr.std(0))
        out[f"{n}_f_hz"] = dom_freq(vr[:, ax], dt)
        out[f"{n}_vstd"] = float(vr.std(0).max())
    dq = d["dq"][s]
    q = d["q"][s]
    for jn, k in (("j7", 10), ("j8", 11)):
        sgn = np.sign(dq[:, k])
        out[f"{jn}_flip"] = float((sgn[1:] * sgn[:-1] < 0).mean()) if len(sgn) > 1 else float("nan")
        out[f"{jn}_q_hp_p2p_mm"] = float(np.ptp(hp(q[:, k], w)) * 1e3)
    for k in range(4, 10):   # arm joints
        out[f"j{k - 3}_q_hp_p2p_mrad"] = float(np.ptp(hp(q[:, k], w)) * 1e3)
        sgn = np.sign(dq[:, k])
        out[f"j{k - 3}_flip"] = float((sgn[1:] * sgn[:-1] < 0).mean()) if len(sgn) > 1 else float("nan")
    tau = d["tau"][s]
    out["tau7_mean"], out["tau7_std"], out["tau7_absmax"] = float(tau[:, 10].mean()), float(tau[:, 10].std()), \
        float(np.abs(tau[:, 10]).max())
    out["tau_arm_hp_std"] = [round(float(v), 2) for v in hp(tau[:, 4:10], w).std(0)]
    lag = d["lag"][s].reshape(-1, 4, 4)
    for j, n in enumerate(COUP):
        m = np.linalg.norm(lag[:, j, :3], axis=1)
        out[f"lag_{n}_mm_mean_max"] = [round(float(m.mean()), 2), round(float(m.max()), 2)]
    cf = d["cforce"][s].reshape(-1, 4, 6)
    for j, n in enumerate(COUP):
        f = np.linalg.norm(cf[:, j, :3], axis=1)
        out[f"cf_{n}_N_mean_std_max"] = [round(float(f.mean()), 2), round(float(f.std()), 2), round(float(f.max()), 2)]
        out[f"cf_{n}_hp_std"] = round(float(np.linalg.norm(hp(cf[:, j, :3], w).std(0))), 2)
    sq = d["squeeze"][s]
    out["squeeze_mean_std"] = [round(float(np.nanmean(sq[:, 0])), 2), round(float(np.nanstd(sq[:, 0])), 2)]
    if "newton" in d:
        nw = d["newton"][s].ravel()
        out["newton_mean_max"] = [round(float(nw.mean()), 1), int(nw.max())]
        out["newton_at64_frac"] = round(float((nw >= 64).mean()), 3)
    gc = d["gcontact"][s]
    out["gcontact_steps"] = int((gc[:, 0] > 0).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="path without extension")
    ap.add_argument("--w", type=int, default=20, help="moving-average window [steps]")
    ap.add_argument("--keys", default="l6_rms_mm,l6_p2p_mm,l6_f_hz,l7_rms_mm,l7_p2p_mm,l7_f_hz,j7_flip,"
                    "j7_q_hp_p2p_mm,tau7_std,lag_link7_mm_mean_max,cf_link7_N_mean_std_max,cf_link7_hp_std,"
                    "newton_mean_max,squeeze_mean_std")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--no-collision", action="store_true")
    a = ap.parse_args()
    d = dict(np.load(a.base + ".npz"))
    m = json.load(open(a.base + ".json"))
    dt = m["diag"]["dt"]
    seg = m["seg"]
    n = len(d["q"])
    starts = [s[0] for s in seg] + [n]
    print(f"# {m['tag']}  steps {n}  dt {dt}  wall {m['wall_s']} s  gains arm kp {m['gains']['kp'][4]} kv "
          f"{m['gains']['kv'][4]:.1f} grip kp {m['gains']['kp'][10]} kv {m['gains']['kv'][10]:.1f}")
    keys = a.keys.split(",")
    rows = []
    for i, s in enumerate(seg):
        st, en = starts[i], starts[i + 1]
        if en - st < 5:
            continue
        r = seg_stats(d, st, en, dt, a.w)
        rows.append((st, en, s, r))
        if a.full:
            print(f"[{st}-{en}) {s[1]}:{s[2]} {s[3]}")
            print("   ", json.dumps(r))
        else:
            print(f"[{st:5d}-{en:5d}) t={st * dt:6.2f} {s[1][:8]}:{s[2]} {s[3][:60]}")
            print("     " + "  ".join(f"{k}={r[k] if not isinstance(r[k], float) else round(r[k], 3)}" for k in keys
                                         if k in r))
    if m.get("windows"):
        print("\n# switch test windows (same pinch): flip = sign-flip rate of the finger joint velocity")
        print("window | j7_flip j8_flip | tau7 std / sat% | tau8 std / sat% | squeeze mean std min frac<5N | "
              "l6 hp p2p mm | l7 hp p2p mm | cf7 hp std N | q7 q8 mm | newton")
        for name, st, en in m["windows"]:
            st2 = st + 10   # skip the switching transient
            r = seg_stats(d, st2, en, dt, a.w)
            tau = d["tau"][st2:en]
            fr = m["res"]  # noqa: F841
            sat = [float((np.abs(tau[:, k]) >= 0.995 * np.abs(tau[:, k]).max()).mean()) if np.abs(tau[:, k]).max() > 0
                   else 0.0 for k in (10, 11)]
            sq = d["squeeze"][st2:en, 0]
            print(f"{name:20s} | {r['j7_flip']:.2f} {r['j8_flip']:.2f} | {tau[:, 10].std():5.1f} / {sat[0]:.2f} | "
                  f"{tau[:, 11].std():5.1f} / {sat[1]:.2f} | {sq.mean():5.1f} {sq.std():5.1f} {sq.min():6.1f} "
                  f"{(sq < 5).mean():.2f} | {r['l6_p2p_mm']:5.2f} | {r['l7_p2p_mm']:5.2f} | {r['cf_link7_hp_std']:5.1f} | "
                  f"{d['q'][st2:en, 10].mean() * 1e3:4.1f} {d['q'][st2:en, 11].mean() * 1e3:5.1f} | "
                  f"{r.get('newton_mean_max')}")
    res = m["res"]
    print("attempts_log:")
    for x in res.get("attempts_log") or []:
        print("  ", {k: x[k] for k in ("miss", "ipc_squeeze_N_closed", "verts_between_fingers_open",
                                       "verts_between_fingers_closed", "touch_closed", "lift_success")})
    print("success", res.get("success"), "error", res.get("error"))
    if a.no_collision:
        return
    dep = d["depth"].reshape(n, 8, 4)
    print("\n# arm vs obstacle: max penetration [mm] (negative = min clearance), time of max, steps with depth > 1 mm "
          "(first-last time [s])")
    for li, ln in enumerate(LINKS):
        cells = []
        for gi, g in enumerate(GROUPS):
            x = dep[:, li, gi]
            k = int(np.argmax(x))
            pen = np.where(x > 0.001)[0]
            seg_k = max(i for i, s in enumerate(starts[:-1]) if s <= k) if starts[0] <= k else -1
            lbl = seg[seg_k][3][:28] if seg_k >= 0 else "-"
            if len(pen):
                cells.append(f"{g}: {x[k] * 1e3:6.1f} @t{k * dt:5.2f} [{lbl}] n={len(pen)} "
                             f"{pen[0] * dt:.2f}-{pen[-1] * dt:.2f}s")
            else:
                cells.append(f"{g}: {x[k] * 1e3:6.1f}")
        print(f"{ln}: " + " | ".join(cells))
    if "gc_pairs_seen" in d:
        ln = m["link_names"]
        seen = {}
        for row in d["gc_pairs_seen"]:
            for a_, b_ in zip(row[1::2], row[2::2]):
                if a_ < 0:
                    continue
                k = f"{ln[str(int(a_))]}-{ln[str(int(b_))]}"
                seen[k] = seen.get(k, 0) + 1
        print("genesis arm contacts seen (first 200 steps with any):", seen)
    gc = d["gcontact"]
    print("genesis arm-contact steps:", int((gc[:, 0] > 0).sum()), "max pen [mm]", float(gc[:, 1].max() * 1e3))


if __name__ == "__main__":
    main()
