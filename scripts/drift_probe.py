"""Drift probe: why does the IPC towel slide to the rear basket wall while the base drives at constant velocity?

Same scene / IPC options as scripts/cloth_ipc_washer_test.py (no house, no VLM, no planner subprocess). The towel is
dropped and settled exactly as in the integrated demo, then the base drives a short straight line in +x (smoothstep
ramp, as scripts/integrated_demo.py) ending at the parked origin, then holds. Every step is logged:

  Genesis base_link pose, the IPC-side base_link transform (coupler._abd_data_by_link), the aim transform used for that
  IPC step (coupler._abd_transforms_by_link), the coupling force / torque returned to Genesis, the libuipc Newton
  iteration count of the frame (uipc Timer report), the towel centroid, p90 height, max edge stretch and the slip
  velocity of the towel vertices lying on the basket floor relative to the IPC basket.

Arms (all other flags are the cloth_ipc_washer_test.py ones, e.g. --newton-iters, --newton-tol, --eps-v, --cs-trans):
  --base-mode pd         (default) base x / y PD-driven as in the demo (kp 8000, kv 2 sqrt(kp))
  --base-mode kinematic  base x / y written every step (set_dofs_position + set_dofs_velocity, base kp = kv = 0):
                         removes the Genesis PD / explicit-coupling loop from the basket motion
  --semi-implicit off    (cloth_ipc_washer_test.py flag) libuipc Newton semi-implicit termination off

Run from the repo root (it imports scripts/cloth_ipc_washer_test.py from the current directory):
  flock /tmp/house_robot-gpu.lock .venv-ipc/bin/python /path/to/drift_probe.py --tag base <flags>
Result (GPU, seed 0, 1.2 m at 0.3 m/s): the default semi-implicit termination lets the towel slide about 95 mm
back within 1 s; --semi-implicit off: 0.1 mm (moving steps about 4.6x slower); --base-mode kinematic: same as pd.
Re-analyse a log on any machine (numpy + scipy only):
  python3 drift_probe.py --analyze outputs/drift_probe_base.npz
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

T_START = time.time()
STANDBY = np.array([0.0, 0.0, 0.0, 0.5, -0.0055, 0.2178, 0.0, -0.002, 0.0091, -0.0612, 0.0, -0.0])  # planner standby
BASKET_CENTER_XY = (0.04, -0.05)   # tools/params.py
COLS = ["t", "phase", "cmd_x", "cmd_y", "g_x", "g_y", "g_z", "g_yaw", "ipc_x", "ipc_y", "ipc_z", "aim_x", "aim_y",
        "aim_z", "rot_ipc_aim_x", "rot_ipc_aim_y", "rot_ipc_aim_z", "F_x", "F_y", "F_z", "Tq_x", "Tq_y", "Tq_z",
        "newton", "frame_ms", "c_x", "c_y", "c_z", "cx_rel", "cy_rel", "p90", "stretch", "slip_x", "slip_y",
        "n_floor"]
PHASES = {0: "settle", 1: "pre", 2: "ramp", 3: "cruise", 4: "stop", 5: "after"}


# ---------------------------------------------------------------- analysis (CPU, numpy / scipy only)
def _rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    ang = math.acos(c)
    if ang < 1e-12:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(ang))
    return ang * w


def newton_per_frame(raw):
    """The uipc Timer count may be per frame or cumulative: difference it if it never decreases and grows."""
    raw = np.asarray(raw, float)
    ok = np.isfinite(raw)
    if ok.sum() > 20:
        r = raw[ok]
        if np.all(np.diff(r) >= 0) and r[-1] > 3 * max(r[0], 1):
            out = np.full_like(raw, np.nan)
            out[ok] = np.r_[np.nan, np.diff(r)]
            return out, "cumulative->differenced"
    return raw, "per-frame"


def analyze(L, dt, cap, meta=None, out=print):
    """L: (n, len(COLS)) array. Returns a summary dict and prints a table."""
    C = {c: L[:, i] for i, c in enumerate(COLS)}
    ph = C["phase"].astype(int)
    nw, nmode = newton_per_frame(C["newton"])
    S = {"newton_mode": nmode, "newton_cap": cap}
    # jitter: deviation from a Savitzky-Golay smooth (window 0.21 s, cubic), its second difference = jitter acceleration
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        savgol_filter = None
    drive = np.isin(ph, [1, 2, 3, 4, 5])
    idx = np.where(drive)[0]
    acc = {}
    for src in ("ipc", "g"):
        for ax in ("x", "y", "z"):
            p = C[f"{src}_{ax}"][idx]
            if savgol_filter is not None and len(p) > 25:
                jit = p - savgol_filter(p, 21, 3)
            else:
                jit = p - np.convolve(p, np.ones(21) / 21, mode="same")
            a = np.full(len(p), np.nan)
            a[1:-1] = (jit[2:] - 2 * jit[1:-1] + jit[:-2]) / dt ** 2
            atot = np.full(len(p), np.nan)
            atot[1:-1] = (p[2:] - 2 * p[1:-1] + p[:-2]) / dt ** 2
            full_j = np.full(len(L), np.nan)
            full_a = np.full(len(L), np.nan)
            full_t = np.full(len(L), np.nan)
            full_j[idx], full_a[idx], full_t[idx] = jit, a, atot
            acc[(src, ax)] = (full_j, full_a, full_t)
    cmd_a = np.full(len(L), np.nan)
    cx = C["cmd_x"]
    cmd_a[1:-1] = (cx[2:] - 2 * cx[1:-1] + cx[:-2]) / dt ** 2
    rows = []
    for k in [0, 1, 2, 3, 4, 5]:
        m = ph == k
        if m.sum() < 3:
            continue
        r = {"phase": PHASES[k], "steps": int(m.sum()), "t0": round(float(C["t"][m][0]), 2),
             "t1": round(float(C["t"][m][-1]), 2)}
        n = nw[m]
        n = n[np.isfinite(n)]
        if len(n):
            r.update(newton_mean=round(float(n.mean()), 1), newton_max=int(n.max()),
                     newton_at_cap=round(float((n >= cap).mean()), 2) if cap else None)
        r["frame_ms"] = round(float(np.nanmean(C["frame_ms"][m])), 1)
        if k > 0:
            for ax in ("x", "y", "z"):
                d = (C[f"ipc_{ax}"] - C[f"aim_{ax}"])[m] * 1e6
                r[f"ipc-aim_{ax}_um"] = [round(float(d.mean()), 1), round(float(d.std()), 1)]
            for ax in ("x", "y"):
                f = C[f"F_{ax}"][m]
                r[f"F_{ax}_N"] = [round(float(f.mean()), 1), round(float(f.std()), 1)]
            r["g_y_std_mm"] = round(float(np.std(C["g_y"][m]) * 1e3), 2)
            r["track_err_x_mm"] = [round(float(np.mean(C["cmd_x"][m] - C["g_x"][m]) * 1e3), 2),
                                   round(float(np.std(C["cmd_x"][m] - C["g_x"][m]) * 1e3), 2)]
            for src in ("ipc", "g"):
                for ax in ("x", "y", "z"):
                    a = acc[(src, ax)][1][m]
                    a = a[np.isfinite(a)]
                    if len(a):
                        r[f"jit_acc_{src}_{ax}"] = [round(float(np.sqrt(np.mean(a ** 2))), 2),
                                                    round(float(np.abs(a).max()), 2)]
            at = acc[("ipc", "x")][2][m]
            at = at[np.isfinite(at)]
            if len(at):
                r["ipc_acc_x_total_max"] = round(float(np.abs(at).max()), 2)
            rv = np.stack([C["rot_ipc_aim_x"], C["rot_ipc_aim_y"], C["rot_ipc_aim_z"]], 1)[m]
            r["rot_ipc_aim_mrad_max"] = round(float(np.linalg.norm(rv, axis=1).max() * 1e3), 3)
        r["cx_rel_mm"] = [round(float(C["cx_rel"][m][0] * 1e3), 1), round(float(C["cx_rel"][m][-1] * 1e3), 1)]
        r["p90_mm"] = [round(float(C["p90"][m][0] * 1e3), 1), round(float(C["p90"][m][-1] * 1e3), 1)]
        r["stretch"] = [round(float(np.mean(C["stretch"][m])), 3), round(float(np.max(C["stretch"][m])), 3)]
        sl = C["slip_x"][m]
        sl = sl[np.isfinite(sl)]
        if len(sl):
            r["floor_slip_x_mm_s"] = [round(float(sl.mean() * 1e3), 1), round(float(np.percentile(sl, 5) * 1e3), 1)]
        # towel centroid world velocity (x) vs base velocity
        cxw = C["c_x"][m]
        if len(cxw) > 2:
            r["towel_vx_world_mean"] = round(float((cxw[-1] - cxw[0]) / (C["t"][m][-1] - C["t"][m][0])), 3)
            gx = C["ipc_x"][m]
            r["basket_vx_mean"] = round(float((gx[-1] - gx[0]) / (C["t"][m][-1] - C["t"][m][0])), 3)
        rows.append(r)
    S["phases"] = rows
    # spectrum of the IPC basket jitter (x, y, z) during the cruise
    m = ph == 3
    if m.sum() > 64:
        spec = {}
        for ax in ("x", "y", "z"):
            j = acc[("ipc", ax)][0][m]
            j = j[np.isfinite(j)]
            F = np.abs(np.fft.rfft(j - j.mean())) ** 2
            f = np.fft.rfftfreq(len(j), dt)
            if F[1:].sum() > 0:
                spec[ax] = dict(peak_hz=round(float(f[1:][np.argmax(F[1:])]), 1),
                                frac_above_25hz=round(float(F[f > 25].sum() / F[1:].sum()), 2),
                                rms_um=round(float(j.std() * 1e6), 2))
        S["ipc_jitter_spectrum_cruise"] = spec
    # drift milestones from the drive start
    d0 = np.where(np.isin(ph, [2, 3, 4]))[0]
    if len(d0):
        t0 = C["t"][d0[0]]
        cx0 = C["cx_rel"][d0[0] - 1] if d0[0] > 0 else C["cx_rel"][d0[0]]
        mil = {}
        for dtm in (0.5, 1.0, 1.75, 2.5, 3.5):
            i = np.searchsorted(C["t"], t0 + dtm)
            if i < len(L) and ph[i] in (2, 3, 4, 5):
                mil[f"+{dtm}s"] = round(float((C["cx_rel"][i] - cx0) * 1e3), 1)
        S["drift_dcx_mm"] = mil
        S["drive_t0"] = round(float(t0), 2)
    if meta:
        S["meta"] = meta
    out(json.dumps(S, indent=1))
    hint(S, out)
    return S


def hint(S, out=print):
    """Automatic reading against the interpretation table (thresholds explained in the probe notes)."""
    ph = {r["phase"]: r for r in S.get("phases", [])}
    cr, pre = ph.get("cruise"), ph.get("pre")
    lines = []
    if cr:
        cap = S.get("newton_cap")
        if cr.get("newton_at_cap") is not None:
            lines.append(f"Newton at cap during cruise: {cr['newton_at_cap']:.0%} of frames (pre/parked: "
                         f"{pre.get('newton_at_cap') if pre else 'n/a'}) mean {cr.get('newton_mean')} vs parked "
                         f"{pre.get('newton_mean') if pre else 'n/a'} (cap {cap})")
        ja = cr.get("jit_acc_ipc_x", [0, 0]), cr.get("jit_acc_ipc_z", [0, 0])
        lines.append(f"IPC basket jitter acc cruise: x rms/max {ja[0]}, z rms/max {ja[1]} m/s2 (slip needs ~11.8)")
        lines.append(f"towel floor slip x cruise: mean/p5 {cr.get('floor_slip_x_mm_s')} mm/s; towel vs basket world "
                     f"vx {cr.get('towel_vx_world_mean')} / {cr.get('basket_vx_mean')} m/s")
        lines.append(f"ipc-aim x um mean/std {cr.get('ipc-aim_x_um')}, F_x N {cr.get('F_x_N')}, g_y std "
                     f"{cr.get('g_y_std_mm')} mm")
    lines.append(f"drift dcx mm: {S.get('drift_dcx_mm')}")
    out("---- reading ----\n" + "\n".join(lines))


# ---------------------------------------------------------------- GPU probe
def run_probe():
    sys.path.insert(0, str(Path.cwd() / "tools"))
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    if not (Path.cwd() / "scripts" / "cloth_ipc_washer_test.py").exists():
        sys.exit("run from the repo root (scripts/cloth_ipc_washer_test.py not found)")
    import genesis as gs
    import uipc
    import cloth_ipc_washer_test as cwt
    import params as P

    def make_parser():
        ap = cwt.make_parser()
        ap.add_argument("--tag", default="base")
        ap.add_argument("--drive-len", type=float, default=1.2, help="straight drive in +x ending at the origin [m]")
        ap.add_argument("--speed", type=float, default=0.30)
        ap.add_argument("--ramp", type=float, default=0.6)
        ap.add_argument("--base-mode", choices=["pd", "kinematic"], default="pd")
        ap.add_argument("--pre-s", type=float, default=0.5, help="parked, logged hold before the drive [s]")
        ap.add_argument("--after-s", type=float, default=2.0, help="logged hold after the stop [s]")
        ap.add_argument("--no-standby", dest="standby", action="store_false",
                        help="keep the q=0 arm (default: the planner standby of the integrated demo, no planner)")
        ap.add_argument("--max-wall", type=float, default=0.0, help="> 0: stop the logged phases after this many "
                        "wall seconds since start (results are still written)")
        ap.add_argument("--uipc-warn", action="store_true", help="libuipc logger at Warn (prints Newton cap exits)")
        ap.add_argument("--log", default=None, help="npz path (default outputs/drift_probe_<tag>.npz)")
        ap.set_defaults(no_video=True, scoop=False, carry_mode="gentle")
        return ap

    ap = make_parser()
    pre_args, _ = ap.parse_known_args()
    if pre_args.json is None:
        ap.set_defaults(json=f"outputs/drift_probe_{pre_args.tag}.json")

    class DriftProbe(cwt.Test):
        base_xy0 = (-pre_args.drive_len, 0.0)

        def __init__(self, args):
            super().__init__(args)
            if args.standby and self.T is None:
                q = STANDBY.copy()
                q[:2] = self.base_xy0
                self.q = q
                self.robot.set_dofs_position(q - self._dq_off(), self.dofs)
                self.cmd = q.copy()
                self.robot.control_dofs_position(self.cmd - self._dq_off(), self.dofs)
            self.cp = self.scene.sim.coupler
            self.bl = self.robot.get_link("base_link")
            self.i_bl = self.cp._coupling_data.link_to_idx_local[self.bl]
            uipc.Timer.enable_all()
            if args.uipc_warn:
                uipc.Logger.set_level(uipc.Logger.Level.Warn)
            self.log_phase = None       # None: only the Newton count is kept (settle)
            self.rows = []
            self.settle_newton = []
            self._dp_prev_x = None
            self._dp_prev_ipc = None
            self.aborted = None
            self.t_wall0 = time.time()

        @staticmethod
        def _newton_count():
            try:
                rep = uipc.Timer.report_as_json()
            except Exception:
                return np.nan, np.nan
            best = [np.nan]

            def walk(n):
                if isinstance(n, dict):
                    name = str(n.get("name", ""))
                    if "newton" in name.lower() and "iteration" in name.lower():
                        c = n.get("count", np.nan)
                        if np.isnan(best[0]) or c > best[0]:
                            best[0] = c
                    for ch in n.get("children", []) or []:
                        walk(ch)
                elif isinstance(n, list):
                    for ch in n:
                        walk(ch)
            walk(rep)
            dur = rep.get("duration", np.nan) if isinstance(rep, dict) else np.nan
            return float(best[0]), float(dur) * 1e3 if np.isfinite(dur) else np.nan

        def step(self):
            t0 = time.time()
            super().step()
            frame_ms_wall = (time.time() - t0) * 1e3
            nc, _ = self._newton_count()
            if self.log_phase is None:
                self.settle_newton.append(nc)
                return
            self.log_row(nc, frame_ms_wall)

        def log_row(self, nc, frame_ms):
            g = self.bl.get_pos().cpu().numpy().reshape(-1).astype(float)
            gq = self.bl.get_quat().cpu().numpy().reshape(-1).astype(float)   # wxyz
            gyaw = math.atan2(2 * (gq[0] * gq[3] + gq[1] * gq[2]), 1 - 2 * (gq[2] ** 2 + gq[3] ** 2))
            Ti = np.array(self.cp._abd_data_by_link[self.bl][0].transform, float)
            Ta = np.array(self.cp._abd_transforms_by_link[self.bl][0], float)
            F = np.array(self.cp._coupling_data.out_forces[0, self.i_bl], float)
            Tq = np.array(self.cp._coupling_data.out_torques[0, self.i_bl], float)
            rv = _rotvec(Ti[:3, :3] @ Ta[:3, :3].T)
            x = self.cloth_pos().astype(float)
            c = x.mean(0)
            bc = Ti[:3, :3] @ np.array([*BASKET_CENTER_XY, 0.0]) + Ti[:3, 3]
            ln = np.linalg.norm(x[self.edges[:, 0]] - x[self.edges[:, 1]], axis=1)
            slip = (np.nan, np.nan, 0)
            if self._dp_prev_x is not None:
                fl = self._dp_prev_x[:, 2] < cwt.BASKET_FLOOR + 0.004
                if fl.sum() > 0:
                    dbask = (Ti[:3, 3] - self._dp_prev_ipc) / cwt.DT
                    v = (x[fl] - self._dp_prev_x[fl]) / cwt.DT - dbask
                    slip = (float(v[:, 0].mean()), float(v[:, 1].mean()), int(fl.sum()))
            self._dp_prev_x, self._dp_prev_ipc = x, Ti[:3, 3].copy()
            self.rows.append([self.nstep * cwt.DT, self.log_phase, self.cmd[0], self.cmd[1], g[0], g[1], g[2], gyaw,
                              *Ti[:3, 3], *Ta[:3, 3], *rv, *F, *Tq, nc, frame_ms, *c, c[0] - bc[0], c[1] - bc[1],
                              float(np.percentile(x[:, 2], 90) - cwt.BASKET_FLOOR),
                              float((ln / self.rest_len).max()), *slip])
            n = len(self.rows)
            if n % 50 == 0:
                r = self.rows[-1]
                print(f"[probe] t={r[0]:.2f} {PHASES[r[1]]:6s} steps/s={1e3 / max(frame_ms, 1e-3):.1f} "
                      f"newton={nc:.0f} cx_rel={r[COLS.index('cx_rel')] * 1e3:+.1f} mm "
                      f"p90={r[COLS.index('p90')] * 1e3:.1f} mm stretch={r[COLS.index('stretch')]:.2f} "
                      f"ipc-aim_x={(Ti[0, 3] - Ta[0, 3]) * 1e6:+.1f} um F_x={F[0]:+.1f} N", flush=True)

        def _wall_ok(self):
            a = self.args
            if a.max_wall > 0 and time.time() - T_START > a.max_wall:
                self.aborted = f"max_wall {a.max_wall} s at t={self.nstep * cwt.DT:.2f}"
                return False
            return True

        def hold_logged(self, dur, phase):
            self.log_phase = phase
            for _ in range(int(round(dur / cwt.DT))):
                if not self._wall_ok():
                    return False
                self.step()
            return True

        def after_settle(self, res):
            try:
                self._drive_logged()
            except Exception as e:   # keep the log of what ran
                import traceback
                traceback.print_exc()
                self.aborted = f"exception: {e!r}"[:300]
                print(f"[probe] {self.aborted}", flush=True)
            finally:
                self._save(res)
            return True   # end the run here (no grasp)

        def _drive_logged(self):
            a = self.args
            DT = cwt.DT
            print(f"[probe] settled at t={self.nstep * DT:.2f}; wall {time.time() - T_START:.0f} s", flush=True)
            ok = self.hold_logged(a.pre_s, 1)
            L = a.drive_len
            T = L / a.speed + a.ramp
            n = int(math.ceil(T / DT - 1e-9))
            t = np.arange(1, n + 1) * DT

            def ramp_s(u):
                return a.speed * a.ramp * (u ** 3 - u ** 4 / 2)
            s = np.where(t < a.ramp, ramp_s(np.clip(t / a.ramp, 0, 1)),
                         np.where(t < T - a.ramp, a.speed * a.ramp / 2 + a.speed * (t - a.ramp),
                                  L - ramp_s(np.clip((T - t) / a.ramp, 0, 1))))
            s[-1] = L
            s = np.minimum(s, L)
            start = np.array(self.base_xy0, float)
            path = np.stack([start[0] + s, np.full_like(s, start[1])], 1)
            off2 = self.q_base_off[:2]
            kin = a.base_mode == "kinematic"
            if kin:
                self.robot.set_dofs_kp(np.zeros(2), self.dofs[:2])
                self.robot.set_dofs_kv(np.zeros(2), self.dofs[:2])
            prev = start
            for k in range(n if ok else 0):
                if not self._wall_ok():
                    break
                tk = t[k]
                self.log_phase = 2 if tk <= a.ramp else (4 if tk > T - a.ramp else 3)
                xy = path[k]
                self.cmd[:2] = xy
                if kin:
                    self.robot.set_dofs_position(prev - off2, self.dofs[:2], zero_velocity=False)
                    self.robot.set_dofs_velocity((xy - prev) / DT, self.dofs[:2])
                self.step()
                prev = xy
            if kin:
                kp = np.array([8000.0, 8000.0])
                self.robot.set_dofs_kp(kp, self.dofs[:2])
                self.robot.set_dofs_kv(2 * np.sqrt(kp), self.dofs[:2])
                self.robot.set_dofs_position(prev - off2, self.dofs[:2], zero_velocity=False)
                self.robot.set_dofs_velocity(np.zeros(2), self.dofs[:2])
            if self.aborted is None:
                self.hold_logged(a.after_s, 5)

        def _save(self, res):
            a = self.args
            DT = cwt.DT
            L = a.drive_len
            Larr = np.array(self.rows, float).reshape(-1, len(COLS))
            log = a.log or f"outputs/drift_probe_{a.tag}.npz"
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            meta = dict(tag=a.tag, base_mode=a.base_mode, semi_implicit=a.semi_implicit, semi_beta=a.semi_beta, newton_iters=a.newton_iters,
                        newton_tol=a.newton_tol, eps_v=a.eps_v, cs_trans=a.cs_trans, cs_rot=a.cs_rot, mu=a.mu,
                        d_hat=a.d_hat, drive_len=L, speed=a.speed, ramp=a.ramp, dt=DT, seed=a.seed,
                        aborted=self.aborted, settle_newton_mean=float(np.nanmean(self.settle_newton))
                        if self.settle_newton else None, wall_s=round(time.time() - T_START, 1))
            np.savez_compressed(log, rows=Larr, cols=np.array(COLS), settle_newton=np.array(self.settle_newton, float),
                                meta=json.dumps(meta))
            res["drift_probe"] = analyze(Larr, DT, a.newton_iters, meta) if len(Larr) > 5 else dict(meta=meta)
            res["drift_probe_log"] = log

    cwt.main(DriftProbe, ap)


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--analyze":
        for p in sys.argv[2:]:
            z = np.load(p, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            print(f"==== {p}")
            analyze(z["rows"], meta["dt"], meta["newton_iters"], meta)
        return
    run_probe()


if __name__ == "__main__":
    main()
