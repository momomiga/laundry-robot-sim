"""Diagnostics for scripts/cloth_ipc_washer_test.py (does not change its behaviour): per-step time series of the
gripper / arm / IPC coupling, plus the arm's collision geometry against the washer, mast, chassis and basket.

Writes an .npz (per-step arrays, small: ~100 floats per step) and a .json (segment table, collision summary, Genesis
collision pairs). Runs the unchanged Test scenario; only the options below are layered on top (monkeypatched before
the scene is built or set right after it):

  --arm-kp / --arm-kv-scale   joint1-6 PD gains (test: kp 300, kv = 2 sqrt(kp))
  --grip-kv-scale             joint7/8 kv multiplier (test: kv = 2 sqrt(kp))
  --one-way                   IPCCouplerOptions(two_way_coupling=False): no IPC force back onto Genesis
  --dt / --substeps           SimOptions (the move durations stay in seconds)
  --newton-stats              read the Newton iteration count per step (uipc.Timer, slower)

Usage (same scenario flags as the test script follow after `--`):
  flock /tmp/house_robot-gpu.lock .venv-ipc/bin/python scripts/cloth_ipc_washer_diag.py --tag base -- \
      --pre-open 0.015 --no-scoop ... --seed 0 --stop-after lift --attempts 1 --no-video
"""
import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))

import genesis as gs  # noqa: E402
import cloth_ipc_washer_test as W  # noqa: E402
W_SRC = Path(W.__file__).read_text().splitlines()
from scipy.spatial.transform import Rotation  # noqa: E402
import trimesh  # noqa: E402


def wxyz_to_R(q):
    q = np.asarray(q, float).reshape(-1, 4)
    return Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


# named factor windows for --switch-plan (kp7 / kv_mul: gripper joint7/8 gain and kv multiplier, arm_kv_mul: joint1-6
# kv multiplier, fmax: finger force limit [N]); a "#k" suffix makes repeated names distinct
SWITCH_FACTORS = {
    "base": {}, "base_after_off": {}, "coupling_force_off": {},
    "grip_kv_x2": dict(kv_mul=2.0), "grip_kv_x3": dict(kv_mul=3.0), "grip_kv_x5": dict(kv_mul=5.0),
    "grip_kv_x0.5": dict(kv_mul=0.5), "fmax80_unsat": dict(fmax=80.0), "fmax20_sat": dict(fmax=20.0),
    "grip_kp2000": dict(kp7=2000.0), "grip_kp2000_kv_x3": dict(kp7=2000.0, kv_mul=3.0),
    "arm_kv_x0.5": dict(arm_kv_mul=0.5), "arm_kv_x2": dict(arm_kv_mul=2.0), "arm_kv_x3": dict(arm_kv_mul=3.0),
}


class Diag(W.Test):
    def setup_diag(self, d):
        self.d = d
        rs = self.scene.sim.rigid_solver
        self.rs = rs
        a = self.args
        kp = np.array([8000, 8000, 3000, 8000] + [d.arm_kp] * 6 + [a.grip_kp] * 2, dtype=float)
        kv = 2 * np.sqrt(kp)
        kv[4:10] *= d.arm_kv_scale
        kv[10:12] *= d.grip_kv_scale
        self.robot.set_dofs_kp(kp, self.dofs)
        self.robot.set_dofs_kv(kv, self.dofs)
        self.gains = dict(kp=kp.tolist(), kv=kv.tolist())
        self.dt = W.DT
        self.coup = self.scene.sim.coupler
        self.coup_links = [self.robot.get_link(n) for n in ("base_link", "link6", "link7", "link8")]
        self.seg = []          # (start step, caller func, lineno, source line)
        self.rows = {}         # name -> list of arrays
        self.ik_steps = []     # steps at which ik() was entered
        # ---- collision geometry: arm sample points (geom frame) and obstacle boxes (world frame, fixed or base) ----
        self.arm_names = [f"link{i}" for i in range(1, 9)]
        self.arm_geoms = []    # (link name, geom idx, local points)
        for n in self.arm_names:
            for g in self.robot.get_link(n).geoms:
                v, f = np.asarray(g.init_verts, float), np.asarray(g.init_faces)
                m = trimesh.Trimesh(v, f, process=False)
                pts = np.concatenate([v, trimesh.sample.sample_surface(m, 400, seed=0)[0]]) if len(f) else v
                self.arm_geoms.append((n, g.idx, pts))
        self.obst = []         # (group, geom idx, half extents, is_washer)
        base = self.robot.get_link("base_link")
        for g in list(self.washer.geoms) + list(base.geoms):
            v = np.asarray(g.init_verts, float)
            c, h = (v.max(0) + v.min(0)) / 2, (v.max(0) - v.min(0)) / 2
            if g.link.name == "base_link":
                cz = float(g.init_pos[2])
                hx = h
                grp = "chassis" if hx[2] > 0.09 and hx[0] > 0.2 else ("mast" if hx[2] > 0.3 else "basket")
            else:
                grp = "washer"
            self.obst.append((grp, g.idx, h, c))
        # verify that init_verts are in the geom frame (world verts == pose * init_verts)
        g0 = self.arm_geoms[2]
        p = rs.get_geoms_pos([g0[1]]).cpu().numpy().reshape(3)
        R = wxyz_to_R(rs.get_geoms_quat([g0[1]]).cpu().numpy())[0]
        wv = rs.geoms[g0[1]].get_verts().cpu().numpy().reshape(-1, 3)
        nv = len(rs.geoms[g0[1]].init_verts)
        self.frame_check_mm = float(np.abs((np.asarray(rs.geoms[g0[1]].init_verts) @ R.T + p) - wv[:nv]).max() * 1e3)
        self.pair_info = self.collision_pairs()
        self.newton_prev = None
        if d.newton_stats:
            import uipc
            uipc.Timer.enable_all()
            self.uipc = uipc

    def collision_pairs(self):
        """Valid Genesis collision pairs (stored at build by the collider), grouped by link / entity name."""
        rs = self.rs
        geoms = rs.geoms
        name = lambda g: "washer" if g.entity is self.washer else ("plane" if g.entity is not self.robot and
                                                                     g.entity is not self.cloth else g.link.name)
        pairs = {}
        for a, b in np.asarray(rs.collider._valid_collision_pairs):
            k = " - ".join(sorted([name(geoms[a]), name(geoms[b])]))
            pairs[k] = pairs.get(k, 0) + 1
        return {"genesis_valid_pairs_geomcount": pairs, "n_valid": int(len(rs.collider._valid_collision_pairs)),
                "geoms_per_link": {l.name: len(l.geoms) for l in self.robot.links}}

    # ---------------- instrumentation ----------------
    def _mark(self):
        fr = inspect.stack()[2]
        # source text as imported (inspect reads the file on disk, which may have been edited during the run)
        src = W_SRC if fr.filename == W.__file__ else None
        line = src[fr.lineno - 1].strip() if src and fr.lineno <= len(src) else (fr.code_context or [""])[0].strip()
        self.seg.append((self.nstep, fr.function, fr.lineno, line[:90]))

    def move(self, goal, dur):
        self._mark()
        return super().move(goal, dur)

    def hold(self, dur):
        self._mark()
        if self.d.switch_test and getattr(self, "_closed_pending", False):
            self._closed_pending = False
            self.switch_test()
            self._mark()
        return super().hold(dur)

    def switch_test(self):
        """Controlled comparison on one pinch state: right after closing, hold in windows that switch one factor at
        a time and back to the baseline (same cloth, same finger position). Window labels go into self.windows."""
        r, dofs = self.robot, self.dofs
        kp0 = np.array(self.gains["kp"])
        kv0 = np.array(self.gains["kv"])
        fmax0 = self.args.grip_fmax or 10.0
        win = self.d.switch_s
        self.windows = []

        def setg(kp7=None, kv_mul=1.0, fmax=None, arm_kv_mul=1.0):
            kp, kv = kp0.copy(), kv0.copy()
            if kp7 is not None:
                kp[10:12] = kp7
                kv[10:12] = 2 * np.sqrt(kp7)
            kv[10:12] *= kv_mul
            kv[4:10] *= arm_kv_mul
            r.set_dofs_kp(kp, dofs)
            r.set_dofs_kv(kv, dofs)
            f = fmax or fmax0
            r.set_dofs_force_range(np.full(2, -f), np.full(2, f), dofs[10:12])

        orig = self.coup._apply_abd_coupling_forces
        plan = [("base", {}), ("grip_kv_x3", dict(kv_mul=3.0)), ("base", {}), ("fmax80_unsat", dict(fmax=80.0)),
                ("base", {}), ("fmax20_sat", dict(fmax=20.0)), ("base", {}), ("grip_kp2000", dict(kp7=2000.0)),
                ("base", {}), ("arm_kv_x3", dict(arm_kv_mul=3.0)), ("base", {}), ("coupling_force_off", {}),
                ("base_after_off", {})]
        if self.d.switch_plan:  # comma-separated window names, see SWITCH_FACTORS
            plan = [(n, SWITCH_FACTORS[n.split("#")[0]]) for n in self.d.switch_plan.split(",")]
        for name, kw in plan:
            setg(**kw)
            if name == "coupling_force_off":
                self.coup._apply_abd_coupling_forces = lambda: None
            st = self.nstep
            self._mark()
            W.Test.hold(self, win)
            self.coup._apply_abd_coupling_forces = orig
            self.windows.append((name, st, self.nstep))
        setg()

    def gripper(self, v, dur=0.8):
        self._mark()
        if v == W.CLOSED and inspect.stack()[1].function == "attempt":
            self._closed_pending = True
        g = self.cmd.copy()
        g[10], g[11] = v, -v
        return W.Test.move(self, g, dur)

    def ik(self, *a, **k):
        self.ik_steps.append(self.nstep)
        return super().ik(*a, **k)

    def _rec(self, k, v):
        self.rows.setdefault(k, []).append(np.asarray(v, np.float32).reshape(-1))

    def step(self):
        super().step()
        r = self.robot
        dofs = self.dofs
        self._rec("q", self.q)
        self._rec("dq", r.get_dofs_velocity(dofs).cpu().numpy())
        self._rec("tau", r.get_dofs_control_force(dofs).cpu().numpy())
        self._rec("cmd", self.cmd)
        lk = [self.link6] + self.fingers
        idx = [l.idx_local for l in lk]
        self._rec("lpos", r.get_links_pos(idx).cpu().numpy())
        self._rec("lvel", r.get_links_vel(idx).cpu().numpy())
        self._rec("lang", r.get_links_ang([self.link6.idx_local]).cpu().numpy())
        self._rec("tcp", self.tcp_world())
        if self.d.mass_mat:
            M = r.get_mass_mat().cpu().numpy().reshape(r.n_dofs, r.n_dofs)
            self._rec("mass_diag", np.diag(M)[np.asarray(dofs[4:12])])
        # IPC copy vs Genesis (translation mm, rotation deg) and coupling force/torque per coupled link
        cd = self.coup._coupling_data
        lag, frc = [], []
        for l in self.coup_links:
            T = self.coup._abd_data_by_link[l][0].transform
            Tg = self.coup._abd_transforms_by_link[l][0]
            dp = T[:3, 3] - Tg[:3, 3]
            ang = np.degrees(Rotation.from_matrix(T[:3, :3] @ Tg[:3, :3].T).magnitude())
            lag += [*dp * 1e3, ang]
            j = cd.link_to_idx_local[l]
            frc += [*cd.out_forces[0, j], *cd.out_torques[0, j]]
        self._rec("lag", lag)        # 4 links x (dx, dy, dz [mm], rot [deg]); lag = IPC pose - Genesis aim pose
        self._rec("cforce", frc)     # 4 links x (fx fy fz tx ty tz)
        sq, fz = self.squeeze_n()
        self._rec("squeeze", [sq, fz])
        # Genesis contacts involving the robot
        try:
            c = r.get_contacts()
            la = c["link_a"].cpu().numpy().reshape(-1)
            lb = c["link_b"].cpu().numpy().reshape(-1)
            pen = c["penetration"].cpu().numpy().reshape(-1) if "penetration" in c else np.zeros(len(la))
            arm = {l.idx for l in (self.robot.get_link(n) for n in self.arm_names)}
            m = np.array([(a_ in arm) or (b_ in arm) for a_, b_ in zip(la, lb)], bool)
            self._rec("gcontact", [int(m.sum()), float(pen[m].max()) if m.any() else 0.0])
            if m.any() and len(self.rows.get("gc_pairs_seen", [])) < 200:
                self.rows.setdefault("gc_pairs_seen", []).append(
                    np.array([self.nstep] + [v for ab in zip(la[m][:4], lb[m][:4]) for v in ab], np.float32))
        except Exception as e:  # noqa: BLE001
            self._rec("gcontact", [-1, 0])
            self.gc_err = repr(e)[:200]
        # arm vs obstacle penetration (analytic box SDF of sampled arm surface points)
        gp = self.rs.get_geoms_pos().cpu().numpy().reshape(-1, 3)
        gq = self.rs.get_geoms_quat().cpu().numpy().reshape(-1, 4)
        Rg = wxyz_to_R(gq)
        groups = ["washer", "mast", "chassis", "basket"]
        depth = np.full((8, 4), -1.0)   # max penetration [m] (>0 = inside); <0 = minus clearance
        for n, gi, pts in self.arm_geoms:
            w = pts @ Rg[gi].T + gp[gi]
            li = int(n[4:]) - 1
            for grp, oi, h, c in self.obst:
                loc = (w - gp[oi]) @ Rg[oi] - c
                qd = np.abs(loc) - h
                out = np.linalg.norm(np.maximum(qd, 0), axis=1)
                ins = np.minimum(qd.max(1), 0)
                sdf = out + ins
                gi_ = groups.index(grp)
                depth[li, gi_] = max(depth[li, gi_], float(-sdf.min()))
        self._rec("depth", depth)
        if self.d.newton_stats:
            rep = self.uipc.Timer.report_as_json()
            self._rec("newton", [self._newton_count(rep)])

    def _newton_count(self, rep):
        """Total count of the 'Newton Iteration' timer (cumulative) -> per-step difference."""
        tot = [0]

        def walk(n):
            if isinstance(n, dict):
                if "name" in n and "newton" in str(n["name"]).lower() and "iter" in str(n["name"]).lower():
                    tot[0] += int(n.get("count", 0))
                for v in n.values():
                    walk(v)
            elif isinstance(n, list):
                for v in n:
                    walk(v)
        walk(rep)
        if self.nstep == 10:
            (Path(self.d.outdir) / f"{self.d.tag}_timer_step10.json").write_text(json.dumps(rep, indent=1, default=str))
        return tot[0]   # Timer.report_as_json() is per frame (uipc.stats.SimulationStats.collect)


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        i = argv.index("--")
        mine, rest = argv[:i], argv[i + 1:]
    else:
        mine, rest = argv, []
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arm-kp", type=float, default=300.0)
    ap.add_argument("--arm-kv-scale", type=float, default=1.0)
    ap.add_argument("--grip-kv-scale", type=float, default=1.0)
    ap.add_argument("--one-way", action="store_true")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--substeps", type=int, default=1)
    ap.add_argument("--newton-stats", action="store_true")
    ap.add_argument("--switch-test", action="store_true", help="after the first close: hold windows switching one "
                    "factor at a time (see Diag.switch_test)")
    ap.add_argument("--switch-s", type=float, default=0.6)
    ap.add_argument("--switch-plan", default=None, help="comma-separated window names from SWITCH_FACTORS "
                    "(default: the fixed plan in Diag.switch_test)")
    ap.add_argument("--mass-mat", action="store_true", help="store the joint-space mass matrix diagonal (incl. "
                    "armature) of joints 1-8 per step (mass_diag)")
    ap.add_argument("--outdir", default=str(ROOT / "outputs" / "ipc_diag"))
    d = ap.parse_args(mine)

    # scenario args: reuse the test script's parser
    sys.argv = [sys.argv[0]] + rest
    parser = W.make_parser()
    args = parser.parse_args(rest)

    W.DT = d.dt
    _Sim, _Cpl = gs.options.SimOptions, gs.options.IPCCouplerOptions

    def SimOpts(**k):
        k["dt"], k["substeps"] = d.dt, d.substeps
        return _Sim(**k)

    def CplOpts(**k):
        if d.one_way:
            k["two_way_coupling"] = False
        return _Cpl(**k)
    gs.options.SimOptions, gs.options.IPCCouplerOptions = SimOpts, CplOpts

    out = Path(d.outdir)
    out.mkdir(parents=True, exist_ok=True)
    t = Diag(args)
    gs.options.SimOptions, gs.options.IPCCouplerOptions = _Sim, _Cpl
    t.setup_diag(d)
    if not args.no_video:
        t.start_video(str(out / f"{d.tag}.mp4"))
    t.res = {"seed": args.seed}
    t0 = time.time()
    try:
        t.run()
    except Exception as e:  # noqa: BLE001
        t.res["error"] = repr(e)[:300]
    finally:
        t.finish_video()
    arr = {k: np.stack(v) for k, v in t.rows.items() if k != "gc_pairs_seen"}
    if "gc_pairs_seen" in t.rows:
        L = max(len(v) for v in t.rows["gc_pairs_seen"])
        arr["gc_pairs_seen"] = np.stack([np.pad(v, (0, L - len(v)), constant_values=-1) for v in t.rows["gc_pairs_seen"]])
    arr["seg_start"] = np.array([s[0] for s in t.seg])
    arr["ik_steps"] = np.array(t.ik_steps)
    np.savez_compressed(out / f"{d.tag}.npz", **arr)
    meta = dict(tag=d.tag, diag=vars(d), args=vars(args), res=t.res, gains=t.gains,
                seg=[list(s) for s in t.seg], frame_check_mm=t.frame_check_mm, pairs=t.pair_info,
                link_names={l.idx: l.name for l in t.rs.links}, wall_s=round(time.time() - t0, 1),
                steps=t.nstep, gc_err=getattr(t, "gc_err", None), windows=getattr(t, "windows", None),
                obst=[(g, int(i), [round(float(v), 3) for v in h]) for g, i, h, _ in t.obst])
    (out / f"{d.tag}.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({k: meta[k] for k in ("tag", "steps", "wall_s", "frame_check_mm", "pairs", "gc_err")}, default=str))
    print({k: t.res.get(k) for k in ("attempts_log", "error", "success", "lift_success")})


if __name__ == "__main__":
    main()
