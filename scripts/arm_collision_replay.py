"""Kinematic replay of a recorded joint trajectory (outputs/ipc_diag/<tag>.npz from cloth_ipc_washer_diag.py) in a
plain Genesis scene (no IPC, no cloth): robot + washer as ordinary fixed rigid bodies, every collision pair enabled
(enable_neutral_collision=True so pairs overlapping at qpos0 are kept). Per replayed step it runs the Genesis
narrow phase (rigid_solver._kernel_detect_collision, as plan_path's collision check does) and reports the
penetration per (arm link, obstacle) pair, plus an analytic box-SDF depth of sampled arm surface points.

CPU backend, no GPU lock needed:
  .venv-ipc/bin/python scripts/arm_collision_replay.py outputs/ipc_diag/full_base [--every 2]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import genesis as gs  # noqa: E402
import trimesh  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import params as P  # noqa: E402

WASHER_XY, WASHER_Z = P.WASHER_BASE_XY, 0.003
NAMES = ["base_x_joint", "base_y_joint", "base_yaw_joint", "lift_joint"] + [f"joint{i}" for i in range(1, 9)]
ARM = [f"link{i}" for i in range(1, 9)]


def R_of(q):
    q = np.asarray(q, float).reshape(-1, 4)
    return Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--neutral", type=int, default=1, help="1: keep pairs overlapping at qpos0")
    a = ap.parse_args()
    d = np.load(a.base + ".npz")
    meta = json.load(open(a.base + ".json"))
    seg = meta["seg"]
    dt = meta["diag"]["dt"]
    Q = d["q"]
    gs.init(backend=gs.cpu, logging_level="warning")
    sc = gs.Scene(rigid_options=gs.options.RigidOptions(enable_self_collision=True,
                                                        enable_neutral_collision=bool(a.neutral)),
                  show_viewer=False)
    robot = sc.add_entity(gs.morphs.URDF(file=str(ROOT / "assets/robot/house_robot.urdf"), fixed=True))
    washer = sc.add_entity(gs.morphs.URDF(file=str(ROOT / "assets/washer/washer.urdf"), pos=(*WASHER_XY, WASHER_Z),
                                          euler=(0, 0, P.WASHER_BASE_YAW_DEG), fixed=True))
    sc.build()
    rs = sc.sim.rigid_solver
    dofs = [robot.get_joint(n).dofs_idx_local[0] for n in NAMES]
    geoms = rs.geoms

    def gname(g):
        if g.entity is washer:
            c = np.round(np.asarray(g.init_pos) + 0, 3)
            return f"washer{tuple(float(v) for v in c)}"
        n = g.link.name
        if n == "base_link":
            v = np.asarray(g.init_verts)
            h = (v.max(0) - v.min(0)) / 2
            return "chassis" if h[0] > 0.2 and h[2] > 0.09 else ("mast" if h[2] > 0.3 else "basket")
        return n

    names = [gname(g) for g in geoms]
    # analytic SDF setup
    arm_pts = []
    for ln in ARM:
        for g in robot.get_link(ln).geoms:
            v, f = np.asarray(g.init_verts, float), np.asarray(g.init_faces)
            pts = np.concatenate([v, trimesh.sample.sample_surface(trimesh.Trimesh(v, f, process=False), 400,
                                                                     seed=0)[0]])
            arm_pts.append((ln, g.idx, pts))
    obst = []
    for g in geoms:
        nm = names[g.idx]
        if nm.startswith("washer") or nm in ("mast", "chassis", "basket"):
            v = np.asarray(g.init_verts, float)
            obst.append((nm, g.idx, (v.max(0) - v.min(0)) / 2, (v.max(0) + v.min(0)) / 2))

    pairs_pen = {}     # (arm link, obstacle) -> list of (step, genesis pen, sdf depth)
    for k in range(0, len(Q), a.every):
        robot.set_dofs_position(Q[k], dofs, zero_velocity=True)
        rs._kernel_detect_collision()
        c = robot.get_contacts()
        ga = c["geom_a"].cpu().numpy().reshape(-1)
        gb = c["geom_b"].cpu().numpy().reshape(-1)
        pen = c["penetration"].cpu().numpy().reshape(-1)
        gen = {}
        for x, y, p in zip(ga, gb, pen):
            na, nb = names[x], names[y]
            if na in ARM and nb not in ARM:
                key = (na, nb)
            elif nb in ARM and na not in ARM:
                key = (nb, na)
            else:
                continue
            gen[key] = max(gen.get(key, 0.0), float(p))
        gp = rs.get_geoms_pos().cpu().numpy().reshape(-1, 3)
        gR = R_of(rs.get_geoms_quat().cpu().numpy())
        sdf = {}
        for ln, gi, pts in arm_pts:
            w = pts @ gR[gi].T + gp[gi]
            for nm, oi, h, cc in obst:
                loc = (w - gp[oi]) @ gR[oi] - cc
                qd = np.abs(loc) - h
                s = np.linalg.norm(np.maximum(qd, 0), axis=1) + np.minimum(qd.max(1), 0)
                dep = float(-s.min())
                if dep > 0:
                    sdf[(ln, nm)] = max(sdf.get((ln, nm), 0.0), dep)
        for key in set(gen) | set(sdf):
            pairs_pen.setdefault(key, []).append((k, gen.get(key, 0.0), sdf.get(key, 0.0)))

    starts = [s[0] for s in seg]

    def seg_of(k):
        i = max([j for j, s in enumerate(starts) if s <= k] or [0])
        return f"{seg[i][1]}:{seg[i][2]} {seg[i][3][:40]}"

    print(f"# {a.base}: {len(Q)} steps, replayed every {a.every}; neutral pairs kept={bool(a.neutral)}")
    print("arm link | obstacle | steps in contact | first-last t [s] | max Genesis pen [mm] | max SDF depth [mm] "
          "@ t | segment at max")
    rows = []
    for (ln, nm), v in sorted(pairs_pen.items()):
        v = np.array(v)
        m = np.argmax(np.maximum(v[:, 1], v[:, 2]))
        rows.append(dict(link=ln, obst=nm, n=len(v), t0=v[0, 0] * dt, t1=v[-1, 0] * dt,
                         gen_max_mm=v[:, 1].max() * 1e3, sdf_max_mm=v[:, 2].max() * 1e3, t_max=v[m, 0] * dt,
                         seg=seg_of(int(v[m, 0]))))
        r = rows[-1]
        print(f"{ln} | {nm} | {r['n']} | {r['t0']:.2f}-{r['t1']:.2f} | {r['gen_max_mm']:.1f} | {r['sdf_max_mm']:.1f} "
              f"@ {r['t_max']:.2f} | {r['seg']}")
    Path(a.base + "_replay.json").write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
