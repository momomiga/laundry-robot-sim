"""Can the parked robot's PiPER arm reach cloth lying on the room floor? (CPU only, plain Genesis, no IPC.)

For a grid of floor points around the parked robot (robot frame: +x forward, +y left; base footprint P.BASE_SIZE
skipped), TCP heights z (top of a crumpled cloth heap; TCP = fingertip centre, the finger bottoms are ~1.4 cm lower),
lift values and gripper tilts it searches collision-free IK solutions and records the feasible (lift, tilt) set and the
minimum clearance. Open gripper (0.035) is assumed: it is the wider, more restrictive pre-grasp state.

Tool orientation: scripts/arm_planner.quat_tilt_keep_yaw(az, yaw, tilt): straight down, tilted by `tilt` so the tool leans
outward along the azimuth az (arm base -> point); yaw = az + {0, 45, 90} deg (free; the best clearance over the
samples is kept). tilt 0 = vertical top-down grasp.

Validity of a state = ArmPlanner.check(q) (Genesis contacts incl. arm-vs-floor-plane, clearance margins arm vs
mast/chassis/basket) PLUS floor clearance of this script: ArmPlanner.check() has no margin against the floor plane
(the SDF only knows boxes), only the exact plane contact (penetration >= 1e-5 m). The wheels/chassis do not touch the
plane at rest (the platform is 5 cm up), so the plane does not invalidate every state. Here the lowest point of every
arm link hull (vertices + surface samples) must be >= --floor-margin above z = 0. The washer is moved far away.

Outputs: outputs/floor_reach.json, outputs/floor_reach.png. Run: .venv/bin/python scripts/floor_reach_check.py [--quick]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))
import params as P
from arm_planner import I_GRIP, ArmPlanner, quat_tilt_keep_yaw
from scipy.spatial.transform import Rotation

OUT = ROOT / "outputs"
HALF = np.array(P.BASE_SIZE) / 2
GRIP_OPEN = 0.035
YAW_OFFS = (0.0, 45.0, 90.0)


def floor_clearance(pl):
    """Lowest z [m] of any arm link hull point (robot already at q)."""
    gp = pl.rs.get_geoms_pos().cpu().numpy().reshape(-1, 3)
    gq = pl.rs.get_geoms_quat().cpu().numpy().reshape(-1, 4)
    gR = Rotation.from_quat(gq[:, [1, 2, 3, 0]]).as_matrix()
    return min(float((pl.arm_pts[ai] @ gR[ai].T + gp[ai])[:, 2].min()) for ai in pl.arm_geoms)


class Reach:
    def __init__(self, pl, a):
        self.pl, self.a = pl, a
        self.base = {}      # lift -> start configuration (standby, open gripper)
        for lift in a.lifts:
            s = pl.find_standby(lift)
            assert s is not None, f"no standby at lift {lift}"
            s[I_GRIP] = (GRIP_OPEN, -GRIP_OPEN)
            self.base[lift] = s
        self.warm = {}      # (lift, tilt) -> last solution
        self.lo, self.hi = pl.lim_lo[4:10], pl.lim_hi[4:10]

    def inits(self, lift, key, tcp):
        pl = self.pl
        q_ref = self.base[lift]
        out = []
        if key in self.warm:
            out.append(self.warm[key])
        out.append(q_ref)
        b = pl.link1_pos(q_ref)
        az = math.atan2(tcp[1] - b[1], tcp[0] - b[0])
        for j1 in (az, az + math.pi, az - math.pi):
            if abs(j1) > 2.618:
                continue
            for sh in ([0.5, -0.8, 0, 0.5, 0], [1.0, -1.2, 0, 0.6, 0], [1.5, -1.0, 0, 0.8, 0], [0.3, -0.3, 0, 0.3, 0],
                       [2.0, -1.8, 0, 0.0, 0]):
                q = q_ref.copy()
                q[4:10] = [j1, *sh]
                out.append(q)
        for _ in range(self.a.n_random):
            q = q_ref.copy()
            q[4:10] = self.lo + pl.rng.random(6) * (self.hi - self.lo)
            out.append(q)
        return out

    def solve(self, tcp, lift, tilt):
        """Best (max clearance) valid solution over yaw samples: (q, clearance) or None."""
        pl, a = self.pl, self.a
        key = (lift, tilt)
        b = pl.link1_pos(self.base[lift])
        az = math.atan2(tcp[1] - b[1], tcp[0] - b[0])
        best = None
        for off in a.yaw_offs:
            quat = quat_tilt_keep_yaw(az, math.degrees(az) + off, tilt)
            for k, q0 in enumerate(self.inits(lift, key, tcp)):
                q, ep, er = pl._ik1(q0, tcp, quat, seed=k, max_samples=1)
                if ep > 0.003 or er > 0.03:
                    continue
                ok, _, c = pl.check(q)   # leaves the robot at q
                if not ok:
                    continue
                fc = floor_clearance(pl)
                if fc < a.floor_margin:
                    continue
                c = min(c, fc)
                self.warm[key] = q
                if best is None or c > best[1]:
                    best = (q, c)
                break
        return best


def dist_to_footprint(x, y):
    return float(np.hypot(max(abs(x) - HALF[0], 0), max(abs(y) - HALF[1], 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=float, default=0.02, help="grid spacing [m]")
    ap.add_argument("--extent", type=float, default=0.8, help="x, y in [-extent, extent] [m]")
    ap.add_argument("--zs", type=float, nargs="+", default=[0.03, 0.06, 0.10])
    ap.add_argument("--lifts", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    ap.add_argument("--tilts", type=float, nargs="+", default=[0.0, 15.0, 30.0, 45.0, 60.0])
    ap.add_argument("--n-random", type=int, default=4, help="random IK initial values per target")
    ap.add_argument("--yaws", type=int, default=3, help="yaw samples (of 0/45/90 deg relative to the azimuth)")
    ap.add_argument("--floor-margin", type=float, default=0.003)
    ap.add_argument("--max-reach", type=float, default=0.85, help="skip points farther than this from the arm base")
    ap.add_argument("--quick", action="store_true", help="5 cm grid, 1 yaw, 2 random inits")
    ap.add_argument("--out", default=str(OUT / "floor_reach"))
    a = ap.parse_args()
    if a.quick:
        a.grid, a.yaws, a.n_random = max(a.grid, 0.05), 1, 2
    a.yaw_offs = YAW_OFFS[:a.yaws]
    OUT.mkdir(exist_ok=True)

    pl = ArmPlanner(washer_pos=(5.0, 5.0, 0.003), backend="cpu", seed=0)
    R = Reach(pl, a)
    ab = np.array(P.CARRIAGE_XY)
    n = round(a.extent / a.grid)
    xs = np.arange(-n, n + 1) * a.grid
    pts = [(x, y) for x in xs for y in xs
           if not (abs(x) < HALF[0] - 1e-9 and abs(y) < HALF[1] - 1e-9) and np.hypot(x - ab[0], y - ab[1]) <= a.max_reach]
    t0 = time.time()

    # sanity: inside the basket, above its floor, vertical tool (the existing IPC grasp is at (0, -0.071, 0.307))
    bfloor = P.BASE_Z1 + P.BASKET_WALL
    sanity = []
    for lift in a.lifts:
        for dz in (0.03, 0.05):
            r = R.solve(np.array([0.0, -0.071, bfloor + dz]), lift, 0.0)
            sanity.append({"lift": lift, "z": bfloor + dz, "tilt": 0.0, "ok": r is not None, "clearance": None if r is None else r[1]})
    print("sanity (basket, tilt 0):", [(s["lift"], round(s["z"], 3), s["ok"]) for s in sanity], flush=True)

    res = {}
    for z in a.zs:
        cells = []
        R.warm.clear()
        for i, (x, y) in enumerate(pts):
            # serpentine order keeps the warm start close
            feas = []
            for lift in a.lifts:
                for tilt in a.tilts:
                    r = R.solve(np.array([x, y, z]), lift, tilt)
                    if r is not None:
                        feas.append([lift, tilt, round(r[1], 4)])
            cells.append({"x": round(float(x), 3), "y": round(float(y), 3), "feasible": feas,
                              "min_clearance": min([f[2] for f in feas]) if feas else None})
            if i % 200 == 0:
                print(f"z={z} {i}/{len(pts)} {time.time() - t0:.0f}s", flush=True)
        res[str(z)] = cells
    summary = summarize(res)
    out = {"args": {k: v for k, v in vars(a).items() if k != "yaw_offs"}, "yaw_offs": list(a.yaw_offs),
               "base_footprint": list(P.BASE_SIZE), "arm_base_xy": list(ab), "basket_center_xy": list(P.BASKET_CENTER_XY),
               "basket_inner": list(P.BASKET_INNER), "sanity": sanity, "summary": summary, "runtime_s": time.time() - t0,
               "cells": res}
    Path(a.out + ".json").write_text(json.dumps(out))
    plot(out, Path(a.out + ".png"))
    print(json.dumps(summary, indent=1))
    print(f"runtime {time.time() - t0:.0f}s; sanity ok: {all(s['ok'] for s in sanity)}")


def summarize(res):
    out = {}
    for z, cells in res.items():
        r = [c for c in cells if c["feasible"]]
        s = {"n_points": len(cells), "n_reachable": len(r)}
        if r:
            d = [dist_to_footprint(c["x"], c["y"]) for c in r]
            s["dist_from_base_edge_min_m"], s["dist_from_base_edge_max_m"] = round(min(d), 3), round(max(d), 3)
            sides = {"front(+x)": lambda c: c["x"] > HALF[0], "back(-x)": lambda c: c["x"] < -HALF[0],
                     "left(+y)": lambda c: c["y"] > HALF[1], "right(-y)": lambda c: c["y"] < -HALF[1]}
            s["by_side"] = {}
            for nm, f in sides.items():
                rs = [c for c in r if f(c)]
                tot = sum(1 for c in cells if f(c))
                if rs:
                    ds = [dist_to_footprint(c["x"], c["y"]) for c in rs]
                    s["by_side"][nm] = {"n": len(rs), "of": tot, "edge_dist_min": round(min(ds), 3),
                                            "edge_dist_max": round(max(ds), 3)}
                else:
                    s["by_side"][nm] = {"n": 0, "of": tot}
            tl = [min(f[1] for f in c["feasible"]) for c in r]
            s["min_tilt_hist"] = {str(t): tl.count(t) for t in sorted(set(tl))}
            s["lifts_used"] = sorted({f[0] for c in r for f in c["feasible"]})
            s["min_clearance_m"] = round(min(c["min_clearance"] for c in r), 4)
        out[z] = s
    return out


def plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    tilts = out["args"]["tilts"]
    lifts = out["args"]["lifts"]
    zs = list(out["cells"].keys())
    fig, axs = plt.subplots(2, len(zs), figsize=(5.2 * len(zs), 10.4), squeeze=False)
    cm = plt.get_cmap("viridis")(np.linspace(0, 0.95, len(tilts)))
    cm_l = plt.get_cmap("plasma")(np.linspace(0, 0.85, len(lifts)))
    g = out["args"]["grid"]
    for j, z in enumerate(zs):
        for row, (what, cols, vals) in enumerate((("smallest feasible tilt [deg]", cm, tilts),
                                                  ("smallest feasible lift [m]", cm_l, lifts))):
            ax = axs[row, j]
            for c in out["cells"][z]:
                if c["feasible"]:
                    v = min(f[1] for f in c["feasible"]) if row == 0 else min(f[0] for f in c["feasible"])
                    col = cols[vals.index(v)]
                else:
                    col = (0.9, 0.9, 0.9, 1)
                ax.add_patch(Rectangle((c["x"] - g / 2, c["y"] - g / 2), g, g, color=col, lw=0))
            hx, hy = out["base_footprint"][0] / 2, out["base_footprint"][1] / 2
            ax.add_patch(Rectangle((-hx, -hy), 2 * hx, 2 * hy, fill=False, ec="k", lw=1.5))
            bx, by = out["basket_center_xy"]
            ix, iy, _ = out["basket_inner"]
            ax.add_patch(Rectangle((bx - ix / 2, by - iy / 2), ix, iy, fill=False, ec="tab:red", lw=1.2, ls="--"))
            ax.plot(*out["arm_base_xy"], "k^", ms=9)
            e = out["args"]["extent"] + g
            ax.set_xlim(-e, e)
            ax.set_ylim(-e, e)
            ax.set_aspect("equal")
            ax.set_xlabel("robot x (forward) [m]")
            ax.set_ylabel("robot y (left) [m]")
            ax.set_title(f"TCP z = {float(z) * 100:.0f} cm: {what}")
            ax.grid(alpha=0.3)
            if j == 0:
                ax.legend(handles=[Rectangle((0, 0), 1, 1, color=c, label=str(v)) for c, v in zip(cols, vals)]
                          + [Rectangle((0, 0), 1, 1, color=(0.9, 0.9, 0.9), label="none")],
                          loc="upper left", fontsize=8, title=what)
    fig.suptitle("Floor reach of the parked robot (black box = base, red dashed = basket, triangle = arm base)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
