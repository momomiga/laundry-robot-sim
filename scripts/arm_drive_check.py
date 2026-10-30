"""Does the robot clear the house while driving along the demo route with the arm at its wait pose?

The robot (base_link boxes: chassis, mast, basket; carriage; PiPER link1-8) is placed at the planner's standby pose
(arm folded outside the towel drop envelope, lift = LIFT_CARRY 0.5, the wait pose of the IPC test) and its base is
translated along the demo_video.py route (room frame: START -> corridor line -> PARK, yaw fixed = PARK yaw) in the
parked-base frame, where the room collision boxes (assets/room/room.urdf, floor slabs left out) live. For every base
position the clearance of every robot geom (box vertices / hull vertices + surface samples) to every room box is
computed (analytic box SDF, negative = inside); reported per route leg: min clearance, base position, robot link,
room box. Flags: arm links (link1-8) < 0.03 m, body (chassis, mast, basket, carriage) < 0.02 m. Exit code 1 if flagged.

  .venv/bin/python scripts/arm_drive_check.py [--step 0.02] [--lift 0.5] [--start-room X,Y] [--out outputs/arm_drive_check.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))
import params as P  # noqa: E402
from arm_planner import ARM_LINKS, I_LIFT, ArmPlanner, TransferSeq  # noqa: E402
from arm_planner_test import LIFT_GRASP, WAIT  # noqa: E402
from house_frame import room_to_park  # noqa: E402

START = (1.35, 2.0)   # scripts/demo_video.py
ARM_MIN, BODY_MIN = 0.03, 0.02


def route_room(start=START):
    """Demo route waypoints (room frame, as demo_video.Demo.run)."""
    cx = P.PARK_POSE[0]
    return [tuple(start), (cx, 3.2), (cx, 4.6), (cx, 5.6), (cx, 6.6), tuple(P.PARK_POSE[:2])]


def box_label(T_park, size):
    """Short label of a room box from its room-frame centre/size (park frame boxes are transformed back)."""
    from house_frame import park_to_room
    c = park_to_room(T_park[:3, 3])
    return f"box c_room=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f}) size={tuple(round(float(v), 3) for v in size)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.02, help="base position step along the route [m]")
    ap.add_argument("--lift", type=float, default=WAIT["wait_lift"])
    ap.add_argument("--samples", type=int, default=3000, help="surface samples per robot geom")
    ap.add_argument("--start-room", default=None, metavar="X,Y", help="first waypoint, room frame (default %.2f,%.2f)"
                    % START)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "arm_drive_check.json"))
    a = ap.parse_args()
    start = tuple(float(v) for v in a.start_room.split(",")) if a.start_room else START

    AP = ArmPlanner(with_room=True, room_reach=None, rrt_inflate=0, seed=0)
    T = TransferSeq(AP, lift_grasp=LIFT_GRASP, **WAIT)
    standby = T.start()
    assert standby is not None, "no collision-free standby pose"
    q = standby.copy()
    q[I_LIFT] = a.lift
    print("standby (wait pose):", np.round(q, 3).tolist())

    # robot geoms at the origin (base joints 0), world points per geom
    import trimesh
    AP.set_q(q)
    gp = AP.rs.get_geoms_pos().cpu().numpy().reshape(-1, 3)
    gq = AP.rs.get_geoms_quat().cpu().numpy().reshape(-1, 4)
    gR = Rotation.from_quat(gq[:, [1, 2, 3, 0]]).as_matrix()
    pts, names = [], []
    for g in AP.rs.geoms:
        if g.entity is not AP.robot:
            continue
        v, f = np.asarray(g.init_verts, float), np.asarray(g.init_faces)
        s = trimesh.sample.sample_surface(trimesh.Trimesh(v, f, process=False), a.samples, seed=0)[0]
        pts.append(np.concatenate([v, s]) @ gR[g.idx].T + gp[g.idx])
        names.append(AP.gnames[g.idx])
    boxes = AP._room_boxes(None)
    labels = [box_label(Tb, sz) for Tb, sz in boxes]
    print(f"{len(boxes)} room boxes, {len(pts)} robot geoms ({sorted(set(names))})")

    def clearance(base_xy):
        """[n_geoms, n_boxes] min SDF of each robot geom (translated by base_xy) to each room box."""
        out = np.empty((len(pts), len(boxes)))
        off = np.array([base_xy[0], base_xy[1], 0.0])
        for j, (Tb, sz) in enumerate(boxes):
            h = sz / 2
            for i, w in enumerate(pts):
                loc = (w + off - Tb[:3, 3]) @ Tb[:3, :3]
                d = np.abs(loc) - h
                out[i, j] = (np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(1), 0)).min()
        return out

    wp = [room_to_park(np.array(p, float)) for p in route_room(start)]
    rows, flagged = [], False
    print("leg | min clearance mm | base (park) x,y | robot link | room box | flag")
    for leg, (p0, p1) in enumerate(zip(wp[:-1], wp[1:])):
        n = max(1, int(np.ceil(np.linalg.norm(p1[:2] - p0[:2]) / a.step)))
        best = dict(arm=(np.inf,), body=(np.inf,))
        for k in range(n + 1):
            xy = p0[:2] + (p1[:2] - p0[:2]) * k / n
            c = clearance(xy)
            for i, nm in enumerate(names):
                kind = "arm" if nm in ARM_LINKS else "body"
                j = int(c[i].argmin())
                if c[i, j] < best[kind][0]:
                    best[kind] = (float(c[i, j]), xy.copy(), nm, j)
        for kind, lim in (("arm", ARM_MIN), ("body", BODY_MIN)):
            cm, xy, nm, j = best[kind]
            flag = cm < lim
            flagged |= flag
            print(f"leg{leg} ({p0[0]:.2f},{p0[1]:.2f})->({p1[0]:.2f},{p1[1]:.2f}) {kind} | {cm * 1e3:.1f} | "
                  f"({xy[0]:.3f},{xy[1]:.3f}) | {nm} | {labels[j]} | {'FLAG' if flag else 'ok'}")
            rows.append(dict(leg=leg, kind=kind, clearance_mm=round(cm * 1e3, 1), base_park=np.round(xy, 3).tolist(),
                             link=nm, box=labels[j], flag=bool(flag)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(standby=np.round(q, 4).tolist(), lift=a.lift, rows=rows), indent=1))
    print("FLAGGED" if flagged else "ALL CLEAR", "json:", a.out)
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
