"""PD-tracking check of a planned trajectory (outputs/*_traj.npz from arm_planner_test.py): the commanded joint
trajectory is fed to position control in a plain Genesis scene with the IPC test's gains (cloth_ipc_washer_test.py:
kp base/lift 8000/8000/3000/8000, arm 300, gripper --grip-kp; kv = 2 sqrt(kp)), gravity on, collisions off (in the IPC
scene links 1-5 have no contacts), and the *actual* joint positions are measured for penetration / clearance with
arm_planner.ArmPlanner (Genesis narrow phase + box SDF). --ref replays a recorded IPC run's cmd the same way and
compares the lag with the recorded q (model check).
  .venv/bin/python scripts/arm_planner_dyn_replay.py outputs/arm_planner_test_traj [--ref outputs/ipc_diag/full_base]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from arm_planner import ArmPlanner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--arm-kp", type=float, default=300.0)
    ap.add_argument("--grip-kp", type=float, default=8000.0)
    ap.add_argument("--settle", type=float, default=1.0, help="hold the first command this long first [s]")
    a = ap.parse_args()
    import json
    P = ArmPlanner()
    gs = P.gs
    sc = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
                  rigid_options=gs.options.RigidOptions(enable_collision=False), show_viewer=False)
    r = sc.add_entity(gs.morphs.URDF(file=str(ROOT / "assets/robot/house_robot.urdf"), fixed=True))
    sc.build()
    kp = np.array([8000, 8000, 3000, 8000] + [a.arm_kp] * 6 + [a.grip_kp] * 2, float)
    r.set_dofs_kp(kp)
    r.set_dofs_kv(2 * np.sqrt(kp))

    def run(C):
        r.set_dofs_position(C[0], zero_velocity=True)
        for _ in range(int(a.settle / 0.01)):
            r.control_dofs_position(C[0])
            sc.step()
        Q = []
        for c in C:
            r.control_dofs_position(c)
            sc.step()
            Q.append(r.get_dofs_position().cpu().numpy().reshape(-1))
        return np.array(Q)

    if a.ref:
        z = np.load(a.ref + ".npz")
        Qr = run(z["cmd"])
        e_sim, e_ipc = np.abs(Qr - z["cmd"])[:, 4:10], np.abs(z["q"] - z["cmd"])[:, 4:10]
        print("model check on", a.ref, ": max |q-cmd| joint1-6 [rad] plain-Genesis PD",
              np.round(e_sim.max(0), 3).tolist(), "| recorded IPC", np.round(e_ipc.max(0), 3).tolist())
    C = np.load(a.traj + ".npz")["q"]
    seg = json.load(open(a.traj + ".json"))["seg"]
    Q = run(C)
    lag = np.abs(Q - C)[:, 4:10]
    starts = [s[0] for s in seg] + [len(C)]
    rows = []
    print("segment | max |q-cmd| j1-6 [rad] | TCP deviation max [mm] | max Genesis pen [mm] | max SDF depth [mm] | "
          "min clearance [mm] (actual q) | min clearance [mm] (planned)")
    for i, s in enumerate(seg):
        pen = dep = 0.0
        cl = cp = np.inf
        dev = 0.0
        for k in range(starts[i], starts[i + 1]):
            for which, q in (("a", Q[k]), ("p", C[k])):
                con = P.contacts(q)
                d = P.sdf(None, reach=0.05)
                m = min(d.values()) if d else np.inf
                if which == "a":
                    pen = max([pen] + [v for v in con.values()])
                    dep = max([dep] + [-v for v in d.values()])
                    cl = min(cl, m)
                    ta = P.fk_tcp(q)[0]
                else:
                    cp = min(cp, m)
                    dev = max(dev, float(np.linalg.norm(P.fk_tcp(q)[0] - ta)))
        rows.append(dict(seg=s[1], lag=float(lag[starts[i]:starts[i + 1]].max()), dev_mm=dev * 1e3, pen_mm=pen * 1e3,
                         sdf_mm=dep * 1e3, clear_mm=cl * 1e3, clear_plan_mm=cp * 1e3))
        x = rows[-1]
        print(f"{x['seg']} | {x['lag']:.3f} | {x['dev_mm']:.1f} | {x['pen_mm']:.2f} | {x['sdf_mm']:.2f} | "
              f"{x['clear_mm']:.1f} | {x['clear_plan_mm']:.1f}")
    Path(a.traj + "_dyn.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
