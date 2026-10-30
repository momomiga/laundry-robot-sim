"""Feasibility test: friction-only pinch of an IPC cloth (FEM.Cloth + IPC coupler) with the PiPER gripper.

No particle attachment: the cloth is held only by IPC contact + friction between the finger pads.
Runs in the separate IPC environment (.venv-ipc, see requirements-ipc.txt), not in the main .venv.

Genesis 1.4.4 facts this test relies on (installed source, genesis/engine/couplers/ipc_coupler/coupler.py):
- Scene(coupler_options=gs.options.IPCCouplerOptions(...)) swaps the legacy coupler for IPCCoupler (libuipc, CUDA).
  FEM entities (incl. FEM.Cloth -> StrainLimitingBaraffWitkinShell + DiscreteShellBending) are simulated by IPC.
- Rigid entities with Rigid(needs_coup=True) become IPC affine bodies (ABD). coup_type:
    'ipc_only' (default for bodies without DOFs; fixed table / ground plane),
    'external_articulation' (default for a fixed-base articulated robot: every link goes to IPC, the joint
      update comes back from IPC),
    'two_way_soft_constraint' (links follow the Genesis pose through a SoftTransformConstraint, the reaction
      wrench is applied back to Genesis; `coup_links` restricts which links enter IPC).
  Here the robot is 'two_way_soft_constraint' with coup_links=(link6, link7, link8): only the gripper is in IPC,
  the rest of the robot (base, mast, basket) stays a plain Genesis rigid body.
- Friction per contact pair = geometric_mean(cloth.friction_mu, rigid.coup_friction) (_register_contact_pairs).
- enable_rigid_rigid_contact=False: link6/link7/link8 meshes overlap at rest, IPC must not see them as a contact.

Layouts
  pile (default): the towel is released upright and collapses into a crumpled heap; pinch its top
  flat: towel lying flat; pinch 3 cm inside its near edge (fingers straddle the edge)
Notes
- bending_stiffness (DiscreteShellBending) of even 0.01 makes a 0.30 m towel behave like a stiff board (it tips
  over flat instead of crumpling), so the default is no bending term.
- The rest of the robot (basket, mast, base) is not in IPC: the cloth passes through it.

Usage
  flock /tmp/house_robot-gpu.lock .venv-ipc/bin/python scripts/cloth_ipc_grasp_test.py
  flock /tmp/house_robot-gpu.lock .venv-ipc/bin/python scripts/cloth_ipc_grasp_test.py --layout flat --grasp-in 0.03
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import genesis as gs
import numpy as np
import trimesh
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))
import params as P  # noqa: E402
from cloth_grasp_test import gpu_mem_mib, quat_down  # noqa: E402

DT = 0.01
W, H, FPS = 640, 480, 30
STEPS_PER_FRAME = 3
TCP = 0.15                       # link6 origin -> fingertip centre along link6 z [m]
OPEN, CLOSED = 0.035, 0.0        # joint7 travel (joint8 = -joint7)
TOWEL = 0.30
TABLE_TOP = 0.40
TABLE_CENTER = (0.42, 0.17)
TABLE_SIZE = (0.40, 0.50)
PILE_X = 0.29                    # x of the dropped towel (pile layout), within the PiPER reach for the carry


def grid_cloth(path, n, size):
    """Square n x n vertex grid in the xy plane, centred at the origin (alternating diagonals). The mesh is also
    the rest shape in IPC, so it must be flat (a pre-crumpled mesh would keep its shape)."""
    xs = np.linspace(-size / 2, size / 2, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    V = np.stack([X.ravel(), Y.ravel(), np.zeros(n * n)], 1)
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            F += [[a, b, c], [a, c, d]] if (i + j) % 2 == 0 else [[a, b, d], [b, c, d]]
    F = np.array(F)
    trimesh.Trimesh(V, F, process=False).export(path)
    edges = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    return edges, np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)


class Test:
    def __init__(self, args):
        self.args = args
        gs.init(backend=gs.gpu, logging_level="warning")
        self.mesh_path = os.path.join(tempfile.mkdtemp(prefix="cloth_ipc_"), "cloth.obj")
        self.edges, self.rest_len = grid_cloth(self.mesh_path, args.n, TOWEL)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=DT, substeps=args.substeps),
            coupler_options=gs.options.IPCCouplerOptions(
                contact_d_hat=args.d_hat,
                contact_friction_enable=True,
                enable_rigid_rigid_contact=False,
                newton_max_iterations=args.newton_iters,
            ),
            show_viewer=False,
            vis_options=gs.options.VisOptions(show_world_frame=False),
        )
        sc = self.scene
        sc.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=0.5))
        self.robot = sc.add_entity(
            gs.morphs.URDF(file=str(ROOT / "assets/robot/house_robot.urdf"), fixed=True),
            material=gs.materials.Rigid(coup_type="two_way_soft_constraint",
                                        coup_links=("link6", "link7", "link8"),
                                        coup_friction=args.mu))
        sc.add_entity(gs.morphs.Box(size=(*TABLE_SIZE, TABLE_TOP), pos=(*TABLE_CENTER, TABLE_TOP / 2), fixed=True),
                      material=gs.materials.Rigid(coup_friction=args.table_mu),
                      surface=gs.surfaces.Default(color=(0.55, 0.40, 0.28, 1)))
        if args.layout == "flat":  # towel lying flat, centred on the table (near -x edge 5 cm inside the table's)
            cpos, ceuler = (TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP + 0.004), (0, 0, 0)
        else:  # 'pile': released standing almost upright (yz plane, leaning +x) so it collapses into a heap
            cpos, ceuler = (PILE_X, TABLE_CENTER[1], TABLE_TOP + TOWEL / 2 + 0.01), (0, 100, 10)
        self.cloth = sc.add_entity(
            morph=gs.morphs.Mesh(file=self.mesh_path, pos=cpos, euler=ceuler),
            material=gs.materials.FEM.Cloth(E=args.E, nu=0.3, rho=args.rho, thickness=args.thickness,
                                            bending_stiffness=args.bending if args.bending >= 0 else None,
                                            friction_mu=args.mu),
            surface=gs.surfaces.Default(color=(0.95, 0.55, 0.15, 1.0)),
        )
        if args.cam == "close":  # looking along +y at the near towel edge: the closing axis is horizontal in view
            cam_pos, cam_look, fov = (0.27, -0.25, 0.55), (0.29, 0.17, 0.42), 30
        else:
            cam_pos, cam_look, fov = (0.95, -0.45, 0.85), (0.33, 0.25, 0.45), 55
        self.cam = sc.add_camera(res=(W, H), pos=cam_pos, lookat=cam_look, fov=fov, GUI=False)
        t0 = time.time()
        sc.build()
        self.build_s = time.time() - t0

        names = ["base_x_joint", "base_y_joint", "base_yaw_joint", "lift_joint"] + \
            [f"joint{i}" for i in range(1, 9)]
        self.dofs = [self.robot.get_joint(n).dofs_idx_local[0] for n in names]
        self.arm_dofs = self.dofs[4:10]
        self.link6 = self.robot.get_link("link6")
        self.fingers = [self.robot.get_link("link7"), self.robot.get_link("link8")]
        self.q = np.zeros(12)
        self.q[3] = args.lift
        self.robot.set_dofs_position(self.q, self.dofs)
        kp = np.array([8000, 8000, 3000, 8000] + [300] * 6 + [args.grip_kp] * 2, dtype=float)
        self.robot.set_dofs_kp(kp, self.dofs)
        self.robot.set_dofs_kv(2 * np.sqrt(kp), self.dofs)
        self.cmd = self.q.copy()
        self.robot.control_dofs_position(self.cmd, self.dofs)
        self.nstep = 0
        self.sim_time = 0.0
        self.max_speed = 0.0
        self.max_stretch = 0.0
        self.nan = False
        self.proc = None
        self.prev_x = None
        self.max_lag = 0.0
        self.coupler = self.scene.sim.coupler

    # ---------------- video ----------------
    def start_video(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "22", path], stdin=subprocess.PIPE)

    def frame(self):
        img = np.asarray(self.cam.render()[0]).astype(np.uint8)
        self.proc.stdin.write(img.tobytes())

    def finish_video(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()

    # ---------------- sim ----------------
    def cloth_pos(self):
        return self.cloth.get_state().pos.cpu().numpy().reshape(-1, 3)

    def step(self):
        self.robot.control_dofs_position(self.cmd, self.dofs)
        t0 = time.time()
        self.scene.step()
        self.sim_time += time.time() - t0
        self.nstep += 1
        self.q = self.robot.get_dofs_position(self.dofs).cpu().numpy().reshape(-1)
        if self.nstep % 5 == 0:
            self.max_lag = max(self.max_lag, max(self.ipc_lag().values()))
            x = self.cloth_pos()
            if not np.isfinite(x).all():
                self.nan = True
            else:
                if self.prev_x is not None:
                    v = np.linalg.norm(x - self.prev_x, axis=1).max() / (5 * DT)
                    self.max_speed = max(self.max_speed, float(v))
                ln = np.linalg.norm(x[self.edges[:, 0]] - x[self.edges[:, 1]], axis=1)
                self.max_stretch = max(self.max_stretch, float((ln / self.rest_len).max()))
            self.prev_x = x
        if self.proc and self.nstep % STEPS_PER_FRAME == 0:
            self.frame()

    def ipc_lag(self):
        """Distance [m] between each gripper link's pose in IPC (ABD) and in Genesis (debug; coupler internals)."""
        out = {}
        for link in [self.link6] + self.fingers:
            T = self.coupler._abd_data_by_link[link][0].transform
            out[link.name] = float(np.linalg.norm(T[:3, 3] - link.get_pos().cpu().numpy().reshape(-1)))
        return out

    def move(self, goal, dur, track=None):
        q0 = self.cmd.copy()
        n = max(1, int(dur / DT))
        for k in range(1, n + 1):
            a = k / n
            a = a * a * (3 - 2 * a)
            self.cmd = q0 + a * (goal - q0)
            self.step()
            if track is not None:
                track.append(self.cloth_pos()[:, 2].max())

    def hold(self, dur, track=None):
        for _ in range(int(dur / DT)):
            self.step()
            if track is not None:
                track.append(self.cloth_pos()[:, 2].max())

    def gripper(self, v, dur=0.8):
        g = self.cmd.copy()
        g[10], g[11] = v, -v
        self.move(g, dur)

    def ik(self, tcp, quat):
        """Damped-LS IK for the fingertip centre with several seeds (PiPER IK has local minima).
        set_dofs_position is only used for seeding and is restored before the next step, so the IPC copies of the
        gripper never see the teleport."""
        base = self.robot.get_link("link1").get_pos().cpu().numpy().reshape(-1)
        az = math.atan2(tcp[1] - base[1], tcp[0] - base[0])
        seeds = [self.cmd[4:10].copy()]
        for j1 in (0.0, az):
            seeds += [np.array([j1, 0.5, -0.8, 0, 0.5, 0]), np.array([j1, 1.0, -1.2, 0, 0.6, 0]),
                      np.array([j1, 1.5, -1.0, 0, 0.8, 0]), np.array([j1, 0.3, -0.3, 0, 0.3, 0])]
        best = None
        for s6 in seeds:
            seed = self.cmd.copy()
            seed[4:10] = s6
            self.robot.set_dofs_position(seed, self.dofs, zero_velocity=False)
            q_full0 = self.robot.get_dofs_position().cpu().numpy().reshape(-1)
            q, err = self.robot.inverse_kinematics(
                link=self.link6, pos=np.array(tcp, float), quat=quat, local_point=(0, 0, TCP),
                dofs_idx_local=self.arm_dofs, init_qpos=q_full0, max_solver_iters=200, return_error=True)
            e = err.cpu().numpy().reshape(-1)
            q = q.cpu().numpy().reshape(-1)[self.arm_dofs]
            score = np.abs(e[:3]).max() + 0.1 * np.abs(e[3:]).max() + 1e-3 * np.abs(q - self.cmd[4:10]).sum()
            if best is None or score < best[0]:
                best = (score, q, e)
        self.robot.set_dofs_position(self.q, self.dofs, zero_velocity=False)
        _, q, e = best
        assert np.abs(e[:3]).max() < 0.01 and np.abs(e[3:]).max() < 0.1, f"IK failed {tcp}: {e}"
        goal = self.cmd.copy()
        goal[4:10] = q
        return goal

    def arm_to(self, tcp, quat, dur, track=None):
        self.move(self.ik(tcp, quat), dur, track)

    def tcp_world(self):
        p = self.link6.get_pos().cpu().numpy().reshape(-1)
        qw = self.link6.get_quat().cpu().numpy().reshape(-1)
        R = Rotation.from_quat([qw[1], qw[2], qw[3], qw[0]])
        return p + R.apply([0, 0, TCP])

    def finger_verts(self):
        return [f.get_verts().cpu().numpy().reshape(-1, 3) for f in self.fingers]

    def finger_bottom_z(self):
        return min(float(v[:, 2].min()) for v in self.finger_verts())

    def finger_gap(self):
        """Distance between the two finger centroids [m] and the closing axis (unit, world)."""
        c = [v.mean(0) for v in self.finger_verts()]
        d = c[1] - c[0]
        return float(np.linalg.norm(d)), d / np.linalg.norm(d)

    def n_inside_fingers(self, x):
        """Cloth vertices inside the (convex) Genesis finger collision hulls = penetration indicator."""
        n = 0
        for v in self.finger_verts():
            n += int((Delaunay(v).find_simplex(x) >= 0).sum())
        return n

    def between_fingers(self, x):
        """Cloth vertices in the slab between the two fingers (within the fingers' extent)."""
        v0, v1 = self.finger_verts()
        c0, c1 = v0.mean(0), v1.mean(0)
        ax = (c1 - c0) / np.linalg.norm(c1 - c0)
        s = (x - c0) @ ax
        lo, hi = (v0 - c0) @ ax, (v1 - c0) @ ax
        allv = np.concatenate([v0, v1])
        zlo, zhi = allv[:, 2].min() - 0.003, allv[:, 2].min() + 0.04
        perp = x - c0 - np.outer(s, ax)
        perp[:, 2] = 0
        return int(((s > lo.max() - 0.005) & (s < hi.min() + 0.005) & (x[:, 2] > zlo) & (x[:, 2] < zhi) &
                    (np.linalg.norm(perp, axis=1) < 0.02)).sum())

    # ---------------- scenario ----------------
    def run(self):
        a = self.args
        res = self.res = {}
        self.hold(a.settle)
        x0 = self.cloth_pos()
        res["n_vertices"] = len(x0)
        res["settled_zmax"] = float(x0[:, 2].max())
        res["settled_zmin"] = float(x0[:, 2].min())
        res["settled_gap_to_table"] = float(np.median(x0[:, 2]) - TABLE_TOP)

        down = quat_down(a.yaw)
        edge_x = float(x0[:, 0].min())
        if a.layout == "flat":
            # finger bottoms `press` above the table top (negative = pressed in; Genesis rigid contact stops it)
            gx, gy = edge_x + a.grasp_in, float(np.median(x0[:, 1]))
            gz = TABLE_TOP
            fb_target = TABLE_TOP + a.press
        else:
            # highest point near the heap's centroid, fingers reaching `depth` below it (not into the table)
            c = x0[:, :2].mean(0)
            sel = np.where(np.linalg.norm(x0[:, :2] - c, axis=1) < 0.05)[0]
            i = sel[np.argmax(x0[sel, 2])]
            gx, gy, gz = (float(v) for v in x0[i])
            fb_target = max(TABLE_TOP + 0.002, gz - a.depth)
        res["grasp_point"] = [round(gx, 3), round(gy, 3), round(gz, 3)]
        res["finger_bottom_target_minus_table"] = fb_target - TABLE_TOP

        self.gripper(OPEN, 0.3)
        self.arm_to((gx, gy, gz + 0.15), down, 2.0)
        self.hold(0.3)
        gap_open, axis = self.finger_gap()
        res["finger_gap_open_centroids"] = gap_open
        res["closing_axis"] = [round(float(v), 3) for v in axis]
        off = self.tcp_world()[2] - self.finger_bottom_z()
        res["tcp_minus_finger_bottom"] = float(off)
        self.arm_to((gx, gy, fb_target + off), down, 1.5)
        self.hold(0.3)
        res["finger_bottom_at_grasp_minus_table"] = self.finger_bottom_z() - TABLE_TOP
        self.gripper(CLOSED, a.close_s)
        self.hold(0.5)
        x1 = self.cloth_pos()
        res["finger_gap_closed_centroids"] = self.finger_gap()[0]
        res["joint7_8_closed"] = [round(float(self.q[10]), 4), round(float(self.q[11]), 4)]
        res["ipc_lag_mm_closed"] = {k: round(v * 1e3, 2) for k, v in self.ipc_lag().items()}
        res["near_edge_shift_x_closed"] = float(x1[:, 0].min() - edge_x)
        res["closed_zmax"] = float(x1[:, 2].max())
        res["verts_between_fingers_closed"] = self.between_fingers(x1)
        res["verts_inside_fingers_closed"] = self.n_inside_fingers(x1)

        tip0 = self.tcp_world()
        track_lift = []
        self.arm_to((gx, gy, fb_target + off + 0.20), down, 2.0, track_lift)
        self.hold(0.5, track_lift)
        x2 = self.cloth_pos()
        res["tcp_lift_dz"] = float(self.tcp_world()[2] - tip0[2])
        res["lift_dzmax"] = float(x2[:, 2].max() - x1[:, 2].max())
        res["lift_dzmean"] = float(x2[:, 2].mean() - x1[:, 2].mean())
        res["lift_frac_verts_above_table+2cm"] = float((x2[:, 2] > TABLE_TOP + 0.02).mean())
        res["tip_to_cloth_top_after_lift"] = float(self.tcp_world()[2] - x2[:, 2].max())
        res["verts_between_fingers_lifted"] = self.between_fingers(x2)
        res["verts_inside_fingers_lifted"] = self.n_inside_fingers(x2)
        res["joint7_8_lifted"] = [round(float(self.q[10]), 4), round(float(self.q[11]), 4)]

        track = []
        # carry towards the robot (PiPER reach shrinks quickly outwards); shorten until reachable
        for k in (1.0, 0.75, 0.5):
            try:
                goal = self.ik((gx + k * a.carry_dx, gy + k * a.carry_dy, fb_target + off + 0.20), down)
                break
            except AssertionError:
                continue
        else:
            raise AssertionError("carry target unreachable")
        res["carry_vector"] = [round(k * a.carry_dx, 3), round(k * a.carry_dy, 3)]
        self.move(goal, 2.0, track)
        self.hold(1.0, track)
        x3 = self.cloth_pos()
        tip = self.tcp_world()
        res["carry_zmax_min"] = float(min(track))
        res["carry_zmax_end"] = float(x3[:, 2].max())
        res["carry_zmean_end"] = float(x3[:, 2].mean())
        top = x3[np.argmax(x3[:, 2])]
        res["carry_cloth_top_to_tip_xy"] = float(np.linalg.norm(top[:2] - tip[:2]))
        res["carry_cloth_top_below_tip"] = float(tip[2] - top[2])
        res["held_after_carry"] = bool(x3[:, 2].max() > TABLE_TOP + 0.10 and
                                       np.linalg.norm(top[:2] - tip[:2]) < 0.05)
        res["verts_inside_fingers_carry"] = self.n_inside_fingers(x3)

        self.gripper(OPEN, 0.6)
        self.hold(2.0)
        x4 = self.cloth_pos()
        res["released_zmax"] = float(x4[:, 2].max())
        res["released_zmean"] = float(x4[:, 2].mean())
        on_table = (np.abs(x4[:, 0] - TABLE_CENTER[0]) < TABLE_SIZE[0] / 2) & \
            (np.abs(x4[:, 1] - TABLE_CENTER[1]) < TABLE_SIZE[1] / 2)
        res["released_min_z_over_table_minus_top"] = float(x4[on_table, 2].min() - TABLE_TOP) if on_table.any() \
            else None
        res["max_ipc_gs_lag_mm"] = self.max_lag * 1e3
        res["nan"] = self.nan
        res["max_vertex_speed"] = self.max_speed
        res["max_edge_stretch_ratio"] = self.max_stretch
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=31, help="cloth grid vertices per side (31 -> 1 cm edges)")
    ap.add_argument("--mu", type=float, default=0.8, help="cloth friction_mu and gripper coup_friction")
    ap.add_argument("--table-mu", type=float, default=0.5)
    ap.add_argument("--E", type=float, default=5e4, help="cloth Young's modulus [Pa]")
    ap.add_argument("--rho", type=float, default=300.0, help="cloth density [kg/m^3] (x thickness = areal)")
    ap.add_argument("--thickness", type=float, default=0.001)
    ap.add_argument("--bending", type=float, default=-1.0,
                    help="DiscreteShellBending stiffness; <0 = none. Even 0.01 makes the towel a stiff board")
    ap.add_argument("--d-hat", type=float, default=0.001, help="IPC contact distance threshold [m]")
    ap.add_argument("--newton-iters", type=int, default=64)
    ap.add_argument("--substeps", type=int, default=1)
    ap.add_argument("--grip-kp", type=float, default=2000.0)
    ap.add_argument("--yaw", type=float, default=90.0, help="gripper yaw [deg]")
    ap.add_argument("--grasp-in", type=float, default=0.02, help="grasp centre inside the near cloth edge [m]")
    ap.add_argument("--press", type=float, default=-0.002, help="finger bottom target above the table top [m]")
    ap.add_argument("--layout", choices=["flat", "pile"], default="pile")
    ap.add_argument("--depth", type=float, default=0.03, help="pile: finger bottoms this far below the top [m]")
    ap.add_argument("--close-s", type=float, default=1.0)
    ap.add_argument("--carry-dx", type=float, default=-0.15, help="horizontal carry vector x [m]")
    ap.add_argument("--carry-dy", type=float, default=-0.10, help="horizontal carry vector y [m]")
    ap.add_argument("--lift", type=float, default=0.35)
    ap.add_argument("--settle", type=float, default=2.5)
    ap.add_argument("--cam", choices=["wide", "close"], default="wide")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "cloth_ipc_grasp.mp4"))
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    t = Test(args)
    if not args.no_video:
        t.start_video(args.out)
    t0 = time.time()
    t.res = {}
    try:
        t.run()
    except Exception as e:  # keep the partial numbers
        t.res["error"] = repr(e)[:300]
    finally:
        t.finish_video()
    res = t.res
    wall = time.time() - t0
    res["build_s"] = round(t.build_s, 1)
    res["steps"] = t.nstep
    res["sim_steps_per_s"] = round(t.nstep / t.sim_time, 1)
    res["wall_steps_per_s_incl_render"] = round(t.nstep / wall, 1)
    res["gpu_mem_mib_process"] = gpu_mem_mib()
    print(f"=== IPC friction-only grasp: {vars(args)} ===")
    for k, v in res.items():
        print(f"{k}: {round(v, 4) if isinstance(v, float) else v}")
    if not args.no_video:
        print(f"video: {args.out}")


if __name__ == "__main__":
    main()
