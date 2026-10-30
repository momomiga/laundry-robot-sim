"""Feasibility test: grasp a PBD cloth (towel) with the PiPER gripper, lift, carry, release.

Genesis 1.4.4 facts this test relies on (from the installed source):
- Cloth = gs.materials.PBD.Cloth (PBD2DEntity). The mesh is remeshed to edge length = particle_size.
- Rigid -> PBD coupling (LegacyCoupler.kernel_pbd_rigid_collide / _func_pbd_collide_with_rigid_geom)
  only projects particles out of the rigid SDF along the normal. Friction is commented out in the
  source ("we don't consider friction for now"), so a pure pinch has nothing tangential to hold on to.
- PBD2DEntity.fix_particles_to_link(link_idx, particles_idx_local) animates particles with a link
  (kinematic, one-way); release_particle() frees them again.

Modes
  A: contact only (close the fingers on the cloth, no attachment)
  B: A + while closed, the particles near the fingertip centre are fixed to link6

Scenarios
  table : towel flat on a fixed table in front of the robot, grasp near its edge
  basket: towel dropped (crumpled) into the robot's basket, grasp its highest point
  washer: (with --washer) carry to a washer opening next to the robot and drop it in

Usage
  flock /tmp/house_robot-gpu.lock uv run python scripts/cloth_grasp_test.py --mode A
  flock /tmp/house_robot-gpu.lock uv run python scripts/cloth_grasp_test.py --mode B --scenario basket --washer
"""
import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import genesis as gs
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import params as P

DT = 0.01
SUBSTEPS = 10
W, H, FPS = 640, 480, 30
STEPS_PER_FRAME = 3
TCP = 0.15                       # link6 origin -> fingertip centre along link6 z [m]
OPEN, CLOSED = 0.035, 0.0        # joint7 travel (joint8 = -joint7)
PARTICLE = 0.01                  # PBD particle size [m] (overridden by --particle)
TOWEL = 0.30                     # towel edge length [m]
TABLE_TOP = 0.40
TABLE_CENTER = (0.42, 0.17)
TABLE_SIZE = (0.40, 0.50)
ATTACH_R = 0.02                  # mode B: fix particles within this radius of the fingertip centre


def quat_down(yaw_deg=0.0, tilt_deg=0.0):
    """wxyz: gripper pointing down (link6 z -> world -z), rotated by yaw about world z,
    then tilted by tilt_deg about world y."""
    r = Rotation.from_euler("y", tilt_deg, degrees=True) * Rotation.from_euler("z", yaw_deg, degrees=True) \
        * Rotation.from_euler("y", 180, degrees=True)
    q = r.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def quat_toward(az, tilt_deg):
    """wxyz: gripper pointing down, then tilted by tilt_deg so it points outward along azimuth az."""
    r = Rotation.from_euler("z", az) * Rotation.from_euler("y", -tilt_deg, degrees=True) \
        * Rotation.from_euler("y", 180, degrees=True)
    q = r.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def gpu_mem_mib():
    """GPU memory of this process from nvidia-smi [MiB], -1 if unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             check=False).stdout
    except OSError:
        return -1
    for line in out.strip().splitlines():
        pid, mem = [s.strip() for s in line.split(",")]
        if pid.isdigit() and int(pid) == os.getpid():
            return int(mem)
    return -1


class Test:
    def __init__(self, args):
        self.args = args
        global PARTICLE
        PARTICLE = args.particle
        gs.init(backend=gs.gpu, logging_level="warning")
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS),
            pbd_options=gs.options.PBDOptions(particle_size=PARTICLE, lower_bound=(-1.0, -1.0, 0.0),
                                              upper_bound=(2.0, 2.0, 1.6),
                                              max_stretch_solver_iterations=args.stretch_iters),
            show_viewer=False,
            vis_options=gs.options.VisOptions(show_world_frame=False),
        )
        sc = self.scene
        sc.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(needs_coup=False))
        self.robot = sc.add_entity(
            gs.morphs.URDF(file=str(ROOT / "assets/robot/house_robot.urdf"), fixed=True))

        basket_c = P.BASKET_CENTER_XY
        if args.scenario == "table":
            sc.add_entity(gs.morphs.Box(size=(*TABLE_SIZE, TABLE_TOP),
                                        pos=(*TABLE_CENTER, TABLE_TOP / 2), fixed=True),
                          surface=gs.surfaces.Default(color=(0.55, 0.40, 0.28, 1)))
            cloth_pos = (TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP + 0.01)
            cloth_euler = (0, 0, 0)
        else:  # basket: drop the towel in tilted so it folds / crumples
            cloth_pos = (basket_c[0], basket_c[1], P.BASE_Z1 + 0.20)
            cloth_euler = (35, 0, 20)
        if args.washer:
            # washer in front of the arm side, opening top at WASHER_H
            sc.add_entity(gs.morphs.URDF(file=str(ROOT / "assets/washer/washer.urdf"),
                                         pos=self.washer_pos(), euler=(0, 0, P.WASHER_BASE_YAW_DEG), fixed=True))
        self.cloth = sc.add_entity(
            material=gs.materials.PBD.Cloth(rho=args.rho, static_friction=0.6, kinetic_friction=0.5,
                                            bending_compliance=1e-4, air_resistance=1e-3),
            morph=gs.morphs.Mesh(file="meshes/cloth.obj", scale=TOWEL / 0.5, pos=cloth_pos,
                                 euler=cloth_euler),
            surface=gs.surfaces.Default(color=(0.95, 0.55, 0.15, 1.0), vis_mode="visual"),
        )
        if args.washer:
            cam_pos, cam_look = (1.15, -0.35, 1.75), (0.15, 0.35, 0.55)
        else:
            cam_pos, cam_look = (0.95, -0.45, 0.85), (0.33, 0.25, 0.45)
        self.cam = sc.add_camera(res=(W, H), pos=cam_pos, lookat=cam_look, fov=55, GUI=False)
        self.tub_cam = None
        if args.washer:
            # inside the tub, just under the opening's visual disc (which hides the tub interior)
            wp = self.washer_pos()
            cx, cy = P.washer_to_base(0.12, 0.0)    # offsets in the washer frame
            lx, ly = P.washer_to_base(-0.02, 0.0)
            self.tub_cam = sc.add_camera(res=(480, 480), pos=(cx, cy, 0.90), lookat=(lx, ly, 0.35), fov=70, GUI=False)
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
        kp = np.array([8000, 8000, 3000, 8000] + [300] * 6 + [200, 200], dtype=float)
        self.robot.set_dofs_kp(kp, self.dofs)
        self.robot.set_dofs_kv(2 * np.sqrt(kp), self.dofs)
        self.cmd = self.q.copy()
        self.robot.control_dofs_position(self.cmd, self.dofs)
        self.n_part = self.cloth.n_particles
        bx, by = P.BASKET_CENTER_XY
        ix, iy, _ = P.BASKET_INNER
        wp = self.washer_pos()
        self.supports = [  # (top z, (xmin, xmax, ymin, ymax))
            (P.BASE_Z1 + 0.01, (bx - ix / 2, bx + ix / 2, by - iy / 2, by + iy / 2)),  # basket floor
        ]
        if args.scenario == "table":
            tx, ty = TABLE_CENTER
            self.supports.append((TABLE_TOP, (tx - TABLE_SIZE[0] / 2, tx + TABLE_SIZE[0] / 2,
                                              ty - TABLE_SIZE[1] / 2, ty + TABLE_SIZE[1] / 2)))
        if args.washer:
            self.supports.append((0.35, (wp[0] - 0.2, wp[0] + 0.2, wp[1] - 0.2, wp[1] + 0.2)))
        self.attached = None
        self.nstep = 0
        self.sim_time = 0.0
        self.max_speed = 0.0
        self.nan = False
        self.proc = None

    def washer_pos(self):
        # washer (opening centre = washer origin) in the parked-base frame, same as the house scene (params.py);
        # its yaw is P.WASHER_BASE_YAW_DEG (front towards the robot, lid on the far side)
        return (*P.WASHER_BASE_XY, 0.0)

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
        return self.cloth.get_particles_pos().cpu().numpy().reshape(-1, 3)

    def step(self):
        self.robot.control_dofs_position(self.cmd, self.dofs)
        t0 = time.time()
        self.scene.step()
        self.sim_time += time.time() - t0
        self.nstep += 1
        self.q = self.robot.get_dofs_position(self.dofs).cpu().numpy().reshape(-1)
        if self.args.support_mu > 0:
            self.support_friction()
        if self.nstep % 10 == 0:
            v = self.cloth.get_particles_vel().cpu().numpy().reshape(-1, 3)
            if not np.isfinite(v).all():
                self.nan = True
            else:
                self.max_speed = max(self.max_speed, float(np.linalg.norm(v, axis=1).max()))
        if self.proc and self.nstep % STEPS_PER_FRAME == 0:
            self.frame()

    def support_friction(self):
        """Approximate Coulomb-like friction against horizontal supports, which the Genesis
        rigid<->PBD coupling lacks: damp the horizontal velocity of particles resting on a support."""
        x = self.cloth.get_particles_pos().reshape(-1, 3)
        v = self.cloth.get_particles_vel().reshape(-1, 3)
        z = x[:, 2]
        near = z < PARTICLE * 0.75                                      # PBD floor (lower bound z=0)
        for h, (x0, x1, y0, y1) in self.supports:
            near |= (z > h) & (z < h + PARTICLE * 0.75) & (x[:, 0] > x0) & (x[:, 0] < x1) & \
                (x[:, 1] > y0) & (x[:, 1] < y1)
        if self.attached is not None:
            near[self.attached] = False
        if near.any():
            v = v.clone()
            v[near, :2] *= (1.0 - self.args.support_mu)
            self.cloth.set_particles_vel(v)

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

    def ik(self, tcp, quat, lift=None):
        """Damped-LS IK for the fingertip centre; PiPER IK has local minima, so try the current
        command and a few elbow-up seeds and keep the best."""
        base = self.robot.get_link("link1").get_pos().cpu().numpy().reshape(-1)
        az = math.atan2(tcp[1] - base[1], tcp[0] - base[0])   # robot yaw is 0 in this test
        seeds = [self.cmd[4:10].copy()]
        for j1 in (0.0, az):
            seeds += [np.array([j1, 0.5, -0.8, 0, 0.5, 0]), np.array([j1, 1.0, -1.2, 0, 0.6, 0]),
                      np.array([j1, 1.5, -1.0, 0, 0.8, 0]), np.array([j1, 0.3, -0.3, 0, 0.3, 0])]
        best = None
        for s6 in seeds:
            seed = self.cmd.copy()
            seed[4:10] = s6
            if lift is not None:
                seed[3] = lift
            self.robot.set_dofs_position(seed, self.dofs, zero_velocity=False)
            q_full0 = self.robot.get_dofs_position().cpu().numpy().reshape(-1)
            q, err = self.robot.inverse_kinematics(
                link=self.link6, pos=np.array(tcp, float), quat=quat, local_point=(0, 0, TCP),
                dofs_idx_local=self.arm_dofs, init_qpos=q_full0, max_solver_iters=200, return_error=True)
            e = err.cpu().numpy().reshape(-1)
            q = q.cpu().numpy().reshape(-1)[self.arm_dofs]
            # prefer small error, then small joint motion
            score = np.abs(e[:3]).max() + 0.1 * np.abs(e[3:]).max() + 1e-3 * np.abs(q - self.cmd[4:10]).sum()
            if best is None or score < best[0]:
                best = (score, q, e)
        self.robot.set_dofs_position(self.q, self.dofs, zero_velocity=False)
        _, q, e = best
        assert np.abs(e[:3]).max() < 0.01 and np.abs(e[3:]).max() < 0.1, f"IK failed {tcp}: {e}"
        goal = self.cmd.copy()
        goal[4:10] = q
        if lift is not None:
            goal[3] = lift
        return goal

    def ik_any(self, x, y, zs, lift=None, tilts=(0, 20, 40, 60, 80)):
        """First reachable (z, tilt) in order: z descending, then tilt towards the target azimuth."""
        base = self.robot.get_link("link1").get_pos().cpu().numpy().reshape(-1)
        az = math.atan2(y - base[1], x - base[0])
        for z in zs:
            for tilt in tilts:
                try:
                    return self.ik((x, y, z), quat_toward(az, tilt), lift), z, tilt
                except AssertionError:
                    pass
        raise AssertionError(f"unreachable {x, y, zs}")

    def arm_to(self, tcp, quat, dur, track=None, lift=None):
        self.move(self.ik(tcp, quat, lift), dur, track)

    def tcp_world(self):
        p = self.link6.get_pos().cpu().numpy().reshape(-1)
        qw = self.link6.get_quat().cpu().numpy().reshape(-1)
        R = Rotation.from_quat([qw[1], qw[2], qw[3], qw[0]])
        return p + R.apply([0, 0, TCP])

    def finger_bottom_z(self):
        return min(float(f.get_verts().cpu().numpy().reshape(-1, 3)[:, 2].min()) for f in self.fingers)

    def finger_gap(self):
        v = [f.get_verts().cpu().numpy().reshape(-1, 3) for f in self.fingers]
        return v

    def attach(self):
        # particles between the finger pads: near the TCP axis, from just below the finger bottom
        # to 4 cm above it
        tip = self.tcp_world()
        fb = self.finger_bottom_z()
        x = self.cloth_pos()
        d = np.linalg.norm(x[:, :2] - tip[:2], axis=1)
        idx = np.where((d < ATTACH_R) & (x[:, 2] > fb - 0.012) & (x[:, 2] < fb + 0.04))[0]
        if len(idx) == 0:  # fall back to the nearest few particles
            idx = np.argsort(np.linalg.norm(x - np.array([tip[0], tip[1], fb + 0.01]), axis=1))[:5]
        self.cloth.fix_particles_to_link(self.link6.idx, particles_idx_local=idx)
        self.attached = idx
        return idx

    def detach(self):
        if self.attached is not None:
            self.cloth.release_particle(particles_idx_local=self.attached)
            self.attached = None

    # ---------------- scenario ----------------
    def run(self):
        a = self.args
        res = {}
        self.hold(1.5)                           # let the cloth settle
        x0 = self.cloth_pos()
        res["n_particles"] = self.n_part
        res["settled_zmax"] = float(x0[:, 2].max())
        res["settled_zmean"] = float(x0[:, 2].mean())

        down = quat_down(0)
        if a.scenario == "table":
            # grasp 3 cm inside the near (-x) edge, in the middle along y
            edge_x = x0[:, 0].min()
            ymid = np.median(x0[:, 1])
            sel = np.abs(x0[:, 1] - ymid) < 0.02
            gx, gy = edge_x + 0.03, ymid
            gz = np.median(x0[sel & (np.abs(x0[:, 0] - gx) < 0.02), 2])
        else:
            # top of the crumpled towel above its centroid (short hanging length for the washer rim)
            c = x0[:, :2].mean(0)
            sel = np.linalg.norm(x0[:, :2] - c, axis=1) < 0.03
            i = np.where(sel)[0][np.argmax(x0[sel, 2])]
            gx, gy, gz = x0[i]
        res["grasp_point"] = [round(float(gx), 3), round(float(gy), 3), round(float(gz), 3)]

        self.gripper(OPEN, 0.3)
        self.arm_to((gx, gy, gz + 0.15), down, 2.0)
        self.hold(0.3)
        off = self.tcp_world()[2] - self.finger_bottom_z()   # TCP point -> lowest finger vertex
        res["tcp_minus_finger_bottom"] = float(off)
        # finger bottom at `press` relative to the cloth particle centres
        self.arm_to((gx, gy, gz + off + a.press), down, 1.5)
        self.hold(0.3)
        res["finger_bottom_at_grasp"] = self.finger_bottom_z()
        self.gripper(CLOSED, 0.8)
        if a.mode == "B":
            idx = self.attach()
            res["attached_particles"] = len(idx)
        self.hold(0.3)

        x1 = self.cloth_pos()
        tip0 = self.tcp_world()
        # lift 0.2 m
        self.arm_to((gx, gy, gz + 0.20), down, 2.0)
        self.hold(0.5)
        x2 = self.cloth_pos()
        res["lift_dzmax"] = float(x2[:, 2].max() - x1[:, 2].max())
        res["lift_dzmean"] = float(x2[:, 2].mean() - x1[:, 2].mean())
        res["tcp_lift_dz"] = float(self.tcp_world()[2] - tip0[2])
        # how far is the cloth's top point from the fingertip after lifting
        res["tip_to_cloth_top_after_lift"] = float(self.tcp_world()[2] - x2[:, 2].max())

        # carry horizontally
        track = []
        if a.washer:
            wp = self.washer_pos()
            top = P.WASHER_H + self.args.drop_dz  # fingertip height over the rim
            # raise as high as reachable above the basket first so the hanging towel clears the rim
            goal, zz, tilt = self.ik_any(gx, gy, (top + 0.15, top + 0.10, top + 0.05, top, top - 0.1, top - 0.2),
                                         lift=0.5)
            res["carry_raise_tcp_z"], res["carry_raise_tilt"] = zz, tilt
            self.move(goal, 2.5, track)
            # straight-down is out of reach over the rim: tilt the gripper towards the washer as needed
            goal, _, tilt = self.ik_any(wp[0], wp[1], (top,), lift=0.5)
            res["release_tilt_deg"] = tilt
            self.move(goal, 3.0, track)
        else:
            self.arm_to((gx, gy + 0.20, gz + 0.20), down, 2.0, track)
        self.hold(1.0, track)
        x3 = self.cloth_pos()
        tip = self.tcp_world()
        res["carry_zmax_min"] = float(min(track))
        res["carry_zmax_end"] = float(x3[:, 2].max())
        res["carry_zmean_end"] = float(x3[:, 2].mean())
        res["carry_xy_dist_cloth_to_tip"] = float(np.linalg.norm(x3.mean(0)[:2] - tip[:2]))
        support = TABLE_TOP if a.scenario == "table" else P.BASE_Z1
        res["held_after_carry"] = bool(x3[:, 2].max() > support + 0.10 and
                                       np.linalg.norm(x3[np.argmax(x3[:, 2])][:2] - tip[:2]) < 0.05)

        # release
        self.detach()
        self.gripper(OPEN, 0.6)
        self.hold(2.0)
        x4 = self.cloth_pos()
        res["released_zmax"] = float(x4[:, 2].max())
        res["released_zmean"] = float(x4[:, 2].mean())
        if a.washer:
            wp = self.washer_pos()
            c = x4.mean(0)
            inside = (np.abs(x4[:, 0] - wp[0]) < 0.2) & (np.abs(x4[:, 1] - wp[1]) < 0.2) & \
                (x4[:, 2] < P.WASHER_H)
            res["frac_particles_in_tub"] = float(inside.mean())
            res["cloth_centroid"] = [round(float(v), 3) for v in c]
            if self.tub_cam is not None:
                img = np.asarray(self.tub_cam.render()[0]).astype(np.uint8)
                out = ROOT / "outputs" / "cloth_grasp_tub_view.png"
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", f"{img.shape[1]}x{img.shape[0]}", "-i", "-", str(out)],
                               input=img.tobytes(), check=True)
                res["tub_view_png"] = str(out)
        res["nan"] = self.nan
        res["max_particle_speed"] = self.max_speed
        res["min_particle_z"] = float(x4[:, 2].min())
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B"], default="A")
    ap.add_argument("--scenario", choices=["table", "basket"], default="table")
    ap.add_argument("--washer", action="store_true")
    ap.add_argument("--lift", type=float, default=0.25)
    ap.add_argument("--press", type=float, default=0.0, help="finger bottom height relative to cloth particle centres")
    ap.add_argument("--rho", type=float, default=0.3, help="cloth areal density [kg/m^2]")
    ap.add_argument("--stretch-iters", type=int, default=8)
    ap.add_argument("--support-mu", type=float, default=0.0,
                    help="per-step horizontal velocity damping (0-1) of particles resting on supports")
    ap.add_argument("--drop-dz", type=float, default=0.08, help="washer: TCP height above the rim at release")
    ap.add_argument("--particle", type=float, default=0.01, help="PBD particle size = cloth edge length [m]")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    t = Test(args)
    out = args.out or str(ROOT / "outputs" / f"cloth_grasp_{args.mode}.mp4")
    if not args.no_video:
        t.start_video(out)
    t0 = time.time()
    try:
        res = t.run()
    finally:
        t.finish_video()
    wall = time.time() - t0
    res["build_s"] = round(t.build_s, 1)
    res["steps"] = t.nstep
    res["sim_steps_per_s"] = round(t.nstep / t.sim_time, 1)
    res["wall_steps_per_s_incl_render"] = round(t.nstep / wall, 1)
    res["gpu_mem_mib_process"] = gpu_mem_mib()
    print(f"=== mode {args.mode} scenario {args.scenario} washer={args.washer} ===")
    for k, v in res.items():
        print(f"{k}: {round(v, 4) if isinstance(v, float) else v}")
    if not args.no_video:
        print(f"video: {out}")


if __name__ == "__main__":
    main()
