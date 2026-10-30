"""Scripted demo video: robot drives from the main room through the corridor to
the washer, then moves 3 substitute laundry items from its basket into the tub.

Output: outputs/demo.mp4 (1280x720, 30 fps). Prints the number of items that
ended up inside the washer tub and "OK" if >= 1. The AI part is not included
(everything is scripted: waypoint driving + IK arm motions).

Grasping: the item is rigidly attached to link6 with Genesis' weld equality
constraint (rigid_solver.add_weld_constraint) only while "held"; fingers still
close/open visually. (Friction grasp is not used.)
"""
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

import genesis as gs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import params as P  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import robot_sensors  # noqa: E402

DT = 0.01
W, H, FPS = 1280, 720, 30
# Video is 1x real time: a frame is captured whenever sim time reaches the next 1/FPS tick
# (every 3 or 4 steps at 100 Hz), so video length == simulated time.
SPEED = 0.30                   # drive speed [m/s]
YAW = P.PARK_POSE[2]
START = (1.35, 2.0)
PARK = P.PARK_POSE[:2]            # robot base position at the washer (params.py, shared with the IPC test)
LIFT_H = 0.35
ITEM_SIZE = (0.10, 0.05, 0.04)  # small enough for the 7 cm PiPER gripper (opens along world y)
TCP = 0.15                     # link6 origin -> fingertip centre along link6 z [m]
GRIP_DOWN = (0.0, 1.0, 0.0, 0.0)  # wxyz: link6 z -> world -z, link6 y -> world -y
OPEN, CLOSED = 0.035, 0.012    # finger joint7 travel (joint8 = -joint7)


def rot(yaw, v):
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1])


class Demo:
    def __init__(self):
        gs.init(backend=gs.gpu, logging_level="warning")
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=DT),
            show_viewer=False,
            vis_options=gs.options.VisOptions(show_world_frame=False),
        )
        sc = self.scene
        sc.add_entity(gs.morphs.URDF(file=str(ROOT / "assets/room/room.urdf"), fixed=True))
        sc.add_entity(gs.morphs.URDF(file=str(ROOT / "assets/washer/washer.urdf"),
                                     pos=P.WASHER_POS, fixed=True))
        self.robot = sc.add_entity(
            gs.morphs.URDF(file=str(ROOT / "assets/robot/house_robot.urdf"), fixed=True))

        # items in the basket (placed for the START pose; basket moves with the robot)
        bc = rot(YAW, P.BASKET_CENTER_XY)
        floor_z = P.BASE_Z1 + 0.01
        self.items = []
        for i in range(3):
            e = sc.add_entity(
                gs.morphs.Box(size=ITEM_SIZE,
                              pos=(START[0] + bc[0], START[1] + bc[1] + (i - 1) * 0.10,
                                   floor_z + ITEM_SIZE[2] / 2 + 0.002)),
                surface=gs.surfaces.Default(color=[(0.9, 0.3, 0.3, 1), (0.3, 0.7, 0.3, 1),
                                                   (0.95, 0.8, 0.2, 1)][i]),
            )
            self.items.append(e)

        self.cam = sc.add_camera(res=(W, H), pos=(2, 0, 3), lookat=(1, 2, 0.3), fov=50, GUI=False)
        self.cam_top = sc.add_camera(res=(300, 420), pos=(1.0, 5.0, 11.0), lookat=(1.0, 5.0, 0.0),
                                     up=(0, 1, 0), fov=26, GUI=False)
        self.cam_basket = robot_sensors.add_basket_camera(sc, self.robot)
        sc.build()

        names = ["base_x_joint", "base_y_joint", "base_yaw_joint", "lift_joint"] + \
            [f"joint{i}" for i in range(1, 9)]
        self.dofs = [self.robot.get_joint(n).dofs_idx_local[0] for n in names]
        self.arm_dofs = self.dofs[4:10]
        self.link6 = self.robot.get_link("link6")
        self.q = np.zeros(12)
        self.q[:3] = [START[0], START[1], YAW]
        self.robot.set_dofs_position(self.q, self.dofs)
        kp = np.array([8000, 8000, 3000, 8000] + [300] * 6 + [200, 200], dtype=float)
        self.robot.set_dofs_kp(kp, self.dofs)
        self.robot.set_dofs_kv(2 * np.sqrt(kp), self.dofs)
        self.robot.control_dofs_position(self.q, self.dofs)

        self.proc = None
        self.nstep = 0
        self.cam_p = None
        self.cam_l = None
        self.held = None

    # ---------------- video ----------------
    def start_video(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "20", path], stdin=subprocess.PIPE)
        self.nframes = 0

    def frame(self):
        x, y = self.q[0], self.q[1]
        # high oblique chase camera (above the 2.4 m walls so it never looks through them)
        tgt_p = np.array([min(x + 0.3, 1.25), y - 0.9, 2.55])
        tgt_l = np.array([x - 0.1, y + 0.05, 0.55])
        if self.cam_p is None:
            self.cam_p, self.cam_l = tgt_p, tgt_l
        self.cam_p = self.cam_p + 0.1 * (tgt_p - self.cam_p)
        self.cam_l = self.cam_l + 0.1 * (tgt_l - self.cam_l)
        self.cam.set_pose(pos=tuple(self.cam_p), lookat=tuple(self.cam_l))
        img = np.asarray(self.cam.render()[0]).astype(np.uint8).copy()
        top = np.asarray(self.cam_top.render()[0]).astype(np.uint8)
        h, w = top.shape[:2]
        img[10:10 + h, W - w - 10:W - 10] = top
        # basket camera wipe (downscaled 640x480 -> 240x180), below the top view
        bk = robot_sensors.get_basket_image(self.cam_basket)[::2, ::2][:, :, :3]
        bk = np.asarray(Image.fromarray(bk).resize((240, 180)))
        img[h + 20:h + 20 + 180, W - 250:W - 10] = bk
        self.proc.stdin.write(img.tobytes())
        self.nframes += 1

    def finish_video(self):
        self.proc.stdin.close()
        self.proc.wait()

    # ---------------- sim ----------------
    def step(self, q_cmd=None):
        if q_cmd is not None:
            self.robot.control_dofs_position(q_cmd, self.dofs)
        self.scene.step()
        self.nstep += 1
        self.q = self.robot.get_dofs_position(self.dofs).cpu().numpy().reshape(-1)
        if self.nstep * DT >= self.nframes / FPS:
            self.frame()

    def move(self, q_goal, dur):
        """Joint-space smooth move of all dofs to q_goal."""
        q0 = self.cmd.copy()
        n = max(1, int(dur / DT))
        for k in range(1, n + 1):
            a = k / n
            a = a * a * (3 - 2 * a)
            self.cmd = q0 + a * (q_goal - q0)
            self.step(self.cmd)

    def hold(self, dur):
        for _ in range(int(dur / DT)):
            self.step(self.cmd)

    def drive(self, pts):
        """Follow world waypoints at SPEED (yaw fixed), with smooth start/stop."""
        pts = [np.array(p, float) for p in pts]
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(1, int(np.linalg.norm(b - a) / SPEED / DT))
            for k in range(1, n + 1):
                f = k / n
                self.cmd[:2] = a + f * (b - a)
                self.step(self.cmd)

    def gripper(self, v, dur=0.8):
        g = self.cmd.copy()
        g[10], g[11] = v, -v
        self.move(g, dur)

    # ---------------- arm ----------------
    def arm_to(self, tcp_world, tilt_deg=0.0, lift=None, dur=2.0):
        """IK for the fingertip centre (link6 local (0,0,TCP)) with the gripper pointing down,
        tilted by tilt_deg about world y (90 = horizontal, towards -x); joint-space move."""
        lift = self.cmd[3] if lift is None else lift
        r = Rotation.from_euler("y", 180 + tilt_deg, degrees=True).as_quat()
        quat = np.array([r[3], r[0], r[1], r[2]])
        # solve IK with the target lift and the commanded configuration as the seed
        seed = self.cmd.copy()
        seed[3] = lift
        self.robot.set_dofs_position(seed, self.dofs, zero_velocity=False)
        q_full0 = self.robot.get_dofs_position().cpu().numpy().reshape(-1)
        q, err = self.robot.inverse_kinematics(
            link=self.link6, pos=np.array(tcp_world, float), quat=quat, local_point=(0, 0, TCP),
            dofs_idx_local=self.arm_dofs, init_qpos=q_full0, max_solver_iters=50, return_error=True)
        e = err.cpu().numpy().reshape(-1)
        assert np.abs(e[:3]).max() < 0.01 and np.abs(e[3:]).max() < 0.1, \
            f"IK failed for {tcp_world} tilt {tilt_deg}: {e}"
        q = q.cpu().numpy().reshape(-1)
        goal = self.cmd.copy()
        goal[3] = lift
        goal[4:10] = q[self.arm_dofs]
        self.robot.set_dofs_position(self.q, self.dofs, zero_velocity=False)  # undo the probe
        self.move(goal, dur)

    def attach(self, item):
        link_a = item.base_link.idx
        link_b = self.link6.idx
        self.scene.sim.rigid_solver.add_weld_constraint(link_a, link_b)
        self.held = (link_a, link_b)

    def detach(self):
        self.scene.sim.rigid_solver.delete_weld_constraint(*self.held)
        self.held = None

    def in_tub(self, item):
        p = item.get_pos().cpu().numpy().reshape(-1)
        wx, wy = P.WASHER_POS[0], P.WASHER_POS[1]
        return abs(p[0] - wx) < 0.2 and abs(p[1] - wy) < 0.2 and p[2] < P.WASHER_H - 0.1

    # ---------------- scenario ----------------
    def run(self):
        self.cmd = self.q.copy()
        self.hold(0.5)
        self.gripper(OPEN, 0.1)

        # 1. drive: main room -> opening centre -> corridor -> washer side
        cx = PARK[0]  # corridor line, centred for the footprint incl. the stowed arm (~5 cm clearances)
        self.drive([START, (cx, 3.2), (cx, 4.6), (cx, 5.6), (cx, 6.6), PARK])
        self.hold(0.8)

        # 2. raise lift, move arm over items
        g = self.cmd.copy()
        g[3] = LIFT_H
        self.move(g, 3.0)
        # stow-ish pose first so the arm unfolds towards the basket
        base = np.array([self.q[0], self.q[1]])
        for i, item in enumerate(self.items):
            ip = item.get_pos().cpu().numpy().reshape(-1)
            above = (ip[0], ip[1], ip[2] + 0.18)
            grasp = (ip[0], ip[1], ip[2] + 0.005)
            self.arm_to(above, 0, LIFT_H, 2.5)
            self.arm_to(grasp, 0, LIFT_H, 1.5)
            self.gripper(CLOSED, 0.8)
            self.attach(item)
            self.hold(0.2)
            self.arm_to((ip[0], ip[1], 0.75), 0, LIFT_H, 1.8)
            self.arm_to((P.WASHER_POS[0], P.WASHER_POS[1], 1.15), 80, 0.5, 3.5)
            self.detach()
            self.gripper(OPEN, 0.6)
            self.hold(1.2)
            print(f"item{i} in tub: {self.in_tub(item)}  pos={np.round(item.get_pos().cpu().numpy(), 3)}")
        return sum(self.in_tub(it) for it in self.items)


def main():
    d = Demo()
    out = str(ROOT / "outputs" / "demo.mp4")
    d.start_video(out)
    try:
        n = d.run()
    finally:
        d.finish_video()
    print(f"frames: {d.nframes}  video length {d.nframes / FPS:.2f}s  sim time {d.nstep * DT:.2f}s (1x)")
    print(f"items in tub: {n}/3")
    if n >= 1:
        print("OK")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
