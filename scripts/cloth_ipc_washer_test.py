"""Feasibility test: friction-only basket -> washer transfer of an IPC cloth (FEM.Cloth + IPCCoupler).

The robot is parked (fixed base, yaw 0) beside a vertical washer, same layout as
`cloth_grasp_test.py --scenario basket --washer` (robot at the origin, washer opening centre at WASHER_XY).
A 0.30 m towel is dropped standing into the robot's basket so it collapses into a crumpled heap (shape varies with
--seed). The PiPER pinches the heap top near its centroid (no particle attachment, no weld: IPC contact + friction
only), lifts it high, carries it over the opening, releases it and the test checks that the towel ends up in the tub.
Runs in the separate IPC environment (.venv-ipc, see requirements-ipc.txt).

How the basket and the tub take part in IPC (Genesis 1.4.4, genesis/engine/couplers/ipc_coupler/coupler.py):
- _add_rigid_geoms_to_ipc() adds rigid geoms per link (fixed-joint children merged into their parent via
  find_target_link_for_fixed_merge). For 'two_way_soft_constraint' only links listed in material.coup_links are
  added. The basket boxes are collision geoms of the robot's `base_link` (tools/gen_robot.py builds chassis, column
  and basket into one link), so coup_links = (base_link, link6, link7, link8) puts the basket (and the chassis /
  mast) into IPC as a soft-constrained affine body that follows the Genesis pose. The arm links 1-5 stay out of IPC
  (the cloth passes through them).
- The washer is a separate fixed URDF without DOFs: coup_type 'ipc_only' (also default_coup_type() for it), a
  static affine body (is_fixed = 1). Its lid (fixed joint) is merged into washer_body. Only collision geoms are
  used, so the visual ring around the opening does not matter; the tub / opening is the square hole WASHER_TUB_SIDE.
- enable_rigid_rigid_contact=False: link6/7/8 overlap at rest, and the gripper must not collide with the basket
  inside IPC (the Genesis rigid solver handles robot-world contacts).

Findings so far (2026-10-19; the friction pinch in the basket mostly FAILS, see the knowledge base IPC notes)
- The towel keeps slumping for > 3 s after the drop (no bending stiffness); a grasp point picked at 3 s was stale by
  up to 9 cm, hence the settle-until-still loop (<= 8 s) and the re-sensing at the hover pose.
- Descending with the fingers fully open pushes most of the cloth out of the gap (vertices within 3.5 cm of the
  grasp axis in the finger band: ~80 at hover -> 0-16 at depth). Closing while descending (--scoop, default)
  keeps more of it, but the cloth still slips out on lifting in most trials.
- The physics after a successful pinch (carry / release / tub) has not been reached in a full run yet.

Video timing (changed 2026-10-27): a frame is captured whenever the simulated time reaches the next 1/FPS tick
(was: every 3 steps of dt 0.01 encoded at 30 fps = 0.9x playback), so the video length equals the sim time.
Only the video changes, not the simulation.

Subclass hooks (defaults keep the command line behaviour): `base_xy0` (class attribute, robot base xy the basket /
towel spawn / in_basket follow, default origin; the yaw is not used), `add_extra_entities(scene)` (before
scene.build(), e.g. a visual-only house), `make_cameras(scene)` (sets self.cam, self.tub_cam, self.obs_cam),
`grasp_center(x, k)` (None or an xy / (xy, radius) overriding the heap centroid as the grasp search centre).
`--save-obs PATH` saves an npz of the settled cloth and the basket camera (see `save_obs`).

Collision-aware arm motion (--planner, opt-in): every arm / lift / gripper motion (standby, pre-open, hover,
descent, close, lift, lift joint, raise, tilted carry waypoints, release, back to standby, and the retry moves after
an empty pinch or a failed lift) is a joint path planned and densely validated by scripts/arm_planner.py
(TransferSeq on an ArmPlanner scene, margins 10 mm arm / 2 mm gripper-basket), retimed (smoothstep) to the usual
durations, longer if a joint would exceed --vmax. The planner scene is a plain scene (never stepped) in a
CPU-backend child process (arm_planner.PlannerClient; --planner-proc inproc: a second scene in this process on
the IPC backend, 10-50x slower planning), never planned on the IPC scene (that would teleport the robot, and the
ipc_only washer is invisible to Genesis collision). Plans start from the last
*commanded* configuration; the run starts at the planned standby pose instead of q = 0 (inside the mast). The IPC
self.ik is not used. Requires --no-scoop, --press-dz 0, --tilt 0 and --carry-mode gentle. The result JSON gets
plan_segments (method, planning time, commanded-path clearance, start time, duration) and measured_segments
(penetration / clearance of the measured q, every --qlog-every steps, checked in the planner scene). CPU check of
the same planning code: `.venv/bin/python scripts/arm_planner_test.py --grasp-set`.
Constant grip (--grip-const, needs --grip-fmax): the closing target is 2 * fmax / kp beyond CLOSED (Genesis does
not clamp position targets to the joint limits; PD force = kp * (target - q) - kv * qdot, clipped to the force
range), so the PD saturates at --grip-fmax for any pinch thinner than ~fmax / kp per finger instead of falling to
kp x remaining gap.

Usage
  flock /tmp/house_robot-gpu.lock .venv-ipc/bin/python scripts/cloth_ipc_washer_test.py --seed 0
  (outputs/cloth_ipc_washer.mp4 for seed 0, outputs/cloth_ipc_washer_s<seed>.mp4 otherwise,
   numbers also in outputs/cloth_ipc_washer_s<seed>.json)
  planned motion + constant 40 N grip (best args):
  flock /tmp/house_robot-gpu.lock .venv-ipc/bin/python scripts/cloth_ipc_washer_test.py --seed 0 --pre-open 0.015 \
    --no-scoop --depth 0.03 --eps-v 1e-3 --grip-kp 8000 --grip-fmax 40 --mu 1.2 --miss-check --newton-tol 1e-3 \
    --attempts 5 --skip-carry-on-fail --carry-mode gentle --carry-s 9 --n-wp 6 --cs-trans 30 --cs-rot 30 \
    --planner --grip-const
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import genesis as gs
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))
import params as P  # noqa: E402
import robot_sensors  # noqa: E402
from cloth_grasp_test import gpu_mem_mib, quat_down, quat_toward  # noqa: E402
from cloth_ipc_grasp_test import grid_cloth  # noqa: E402
from arm_planner import PlannerClient, drop_keepout, retime, seg_duration  # noqa: E402
from arm_control import MODES as FF_MODES, FeedForward  # noqa: E402

DT = 0.01
W, H, FPS = 640, 480, 30
TCP = 0.15                       # link6 origin -> fingertip centre along link6 z [m]
OPEN, CLOSED = 0.035, 0.0        # joint7 travel (joint8 = -joint7)
TOWEL = 0.30
WASHER_XY = P.WASHER_BASE_XY     # washer origin = opening centre in the parked-base frame (= house layout, params.py)
WASHER_YAW_DEG = P.WASHER_BASE_YAW_DEG   # washer yaw: front towards the robot, lid on the far side (base +y)
WASHER_Z = 0.003                 # IPC sanity check rejects a body touching the ground plane (distance <= 0)
BASKET_FLOOR = P.BASE_Z1 + P.BASKET_WALL
BASKET_RIM = BASKET_FLOOR + P.BASKET_INNER[2]
TUB_HALF = P.WASHER_TUB_SIDE / 2   # the tub is a square prism of inner side WASHER_TUB_SIDE
LIFT_GRASP, LIFT_CARRY = 0.25, 0.5


class PlanningError(RuntimeError):
    pass


def _basket_xy(obj):
    """Basket centre in the world frame: the parked-base value shifted by the subclass's `base_xy0`."""
    ox, oy = getattr(obj, "base_xy0", (0.0, 0.0))
    return P.BASKET_CENTER_XY[0] + ox, P.BASKET_CENTER_XY[1] + oy


class Test:
    base_xy0 = (0.0, 0.0)   # robot base xy at the start (basket centre / towel spawn / in_basket follow it)

    def __init__(self, args):
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        gs.init(backend=gs.gpu, logging_level="warning", seed=args.seed)
        self.T = None
        self.drop_env = self.drop_envelope()   # drop envelope of the garments + gripper keep-out distance
        if args.planner:  # plain planning scene (collision shapes only), same layout, built before the IPC scene
            # subprocess (default): CPU-backend child process, started now, awaited at the first planning call
            # (planning on this process' gs.gpu backend is 10-50x slower); inproc: a second scene in this process
            self.T = PlannerClient(
                dict(washer_pos=(*WASHER_XY, WASHER_Z), washer_yaw_deg=WASHER_YAW_DEG,
                     margin=args.planner_margin, with_room=args.planner_room, seed=args.seed),
                dict(lift_grasp=LIFT_GRASP, lift_carry=LIFT_CARRY, open_=OPEN, lift1_z=args.lift1_z,
                     carry_z=args.carry_z, n_wp=args.n_wp, tilt_step=args.tilt_step,
                     wait_lift=LIFT_CARRY, keepout=self.drop_env), mode=args.planner_proc)
            self.AP = self.T.AP
            self.planner_build_s = self.T.build_s
        self.mesh_path = os.path.join(tempfile.mkdtemp(prefix="cloth_ipc_w_"), "cloth.obj")
        self.edges, self.rest_len = grid_cloth(self.mesh_path, args.n, TOWEL)
        semi_implicit = {"on": True, "off": False}.get(args.semi_implicit)   # None = libuipc default
        scene_kw = dict(
            sim_options=gs.options.SimOptions(dt=DT, substeps=1),
            coupler_options=gs.options.IPCCouplerOptions(
                contact_d_hat=args.d_hat,
                contact_friction_enable=True,
                enable_rigid_rigid_contact=False,
                newton_max_iterations=args.newton_iters,
                **{k: v for k, v in dict(contact_eps_velocity=args.eps_v, newton_tolerance=args.newton_tol,
                                         constraint_strength_translation=args.cs_trans,
                                         constraint_strength_rotation=args.cs_rot,
                                         contact_resistance=args.contact_resistance,
                                         newton_semi_implicit_enable=semi_implicit,
                                         newton_semi_implicit_beta_tolerance=args.semi_beta).items()
                    if v is not None},
            ),
            show_viewer=False,
            vis_options=gs.options.VisOptions(
                show_world_frame=False, ambient_light=(0.40, 0.40, 0.40),
                lights=[dict(type="directional", dir=(-0.4, 0.3, -1.0), color=(1.0, 1.0, 1.0), intensity=5.0)]),
        )
        scene_kw.update(self.scene_options())
        self.scene = gs.Scene(**scene_kw)
        sc = self.scene
        sc.add_entity(gs.morphs.Plane(**self.plane_options()),
                      material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=0.5))
        # the robot is built (and enters IPC) at base_xy0 with its base joints at 0: dof value = world - q_base_off
        self.q_base_off = np.array([*self.base_xy0, 0.0])
        self.robot = sc.add_entity(
            gs.morphs.URDF(file=str(ROOT / "assets/robot/house_robot.urdf"), fixed=True,
                           pos=(*self.base_xy0, 0.0)),
            material=gs.materials.Rigid(coup_type="two_way_soft_constraint",
                                        coup_links=("base_link", "link6", "link7", "link8"),
                                        coup_friction=args.mu))
        self.washer = sc.add_entity(
            gs.morphs.URDF(file=str(ROOT / "assets/washer/washer.urdf"), pos=(*WASHER_XY, WASHER_Z),
                           euler=(0, 0, WASHER_YAW_DEG), fixed=True),
            material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=args.washer_mu))
        self.add_cloth(sc)
        self.make_cameras(sc)
        self.add_extra_entities(sc)
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
        self.q[3] = LIFT_GRASP
        if self.T is not None:  # start at the planned standby fold (the PiPER zero pose lies inside the mast)
            if self.T.start() is None:
                raise RuntimeError("planner: no collision-free standby pose")
            self.planner_build_s = self.T.build_s
            self.q = self.T.standby.copy()
        self.q[:2] = self.base_xy0   # (default: the origin = standby base position of the planner)
        self.robot.set_dofs_position(self.q - self._dq_off(), self.dofs)
        kp = np.array([8000, 8000, 3000, 8000] + [300] * 6 + [args.grip_kp] * 2, dtype=float)
        kv = 2 * np.sqrt(kp)
        # Genesis integrates the PD damping implicitly (kv * dt is added to the joint-space mass matrix). Doubling the
        # finger or arm kv did not remove the 50 Hz pinch chatter (the IPC coupling force loop); --cs-trans/--cs-rot
        # 30 did (knowledge base: debugging.md, gripper vibration)
        kv[4:10] *= args.arm_kv_scale
        kv[10:12] *= args.grip_kv_scale
        self.robot.set_dofs_kp(kp, self.dofs)
        self.robot.set_dofs_kv(kv, self.dofs)
        # URDF effort of joint7/8 = 10 N caps the finger force; the real PiPER gripper is rated 40 N (max 50 N, PiPER
        # manual values in the reseller spec sheet), so --grip-fmax 40 stays within the hardware rating
        if args.grip_fmax:
            self.robot.set_dofs_force_range(np.full(2, -args.grip_fmax), np.full(2, args.grip_fmax), self.dofs[10:12])
        self.cmd = self.q.copy()
        self.ff = FeedForward(self.robot, kp, kv, args.dt, args.ff)   # --ff: feed-forward on the PD target
        self.robot.control_dofs_position(self.cmd - self._dq_off(), self.dofs)
        # closing target: CLOSED (force = kp x remaining gap) or, --grip-const, far enough beyond it that the PD
        # saturates at --grip-fmax (Genesis does not clamp position targets to the joint limits)
        self.close_target = CLOSED - 2.0 * args.grip_fmax / args.grip_kp if args.grip_const else CLOSED
        self.grip_hold = None     # planner mode: commanded opening that overrides the planned one (after a grip move)
        self.plan_rows = [dict(name="start (standby)", method="find_standby", ok=True, t0_s=0.0,
                               plan_s=round(self.T.standby_s, 2))] if self.T is not None else []
        self.seg_i = 0
        self.qlog = [] if self.T is not None else None   # (executed segment index, measured q)
        c = [v.mean(0) for v in self.finger_verts()]
        R6 = self.link6_rot()
        self.close_axis_l6 = R6.inv().apply((c[1] - c[0]) / np.linalg.norm(c[1] - c[0]))
        # static collision boxes the cloth must not enter (penetration check)
        self.solid_geoms = list(self.washer.geoms) + list(self.robot.get_link("base_link").geoms)
        self.nstep = 0
        self.nframes = 0
        self.sim_time = 0.0
        self.max_speed = 0.0
        self.max_stretch = 0.0
        self.nan = False
        self.prev_x = None
        self.grasp_yaw = args.yaw
        self.proc = None
        self.view = self.cam
        self.carry_track = None   # list of (tip-to-nearest-vertex, zmax) samples while carrying

    # ---------------- subclass hooks ----------------
    resense = True   # False: no re-sense of the heap at the hover pose (x0 is not a vertex array of the cloth entity)

    def drop_envelope(self):
        """(lo, hi, dist) keep-out box of the waiting gripper (arm_planner.drop_keepout): the drop envelope of the
        cloth. A subclass with a taller drop (several stacked garments) overrides it."""
        return drop_keepout(self.args.drop_h, TOWEL)

    def scene_options(self):
        """Extra / overriding gs.Scene keyword arguments (e.g. vis_options with the house lights)."""
        return {}

    def plane_options(self):
        """Keyword arguments of the ground plane morph (e.g. visualization=False under a visual house floor)."""
        return {}

    def after_settle(self, res):
        """Called after the first settle (and its `--stop-after settle` exit), before the planned lowering to the
        grasp lift. A subclass may move the robot / re-settle here (then set self.base_xy0 and self.qlog). A true
        return value ends the run with the numbers so far."""
        return False

    def _dq_off(self):
        """12-vector subtracted from the commanded configuration to get the dof values (zero by default; the base
        joints are the offset from the build pose when the robot is built away from the origin)."""
        off = np.zeros(12)
        off[:3] = self.q_base_off
        return off

    def add_extra_entities(self, scene):
        """Called right before scene.build(): a subclass adds visual-only entities (e.g. the house) here."""

    def make_cameras(self, scene):
        """Creates self.cam (video view), self.tub_cam and, with --save-obs, self.obs_cam (basket camera)."""
        args = self.args
        sc = scene
        wx, wy = WASHER_XY
        if args.cam == "close":
            bx, by = _basket_xy(self)
            self.cam = sc.add_camera(res=(W, H), pos=(bx + 0.42, by - 0.12, 0.85), lookat=(bx, by, 0.27), fov=40,
                                     GUI=False)
        else:
            self.cam = sc.add_camera(res=(W, H), pos=(1.15, -0.45, 1.75), lookat=(0.12, 0.30, 0.60), fov=55,
                                     GUI=False)
        # looks steeply down into the opening (the bottom of the 0.60 m deep tub is in view), used after the arm has
        # retracted; the former front-right view (wx + 0.45, wy - 0.25, 1.65) only showed the tub walls.
        # Offsets are in the base frame (the camera stays on the robot's +x side, away from the lid on base +y)
        self.tub_cam = sc.add_camera(res=(W, H), pos=(wx + 0.20, wy - 0.12, 1.85), lookat=(wx - 0.02, wy + 0.01, 0.40),
                                     fov=40, GUI=False)
        self.obs_cam = robot_sensors.add_basket_camera(sc, self.robot) if args.save_obs else None

    def add_cloth(self, scene):
        """Creates self.cloth (the towel entity), self.drop and consumes the seed's RNG draws. A subclass may replace it
        (then also set self.edges / self.rest_len for the stretch metric)."""
        args = self.args
        sc = scene
        # towel released standing (plane ~ xz, along the long basket side), leaning and yawed by the seed
        # dropped from `drop_h` above the floor: the lower edge lands first while the rest still falls, so the
        # towel buckles into a heap instead of tipping over flat
        yaw = self.rng.uniform(-20, 20)
        lean = self.rng.choice([-1, 1]) * self.rng.uniform(0, args.lean_max)
        dx, dy = self.rng.uniform(-0.01, 0.01), self.rng.uniform(-0.03, 0.03)
        bx, by = _basket_xy(self)
        cpos = (bx + dx, by + dy, BASKET_FLOOR + TOWEL / 2 * math.cos(math.radians(lean)) + args.drop_h)
        self.drop = dict(yaw=round(float(yaw), 1), lean=round(float(lean), 1), dx=round(float(dx), 3),
                         dy=round(float(dy), 3))
        self.cloth = sc.add_entity(
            morph=gs.morphs.Mesh(file=self.mesh_path, pos=cpos, euler=(90 + lean, 0, yaw)),
            material=gs.materials.FEM.Cloth(E=args.E, nu=0.3, rho=args.rho, thickness=args.thickness,
                                            bending_stiffness=args.bend, friction_mu=args.mu),
            surface=gs.surfaces.Default(color=(0.95, 0.55, 0.15, 1.0)),
        )

    def grasp_center(self, x, k):
        """Externally supplied grasp search centre for attempt k (0-based) on the heap x: None (default: the heap
        centroid), an xy point, or (xy, radius). A subclass can return a VLM point here."""
        return None

    def save_obs(self, path, x):
        """npz of the settled cloth vertices and the basket camera: rgb, depth and the world-frame point cloud (+
        valid mask) when the Genesis camera offers them (unverified whether the FEM cloth is in the depth buffer:
        check `depth` / `pc_mask` of a real file), `intrinsics` (3x3), `extrinsics` (4x4 world -> camera, OpenGL
        axes flipped as in Genesis), `cam_transform` (4x4 camera -> world). Returns a dict of what was saved."""
        cam = self.obs_cam
        out = dict(cloth_verts=np.asarray(x, np.float32), base_xy0=np.asarray(self.base_xy0, float))
        info = {}
        if cam is not None:
            cam.move_to_attach()
            out["intrinsics"] = np.asarray(cam.intrinsics, float)
            out["extrinsics"] = np.asarray(cam.extrinsics, float)
            out["cam_transform"] = np.asarray(cam.transform, float)
            try:
                r = cam.render(rgb=True, depth=True)
                out["rgb"] = np.asarray(r[0]).astype(np.uint8)
                if r[1] is not None:
                    out["depth"] = np.asarray(r[1], np.float32)
            except Exception as e:  # keep what we have
                info["render_error"] = repr(e)[:200]
            try:
                pc, mask = cam.render_pointcloud(world_frame=True)
                out["pc_world"] = np.asarray(pc, np.float32)
                out["pc_mask"] = np.asarray(mask, bool)
            except Exception as e:
                info["pointcloud_error"] = repr(e)[:200]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, **out)
        info["path"] = str(path)
        info["keys"] = sorted(out)
        return info

    # ---------------- video ----------------
    def start_video(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "22", path], stdin=subprocess.PIPE)

    def frame(self):
        img = np.asarray(self.view.render()[0]).astype(np.uint8)
        self.proc.stdin.write(img.tobytes())

    def finish_video(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()

    # ---------------- sim ----------------
    def cloth_pos(self):
        return self.cloth.get_state().pos.cpu().numpy().reshape(-1, 3)

    def step(self):
        # self.cmd stays the plan (planner seeds, goals); --ff only changes the PD target sent to the solver
        self.robot.control_dofs_position(self.ff.command(self.cmd) - self._dq_off(), self.dofs)
        t0 = time.time()
        self.scene.step()
        self.sim_time += time.time() - t0
        self.nstep += 1
        self.q = self.robot.get_dofs_position(self.dofs).cpu().numpy().reshape(-1) + self._dq_off()
        if self.qlog is not None and self.nstep % self.args.qlog_every == 0:
            self.qlog.append((self.seg_i, self.q.copy()))
        if self.nstep % 5 == 0:
            x = self.cloth_pos()
            if not np.isfinite(x).all():
                self.nan = True
                raise FloatingPointError(f"cloth NaN at step {self.nstep}")
            if self.prev_x is not None:
                v = np.linalg.norm(x - self.prev_x, axis=1).max() / (5 * DT)
                self.max_speed = max(self.max_speed, float(v))
            ln = np.linalg.norm(x[self.edges[:, 0]] - x[self.edges[:, 1]], axis=1)
            self.max_stretch = max(self.max_stretch, float((ln / self.rest_len).max()))
            self.prev_x = x
            if self.carry_track is not None:
                tip = self.tcp_world()
                t0_, t1_ = self.touching(x)
                S = self.pinch_set   # vertices touching a finger at the end of closing: slide relative to the tip
                slide = float(np.linalg.norm((x[S] - tip) - self.pinch_rel0, axis=1).mean()) if len(S) else np.nan
                self.carry_track.append((float(np.linalg.norm(x - tip, axis=1).min()), float(x[:, 2].max()),
                                         float(tip[2]), self.squeeze_n()[0], t0_, t1_, slide))
        if self.proc and self.nstep * DT >= self.nframes / FPS - 1e-9:   # one frame per 1/FPS of sim time
            self.frame()
            self.nframes += 1

    def move(self, goal, dur):
        q0 = self.cmd.copy()
        n = max(1, int(dur / DT))
        for k in range(1, n + 1):
            a = k / n
            a = a * a * (3 - 2 * a)
            self.cmd = q0 + a * (goal - q0)
            self.step()

    def hold(self, dur):
        for _ in range(int(dur / DT)):
            self.step()

    def gripper(self, v, dur=0.8):
        g = self.cmd.copy()
        g[10], g[11] = v, -v
        self.move(g, dur)

    # ---------------- planned motion (--planner) ----------------
    def follow(self, seg, row, grip_to=None):
        """Execute one planned segment: its joint path retimed (smoothstep in joint arc length) to max(nominal
        duration, arc / --vmax). Gripper command: ramped to grip_to (same smoothstep) on a gripper move, else the
        opening commanded last (self.grip_hold, e.g. the closing target beyond CLOSED) overrides the planned one,
        which is only the collision model (TransferSeq.hold)."""
        r = seg["res"]
        dur = seg_duration(r.path, seg["dur"], self.args.vmax)
        row.update(t0_s=round(self.nstep * DT, 2), dur_s=round(dur, 2))
        self.plan_rows.append(row)
        self.seg_i = len(self.plan_rows) - 1
        Q = retime(r.path, dur, DT)
        g0, n = self.cmd[10], len(Q)
        for k, qk in enumerate(Q, 1):
            c = qk.copy()
            if grip_to is not None:
                a = k / n
                a = a * a * (3 - 2 * a)
                c[10] = g0 + a * (grip_to - g0)
                c[11] = -c[10]
            elif self.grip_hold is not None:
                c[10], c[11] = self.grip_hold, -self.grip_hold
            self.cmd = c
            self.step()
        if grip_to is not None:
            self.grip_hold = grip_to

    def plan_exec(self, fn, *args, grip_to=None):
        """Call a TransferSeq method (plans on the planner scene from the last commanded configuration) and execute
        the segments it added; raise PlanningError (nothing executed for that segment) if one failed."""
        n0 = len(self.T.segs)
        fn(*args)
        for seg in self.T.segs[n0:]:
            r = seg["res"]
            row = dict(name=seg["name"], ok=bool(r.ok), method=r.method, plan_s=round(r.plan_s, 3),
                       rrt_calls=r.rrt_calls, min_clear_plan_mm=round(float(r.min_clearance) * 1e3, 1),
                       **{k: v for k, v in r.info.items() if k in ("yaw", "yaws_tried", "bottom_up", "route", "z",
                                                                   "goal_rank", "pose_candidates_tried", "knots")})
            if not r.ok:
                row.update(reason=r.reason[:300], t0_s=round(self.nstep * DT, 2))
                self.plan_rows.append(row)
                raise PlanningError(f"{seg['name']}: {r.reason[:200]}")
            self.follow(seg, row, grip_to)

    def p_grip(self, name, v, dur, plan_v=None):
        """Planned gripper move (arm fixed) to plan_v (default v), commanded opening ramped to v (the closing
        target may lie beyond the planned collision-model opening TransferSeq.hold)."""
        self.plan_exec(self.T.grip, name, v if plan_v is None else plan_v, dur, grip_to=v)

    def yaw_candidates(self, x, i):
        """Gripper yaws to try in order: across the local fold first (--yaw-mode ridge, 8 best) or the fixed yaw
        and +-45 / +-90. --planner: the whole 15 deg grid in that order (along a basket wall only about half the
        yaws are collision-free: the gripper body has to lie along the wall)."""
        a = self.args
        if a.yaw_mode == "ridge":
            yaws, self.res["ridge_anisotropy"] = self.ridge_yaws(
                x, x[i], self.hmap_ridge if a.select == "hmap" else None)
            return yaws if self.T is not None else yaws[:8]
        yaws = [a.yaw, a.yaw - 45, a.yaw + 45, a.yaw - 90, a.yaw + 90]
        if self.T is not None:
            yaws += sorted((float(y) for y in np.arange(-180, 180, 15.0) if y not in yaws),
                           key=lambda y: abs((y - a.yaw + 90) % 180 - 90))
        return yaws

    def approach_planned(self, name, x, i, g, fb, dur):
        """Planned move to the hover pose above g with the first candidate yaw whose descent also plans."""
        self.plan_exec(self.T.approach, name, g, fb, self.yaw_candidates(x, i), dur)
        self.grasp_yaw = self.T.target["yaw"]
        self.grasp_tilt = 0.0
        return self.T.target["quat"], self.grasp_yaw

    def measured_stats(self):
        """Penetration / clearance of the measured joint positions (every --qlog-every steps) in the planner scene,
        per executed segment (hold phases count to the segment before them)."""
        per = {}
        worst_all = (0.0, "")
        stats = self.AP.clearance_stats_many([q for _, q in self.qlog])   # one request (planner process)
        for (si, _), (pen, clr, worst) in zip(self.qlog, stats):
            S = per.setdefault(si, dict(n=0, max_pen_mm=0.0, min_clear_mm=np.inf, worst=""))
            S["n"] += 1
            if pen * 1e3 > S["max_pen_mm"]:
                S["max_pen_mm"], S["worst"] = pen * 1e3, worst
            S["min_clear_mm"] = min(S["min_clear_mm"], clr * 1e3)
            if pen > worst_all[0]:
                worst_all = (pen, worst)
        rows = []
        for si, S in sorted(per.items()):
            rows.append(dict(seg=si, name=self.plan_rows[si]["name"], n=S["n"], max_pen_mm=round(S["max_pen_mm"], 2),
                             min_clear_mm=round(S["min_clear_mm"], 1) if np.isfinite(S["min_clear_mm"]) else None,
                             worst=S["worst"]))
        return rows, worst_all

    def ik(self, tcp, quat, lift=None):
        """Damped-LS IK for the fingertip centre with several seeds (PiPER IK has local minima).
        set_dofs_position is only used for seeding and is restored before the next step."""
        lift = self.cmd[3] if lift is None else lift
        seed = self.cmd.copy()
        seed[3] = lift
        self.robot.set_dofs_position(seed, self.dofs, zero_velocity=False)
        base = self.robot.get_link("link1").get_pos().cpu().numpy().reshape(-1)
        az = math.atan2(tcp[1] - base[1], tcp[0] - base[0])
        seeds = [self.cmd[4:10].copy()]
        for j1 in (az,):
            seeds += [np.array([j1, 0.5, -0.8, 0, 0.5, 0]), np.array([j1, 1.0, -1.2, 0, 0.6, 0]),
                      np.array([j1, 1.5, -1.0, 0, 0.8, 0]), np.array([j1, 0.3, -0.3, 0, 0.3, 0])]
        best = None
        for s6 in seeds:
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
            if np.abs(e[:3]).max() < 0.003 and np.abs(e[3:]).max() < 0.03 and best[1] is q:
                break
        self.robot.set_dofs_position(self.q, self.dofs, zero_velocity=False)
        _, q, e = best
        assert np.abs(e[:3]).max() < 0.01 and np.abs(e[3:]).max() < 0.1, f"IK failed {tcp}: {e}"
        goal = self.cmd.copy()
        goal[3] = lift
        goal[4:10] = q
        return goal

    def ik_high(self, x, y, zs, lift, tilts=(0, 30, 45, 60)):
        """Highest reachable z in `zs` (descending) above (x, y); for each z the smallest tilt towards the target
        azimuth that reaches it."""
        seed = self.cmd.copy()
        seed[3] = lift
        self.robot.set_dofs_position(seed, self.dofs, zero_velocity=False)
        base = self.robot.get_link("link1").get_pos().cpu().numpy().reshape(-1)
        self.robot.set_dofs_position(self.q, self.dofs, zero_velocity=False)
        az = math.atan2(y - base[1], x - base[0])
        for z in zs:
            for tilt in tilts:
                try:
                    quat = quat_down(self.grasp_yaw) if tilt == 0 else quat_toward(az, tilt)
                    return self.ik((x, y, z), quat, lift), z, tilt
                except AssertionError:
                    pass
        raise AssertionError(f"unreachable {x, y, zs}")

    def link6_rot(self):
        qw = self.link6.get_quat().cpu().numpy().reshape(-1)
        return Rotation.from_quat([qw[1], qw[2], qw[3], qw[0]])

    def ipc_lag_mm(self):
        """Distance [mm] between each gripper link's pose in IPC (ABD) and in Genesis (coupler internals)."""
        out = {}
        for link in [self.link6] + self.fingers:
            T = self.scene.sim.coupler._abd_data_by_link[link][0].transform
            out[link.name] = round(float(np.linalg.norm(T[:3, 3] - link.get_pos().cpu().numpy().reshape(-1))) * 1e3, 2)
        return out

    def ridge_yaws(self, x, g, ridge=None):
        """Gripper yaws [deg] ordered so the closing axis is most perpendicular to the local fold ridge at g
        (principal xy direction of the vertices within 4 cm of g and less than 3 cm below it)."""
        sel = (np.linalg.norm(x[:, :2] - g[:2], axis=1) < 0.04) & (x[:, 2] > g[2] - 0.03)
        d = x[sel, :2] - x[sel, :2].mean(0)
        w, V = np.linalg.eigh(d.T @ d)
        ridge = np.r_[V[:, 1], 0.0] if ridge is None else ridge
        cands = np.arange(-180, 180, 15)
        score = [abs(Rotation.from_quat(np.roll(quat_down(y), -1)).apply(self.close_axis_l6) @ ridge) for y in cands]
        self.ridge = ridge
        return [float(cands[i]) for i in np.argsort(score)], float(w[1] / max(w[0], 1e-9))

    def squeeze_n(self):
        """IPC-side squeeze [N]: coupling force of the IPC finger copies pushed outward along the closing axis
        (force = strength / dt^2 * mass * (pose_ipc - pose_genesis), genesis/engine/couplers/ipc_coupler/utils.py),
        mean of both fingers; and the summed vertical coupling force of link6/7/8."""
        cd = self.scene.sim.coupler._coupling_data
        if cd is None:
            return float("nan"), float("nan")
        v0, v1 = self.finger_verts()
        ax = v1.mean(0) - v0.mean(0)
        ax /= np.linalg.norm(ax)
        f = [cd.out_forces[0, cd.link_to_idx_local[k]] for k in [self.link6] + self.fingers]
        return float((-f[1] @ ax + f[2] @ ax) / 2), float(sum(fi[2] for fi in f))

    def tcp_world(self):
        p = self.link6.get_pos().cpu().numpy().reshape(-1)
        qw = self.link6.get_quat().cpu().numpy().reshape(-1)
        R = Rotation.from_quat([qw[1], qw[2], qw[3], qw[0]])
        return p + R.apply([0, 0, TCP])

    def finger_verts(self):
        return [f.get_verts().cpu().numpy().reshape(-1, 3) for f in self.fingers]

    def finger_bottom_z(self):
        return min(float(v[:, 2].min()) for v in self.finger_verts())

    def between_fingers(self, x, half_w=0.02):
        """Cloth vertices in the slab between the two finger pads (within half_w of the closing axis)."""
        v0, v1 = self.finger_verts()
        c0, c1 = v0.mean(0), v1.mean(0)
        ax = (c1 - c0) / np.linalg.norm(c1 - c0)
        s = (x - c0) @ ax
        lo, hi = (v0 - c0) @ ax, (v1 - c0) @ ax
        zb = np.concatenate([v0, v1])[:, 2].min()
        perp = x - c0 - np.outer(s, ax)
        perp[:, 2] = 0
        return int(((s > lo.max() - 0.005) & (s < hi.min() + 0.005) & (x[:, 2] > zb - 0.003) &
                    (x[:, 2] < zb + 0.04) & (np.linalg.norm(perp, axis=1) < half_w)).sum())

    def touching(self, x, tol=0.003):
        """Cloth vertices within `tol` of each finger's convex hull (Genesis geometry); a pinch needs both > 0."""
        out = []
        for v in self.finger_verts():
            eq = ConvexHull(v).equations           # n.x + d <= 0 inside, unit normals
            out.append(int(((x @ eq[:, :3].T + eq[:, 3]).max(1) < tol).sum()))
        return out

    def n_inside(self, x, hulls):
        n = 0
        for v in hulls:
            n += int((Delaunay(v).find_simplex(x) >= 0).sum())
        return n

    def penetration(self, x):
        """Cloth vertices inside the gripper fingers / washer / base+basket collision boxes (Genesis geometry)."""
        solids = [g.get_verts().cpu().numpy().reshape(-1, 3) for g in self.solid_geoms]
        return {"fingers": self.n_inside(x, self.finger_verts()), "washer_basket": self.n_inside(x, solids)}

    @staticmethod
    def in_tub_xy(x):
        lx, ly = P.base_to_washer(x[:, 0], x[:, 1], WASHER_XY, math.radians(WASHER_YAW_DEG))
        return (np.abs(lx) < TUB_HALF) & (np.abs(ly) < TUB_HALF)

    @staticmethod
    def in_tub(x):
        lx, ly = P.base_to_washer(x[:, 0], x[:, 1], WASHER_XY, math.radians(WASHER_YAW_DEG))
        return (np.abs(lx) < TUB_HALF) & (np.abs(ly) < TUB_HALF) & (x[:, 2] < P.WASHER_H + WASHER_Z)

    def in_basket(self, x):
        bx, by = _basket_xy(self)
        ix, iy, iz = P.BASKET_INNER
        return (np.abs(x[:, 0] - bx) < ix / 2) & (np.abs(x[:, 1] - by) < iy / 2) & (x[:, 2] < BASKET_FLOOR + iz)

    def select_grasp(self, x, exclude=(), center=None, radius=None):
        """Highest vertex within grasp_r of the heap centroid (kept 5 cm inside the basket walls); finger bottom
        target `depth` below it (or at floor + press), but at least `floor_clear` above the basket floor. With
        `min_bite` > 0 the radius grows (up to 0.5 m) until the highest vertex is `floor_clear + min_bite` above the
        floor (the highest one found otherwise): on a flat heap the fingertips would press on the basket floor and
        stall half open, holding only a thin edge.
        `center` (xy) replaces the heap centroid as the search centre and `radius` replaces `grasp_r` (the returned
        `c` is the search centre)."""
        a = self.args
        c = x[:, :2].mean(0) if center is None else np.asarray(center, float)[:2]
        r = a.grasp_r if radius is None else radius
        bx, by = _basket_xy(self)
        ix, iy, _ = P.BASKET_INNER
        inner = (np.abs(x[:, 0] - bx) < ix / 2 - 0.05) & (np.abs(x[:, 1] - by) < iy / 2 - 0.05)
        for ex in exclude:  # earlier failed attempts
            inner &= np.linalg.norm(x[:, :2] - np.array(ex), axis=1) > a.exclude_r
        if a.select == "hmap":
            return self.select_grasp_hmap(x, inner)
        need = BASKET_FLOOR + a.floor_clear + a.min_bite
        i = None
        while r < 0.5 + 1e-9:
            sel = np.where((np.linalg.norm(x[:, :2] - c, axis=1) < r) & inner)[0]
            if len(sel):
                j = sel[np.argmax(x[sel, 2])]
                if i is None or x[j, 2] > x[i, 2]:
                    i = j
                if a.min_bite <= 0 or x[i, 2] >= need:
                    break
            r += 0.02
        if i is None:
            raise AssertionError("no grasp candidate left")
        gx, gy, gz = (float(v) for v in x[i])
        fb = BASKET_FLOOR + a.press if a.press < 0.01 else max(BASKET_FLOOR + a.floor_clear, gz - a.depth)
        return gx, gy, gz, fb, i, c

    def select_grasp_hmap(self, x, inner, cell=0.01):
        """Top-down height map of the heap (what the basket camera sees, 1 cm cells, smoothed); among the cells
        within `hmap_band` of the highest one, take the sharpest ridge (most negative principal curvature). The
        closing axis goes along that curvature direction (across the ridge), see self.hmap_ridge."""
        a = self.args
        xi = x[inner]
        lo = xi[:, :2].min(0) - 2 * cell
        ij = np.floor((x[:, :2] - lo) / cell).astype(int)
        n = ij[inner].max(0) + 3
        ok = inner & (ij[:, 0] < n[0]) & (ij[:, 1] < n[1]) & (ij >= 0).all(1)
        Hm = np.full(n, -np.inf)
        np.maximum.at(Hm, (ij[ok, 0], ij[ok, 1]), x[ok, 2])
        valid = np.isfinite(Hm)
        Hf = np.where(valid, Hm, ndimage.maximum_filter(np.where(valid, Hm, -1.0), size=3))
        Hf = np.where(Hf < 0, BASKET_FLOOR, Hf)
        Hs = ndimage.gaussian_filter(Hf, 1.0)
        gx_, gy_ = np.gradient(Hs, cell)
        hxx, hxy = np.gradient(gx_, cell)
        _, hyy = np.gradient(gy_, cell)
        tr, det = hxx + hyy, hxx * hyy - hxy ** 2
        lam_min = tr / 2 - np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
        cand = valid & (Hm >= Hm[valid].max() - a.hmap_band)
        score = np.where(cand, -lam_min, -np.inf)
        ci, cj = np.unravel_index(np.argmax(score), score.shape)
        # eigenvector of lam_min: (hxy, lam - hxx) or (lam - hyy, hxy)
        v = np.array([hxy[ci, cj], lam_min[ci, cj] - hxx[ci, cj]])
        if np.linalg.norm(v) < 1e-9:
            v = np.array([lam_min[ci, cj] - hyy[ci, cj], hxy[ci, cj]])
        v = v / max(np.linalg.norm(v), 1e-12)
        self.hmap_ridge = np.array([-v[1], v[0], 0.0])
        self.res["hmap_curv_min"] = float(lam_min[ci, cj])
        idx = np.where(ok & (ij[:, 0] == ci) & (ij[:, 1] == cj))[0]
        i = idx[np.argmax(x[idx, 2])]
        gx, gy, gz = (float(v) for v in x[i])
        fb = BASKET_FLOOR + a.press if a.press < 0.01 else max(BASKET_FLOOR + 0.003, gz - a.depth)
        return gx, gy, gz, fb, i, x[:, :2].mean(0)

    def grasp_orientation(self, x, i, gx, gy, gz, fb):
        """First reachable gripper yaw, preferring the closing axis across the local fold (ridge mode)."""
        a = self.args
        yaws = self.yaw_candidates(x, i)
        tilts = (0.0,) if a.tilt == 0 else (a.tilt, -a.tilt)
        for yaw in yaws:
            for tilt in tilts:
                down = self.tilted(quat_down(yaw), tilt)
                try:
                    self.ik(self.hover_point((gx, gy, gz), down, gz + 0.15), down)
                    self.ik((gx, gy, fb + 0.02), down)
                    self.grasp_yaw = yaw
                    self.grasp_tilt = tilt
                    return down, yaw
                except AssertionError:
                    continue
        raise AssertionError("no gripper yaw reaches the grasp point")

    def tilted(self, quat, tilt):
        """wxyz `quat` rotated by `tilt` deg about the horizontal axis perpendicular to its closing axis (the
        fold ridge): the closing axis and the approach direction tilt, one finger comes in lower than the other."""
        if tilt == 0:
            return quat
        R = Rotation.from_quat(np.roll(quat, -1))
        ax = np.cross(R.apply(self.close_axis_l6), [0, 0, 1.0])
        q = (Rotation.from_rotvec(np.radians(tilt) * ax / np.linalg.norm(ax)) * R).as_quat()
        return np.array([q[3], q[0], q[1], q[2]])

    @staticmethod
    def hover_point(g, quat, z):
        """Point on the approach line through g (along the tool axis, link6 z) at height z."""
        d = Rotation.from_quat(np.roll(quat, -1)).apply([0, 0, 1.0])   # tool axis, pointing down
        return np.asarray(g, float) - d * (z - g[2]) / -d[2]

    def attempt(self, x0, exclude):
        """One pinch + lift from the settled heap x0 (grasp point away from the xy points in `exclude`); writes the
        numbers into self.res; returns the grasp xy, finger-bottom target and TCP-to-finger-bottom offset."""
        a = self.args
        res = self.res
        for key in ("grasp_point_resensed", "miss", "stall", "weak_hold", "lift_slide_mm", "lift_success",
                    "lift_lost_at_tip_dz", "lift_dzmean",
                    "lifted_frac_above_rim", "lifted_tip_to_nearest_vertex", "plan_fail"):
            res.pop(key, None)
        ext = self.grasp_center(x0, len(exclude))   # same external centre for the re-sense below
        ctr, rad = ext, None
        if ext is not None and len(ext) == 2 and np.ndim(ext[0]) > 0:   # (xy, radius)
            ctr, rad = ext
        gx, gy, gz, fb_target, i, c = self.select_grasp(x0, exclude, center=ctr, radius=rad)
        res["grasp_point"] = [round(gx, 3), round(gy, 3), round(gz, 3)]
        res["grasp_offset_from_centroid"] = float(np.linalg.norm(np.array([gx, gy]) - c))
        if self.T is None:
            down, yaw = self.grasp_orientation(x0, i, gx, gy, gz, fb_target)
            res["grasp_yaw"] = yaw
            res["grasp_tilt"] = getattr(self, "grasp_tilt", 0.0)
        bx, by = _basket_xy(self)
        if a.cam == "close":  # close-up of the grasp spot from the basket centre side
            d = np.array([bx - gx, by - gy])
            d = d / np.linalg.norm(d) if np.linalg.norm(d) > 0.03 else np.array([1.0, 0.0])
            self.cam.set_pose(pos=(gx + 0.18 * d[0], gy + 0.18 * d[1], gz + 0.10), lookat=(gx, gy, gz - 0.03))
        if self.T is None:
            self.gripper(a.pre_open, 0.3)
            self.move(self.ik(self.hover_point((gx, gy, gz), down, gz + 0.15), down), 2.0)
        else:  # yaw = first candidate whose hover pose and descent both plan; unplannable point = no motion, retry
            self.p_grip("pre_open", a.pre_open, 0.3)
            try:
                down, yaw = self.approach_planned("standby->hover" if len(self.plan_rows) <= 3 else "hover", x0, i,
                                                  (gx, gy, gz), fb_target, 2.0)
            except PlanningError as e:
                res.update(miss=True, lift_success=False, plan_fail=str(e)[:300])
                return gx, gy, fb_target, None
            res["grasp_yaw"] = yaw
            res["grasp_tilt"] = 0.0
        # re-sense at the hover pose (closed loop): if the heap moved, re-pick the grasp point
        xh = self.cloth_pos() if self.resense else x0
        res["heap_drift_mm_settle_to_hover"] = float(np.linalg.norm(xh - x0, axis=1).max() * 1e3) if self.resense else None
        if self.resense and res["heap_drift_mm_settle_to_hover"] > 5.0:
            gx, gy, gz, fb_target, i, c = self.select_grasp(xh, exclude, center=ctr, radius=rad)
            res["grasp_point_resensed"] = [round(gx, 3), round(gy, 3), round(gz, 3)]
            if self.T is None:
                down, yaw = self.grasp_orientation(xh, i, gx, gy, gz, fb_target)
                self.move(self.ik(self.hover_point((gx, gy, gz), down, gz + 0.15), down), 1.0)
            else:
                try:
                    down, yaw = self.approach_planned("hover (re-sensed)", xh, i, (gx, gy, gz), fb_target, 1.0)
                except PlanningError as e:
                    res.update(miss=True, lift_success=False, plan_fail=str(e)[:300])
                    return gx, gy, fb_target, None
            res["grasp_yaw"] = yaw
        x0 = xh

        cyl = lambda x: int(((np.linalg.norm(x[:, :2] - np.array([gx, gy]), axis=1) < 0.035) &
                             (x[:, 2] > fb_target) & (x[:, 2] < fb_target + 0.05)).sum())
        res["cyl_verts"] = {"hover": cyl(xh)}
        self.hold(0.3)
        off = self.tcp_world()[2] - self.finger_bottom_z()
        c = [v.mean(0) for v in self.finger_verts()]
        ax = (c[1] - c[0]) / np.linalg.norm(c[1] - c[0])
        res["closing_axis_world"] = [round(float(v), 2) for v in ax]
        res["tip_xy_err_at_approach"] = float(np.linalg.norm(self.tcp_world()[:2] - np.array([gx, gy])))
        if hasattr(self, "ridge"):
            res["ridge_dir"] = [round(float(v), 2) for v in self.ridge]
        if self.T is None:
            goal = self.ik((gx, gy, fb_target + off), down)
            if a.scoop:  # close while descending: the pads gather the cloth instead of pressing it out of the gap
                goal[10], goal[11] = a.scoop_close, -a.scoop_close
            self.move(goal, a.descend_s)
        else:  # TCP target from the planner kinematics (fb_target + FINGER_BOTTOM_TO_TCP), branch checked at hover
            res["tcp_minus_finger_bottom_measured"] = round(float(off), 4)
            self.plan_exec(self.T.descend, a.descend_s)
        self.hold(0.3)
        res["finger_bottom_at_grasp_minus_floor"] = self.finger_bottom_z() - BASKET_FLOOR
        xo = self.cloth_pos()
        res["cyl_verts"]["open_at_depth"] = cyl(xo)
        res["verts_between_fingers_open"] = self.between_fingers(xo, 0.028)
        res["touch_open"] = self.touching(xo)
        res["ipc_lag_mm_open"] = self.ipc_lag_mm()
        if a.press_dz > 0:  # close while pressing further down (slowly): the pads drag the cloth into the gap
            fb2 = max(BASKET_FLOOR + 0.003, fb_target - a.press_dz)
            goal = self.ik((gx, gy, fb2 + off), down)
            goal[10], goal[11] = self.close_target, -self.close_target
            self.move(goal, a.close_s)
            res["finger_bottom_after_press_close_minus_floor"] = self.finger_bottom_z() - BASKET_FLOOR
        elif self.T is None:
            self.gripper(self.close_target, a.close_s)
        else:
            self.p_grip("close", self.close_target, a.close_s, plan_v=self.T.hold)
        self.hold(0.5)
        x1 = self.cloth_pos()
        res["joint7_8_closed"] = [round(float(self.q[10]), 4), round(float(self.q[11]), 4)]
        res["pen_closed"] = self.penetration(x1)
        res["verts_between_fingers_closed"] = self.between_fingers(x1, 0.028)
        res["touch_closed"] = self.touching(x1)
        res["ipc_lag_mm_closed"] = self.ipc_lag_mm()
        res["ipc_squeeze_N_closed"], res["ipc_fz_N_closed"] = (round(v, 3) for v in self.squeeze_n())
        gap = float(self.q[10] - self.q[11])
        # fingertips on the basket floor: stalled half open, pushed up by the floor (only a thin edge is held)
        res["stall"] = bool(a.stall_check and gap > a.stall_gap and res["ipc_fz_N_closed"] > a.stall_fz)
        if res["stall"] or (a.miss_check and (res["ipc_squeeze_N_closed"] < a.miss_n or min(res["touch_closed"]) == 0)):
            # empty or floor-stalled pinch: do not lift; open and pull the fingers out of the heap, the caller
            # retries elsewhere
            res["miss"] = True
            res["lift_success"] = False
            if self.T is None:
                self.gripper(a.pre_open, 0.4)
                self.move(self.ik(self.hover_point((gx, gy, fb_target), down, fb_target + 0.15), down), 1.0)
            else:
                self.p_grip("miss_open", a.pre_open, 0.4)
                self.plan_exec(self.T.up_from_grasp, "miss_up", 1.0)
            return gx, gy, fb_target, off
        tip0 = self.tcp_world()
        v0, v1 = self.finger_verts()
        S = []
        for v in (v0, v1):
            eq = ConvexHull(v).equations
            S += list(np.where((x1 @ eq[:, :3].T + eq[:, 3]).max(1) < 0.003)[0])
        self.pinch_set = np.unique(np.array(S, dtype=int))
        self.pinch_rel0 = x1[self.pinch_set] - tip0

        # lift 1: straight up with the arm (lift joint at the grasp height)
        self.carry_track = []
        if self.T is not None:
            self.plan_exec(self.T.lift1, a.lift1_s)
            z1 = self.T.segs[-1]["res"].info["z"]
        elif a.tilt == 0:
            goal, z1, _ = self.ik_high(gx, gy, np.arange(a.lift1_z, fb_target + off + 0.1, -0.05), LIFT_GRASP,
                                       tilts=(0,))
        else:  # keep the tilted grasp orientation
            for z1 in np.arange(a.lift1_z, fb_target + off + 0.1, -0.05):
                try:
                    goal = self.ik((gx, gy, z1), down, LIFT_GRASP)
                    break
                except AssertionError:
                    continue
            else:
                raise AssertionError("tilted lift unreachable")
        res["lift1_tip_z"] = round(float(z1), 3)
        if self.T is None:
            self.move(goal, a.lift1_s)
            # lift 2: raise the lift joint (pure vertical translation of the arm)
            goal = self.cmd.copy()
            goal[3] = LIFT_CARRY
            self.move(goal, a.lift2_s)
        else:
            self.plan_exec(self.T.lift_to, "lift2 (lift joint)", LIFT_CARRY, a.lift2_s)
        self.hold(0.5)
        x2 = self.cloth_pos()
        tip2 = self.tcp_world()
        res["tcp_lift_dz"] = float(tip2[2] - tip0[2])
        res["lift_dzmax"] = float(x2[:, 2].max() - x1[:, 2].max())
        res["lift_dzmean"] = float(x2[:, 2].mean() - x1[:, 2].mean())
        res["lifted_tip_z"] = float(tip2[2])
        res["lifted_cloth_zmin"] = float(x2[:, 2].min())
        res["lifted_frac_out_of_basket"] = float((~self.in_basket(x2)).mean())
        res["lifted_tip_to_nearest_vertex"] = float(np.linalg.norm(x2 - tip2, axis=1).min())
        res["touch_lifted"] = self.touching(x2)
        tr = np.array(self.carry_track)   # lost = tip rise when the nearest vertex first got > 3 cm from the tip
        lost = np.where(tr[:, 0] > 0.03)[0]
        res["lift_lost_at_tip_dz"] = None if len(lost) == 0 else round(float(tr[lost[0], 2] - tip0[2]), 3)
        held = tr[: lost[0]] if len(lost) else tr
        # samples (tip rise [m], touching finger0/finger1, mean slide of the initially pinched vertices [mm])
        k = np.linspace(0, len(tr) - 1, min(len(tr), 14)).astype(int)
        res["lift_trace_dz_touch_slide"] = [[round(float(tr[j, 2] - tip0[2]), 3), int(tr[j, 4]), int(tr[j, 5]),
                                             round(float(tr[j, 6]) * 1e3, 1)] for j in k]
        if len(held):  # IPC squeeze while the cloth was still held
            res["ipc_squeeze_N_lift_min_mean"] = [round(float(held[:, 3].min()), 3), round(float(held[:, 3].mean()), 3)]
        res["lifted_frac_above_rim"] = float((x2[:, 2] > BASKET_RIM).mean())
        # lift success (stage-1 criterion): >= 50 % of the vertices above the basket rim, a vertex within 3 cm of the
        # fingertip centre
        res["lift_success"] = bool(res["lifted_frac_above_rim"] >= 0.5 and res["lifted_tip_to_nearest_vertex"] <= 0.03)
        # --lift-slide-max: the pinched vertices slid this far during the lift = only a thin edge is held, which slips
        # out during the carry (all 5 failures of 20 runs slid > 10 mm, 13 of 15 successes <= 10 mm) -> retry instead
        res["lift_slide_mm"] = round(float(tr[-1, 6]) * 1e3, 1) if len(tr) and np.isfinite(tr[-1, 6]) else None
        res["weak_hold"] = bool(a.lift_slide_max > 0 and len(tr) and tr[-1, 6] > a.lift_slide_max)
        if res["weak_hold"]:
            res["lift_success"] = False
        res["pen_lifted"] = self.penetration(x2)
        return gx, gy, fb_target, off

    def quat_tilt_keep_yaw(self, x, y, tilt):
        """wxyz: the vertical grasp orientation (gripper yaw kept) tilted by `tilt` deg so the tool axis leans
        outward along the azimuth of (x, y) from link1 (no wrist yaw flip when the tilt grows)."""
        base = self.robot.get_link("link1").get_pos().cpu().numpy().reshape(-1)
        az = math.atan2(y - base[1], x - base[0])
        u = np.array([-math.sin(az), math.cos(az), 0.0])
        R = Rotation.from_rotvec(-math.radians(tilt) * u) * Rotation.from_quat(np.roll(quat_down(self.grasp_yaw), -1))
        q = R.as_quat()
        return np.array([q[3], q[0], q[1], q[2]])

    def carry_gentle(self, gx, gy, rx, ry):
        """Raise straight up above the grasp point first, then move to the release point through the waypoints with
        the tilt ramped by --tilt-step per waypoint (0 -> 60 deg; a vertical gripper only reaches ~0.9 m, crossing
        the washer top needs the tilt). Each waypoint: the planned tilt (or the next larger one if unreachable), the
        gripper yaw kept if reachable (else the default yaw toward the target), as high as reachable; waypoints
        outside the washer footprint may go down to 0.85 m."""
        a = self.args
        goal, z0, _ = self.ik_high(gx, gy, np.arange(a.carry_z, 0.6, -0.05), LIFT_CARRY, tilts=(0,))
        self.move(goal, a.raise_s)
        wps = [(round(float(gx), 3), round(float(gy), 3), round(float(z0), 3), 0, "keep")]
        for k in range(1, a.n_wp + 1):
            s = k / a.n_wp
            px, py = gx + s * (rx - gx), gy + s * (ry - gy)
            lx, ly = P.base_to_washer(px, py, WASHER_XY, math.radians(WASHER_YAW_DEG))
            over = abs(lx - P.WASHER_BODY_CX) < P.WASHER_W / 2 + 0.05 and abs(ly) < P.WASHER_D / 2 + 0.05
            zs = np.arange(a.carry_z, (P.WASHER_H + 0.04) if over else 0.85, -0.05)
            plan = min(60.0, k * a.tilt_step)
            found = None
            for t in [t for t in np.arange(plan, 61.0, 5.0)]:
                for z in zs:
                    for kind, q in (("keep", self.quat_tilt_keep_yaw(px, py, t)),
                                    ("toward", quat_toward(self._az(px, py), t))):
                        try:
                            found = (self.ik((px, py, z), q, LIFT_CARRY), z, t, kind)
                            break
                        except AssertionError:
                            continue
                    if found:
                        break
                if found:
                    break
            if found is None:  # fall back to the default carry orientation
                g, z, t = self.ik_high(px, py, zs, LIFT_CARRY, tilts=(40, 50, 60))
                found = (g, z, t, "default")
            goal, z, t, kind = found
            wps.append((round(float(px), 3), round(float(py), 3), round(float(z), 3), float(t), kind))
            self.move(goal, a.carry_s / a.n_wp)
        return wps

    def _az(self, x, y):
        base = self.robot.get_link("link1").get_pos().cpu().numpy().reshape(-1)
        return math.atan2(y - base[1], x - base[0])

    # ---------------- scenario ----------------
    def run(self):
        a = self.args
        res = self.res
        res["drop"] = self.drop
        # the towel (no bending stiffness) keeps slumping for seconds: wait until it is still, otherwise the grasp
        # point picked from it is stale by the time the fingers arrive (heap top drops several cm)
        self.hold(a.settle)
        xp = self.cloth_pos()
        drift = float("nan")
        while self.nstep * DT < a.settle_max:
            self.hold(0.5)
            xn = self.cloth_pos()
            drift = float(np.linalg.norm(xn - xp, axis=1).max())
            xp = xn
            if drift < a.still_tol:
                break
        res["settle_s"] = round(self.nstep * DT, 2)
        res["settle_last_0.5s_max_drift_mm"] = round(drift * 1e3, 2)
        x0 = self.cloth_pos()
        res["n_vertices"] = len(x0)
        np.save(ROOT / "outputs" / f"cloth_ipc_washer_s{a.seed}_settled.npy", x0)
        res["settled_in_basket_frac"] = float(self.in_basket(x0).mean())
        res["settled_zmax_above_floor"] = float(x0[:, 2].max() - BASKET_FLOOR)
        res["settled_zmin_minus_floor"] = float(x0[:, 2].min() - BASKET_FLOOR)
        res["pen_settled"] = self.penetration(x0)
        if a.save_obs:
            res["save_obs"] = self.save_obs(a.save_obs, x0)
        if res["settled_in_basket_frac"] < 0.99:
            raise RuntimeError("towel did not settle inside the basket")
        if a.stop_after == "settle":
            return res
        if self.after_settle(res):
            return res
        if self.T is not None:   # waited at LIFT_CARRY (outside the drop envelope): planned lowering to the grasp lift
            self.plan_exec(self.T.lower_to_grasp, "lower to grasp lift", 1.5)

        tried = []
        res["attempts_log"] = []
        for k in range(a.attempts):
            if k > 0 and not res["attempts_log"][-1].get("miss"):
                # lift failed: open, lower the arm back into the basket region, let the heap settle, try elsewhere
                if self.T is None:
                    self.gripper(OPEN, 0.5)
                    goal = self.cmd.copy()
                    goal[3] = LIFT_GRASP
                    self.move(goal, 1.5)
                else:
                    self.p_grip("retry_open", OPEN, 0.5)
                    self.plan_exec(self.T.lift_to, "retry_lower", LIFT_GRASP, 1.5)
                self.hold(a.retry_settle)
            elif k > 0:  # empty pinch (arm already out of the heap at the grasp height)
                self.hold(a.retry_settle)
            gx, gy, fb_target, off = self.attempt(self.cloth_pos(), tried)
            tried.append((gx, gy))
            res["attempts"] = k + 1
            res["attempts_log"].append({key: res.get(key) for key in (
                "miss", "stall", "weak_hold", "lift_slide_mm", "ipc_squeeze_N_closed", "ipc_fz_N_closed", "joint7_8_closed",
                "grasp_point",
                "grasp_yaw",
                "finger_bottom_at_grasp_minus_floor", "verts_between_fingers_open", "verts_between_fingers_closed",
                "touch_closed", "lift_lost_at_tip_dz", "lift_dzmean", "lifted_frac_above_rim", "lifted_tip_to_nearest_vertex",
                "lift_success") + (("plan_fail",) if self.T is not None else ())})
            if res["lift_success"]:
                break
        if a.stop_after == "lift":
            return res
        if a.skip_carry_on_fail and not res["lift_success"]:
            res["success"] = False
            res["carry_skipped"] = True
            return res

        # carry: waypoints from the grasp xy to the release xy, each as high as reachable
        # --release-dx/dy are offsets in the washer frame
        rx, ry = P.washer_to_base(a.release_dx, a.release_dy, WASHER_XY, math.radians(WASHER_YAW_DEG))
        wps = []
        if self.T is not None:  # planned: straight raise, then the tilted waypoints (all planned before moving)
            self.plan_exec(self.T.raise_, a.raise_s)
            p0, _ = self.AP.fk_tcp(self.T.q)
            wps = [(round(float(gx), 3), round(float(gy), 3), round(float(p0[2]), 3), 0, "keep")]
            self.plan_exec(self.T.carry, rx, ry, a.carry_s / a.n_wp)
            wps += self.T.carry_wps
        elif a.carry_mode == "gentle":
            wps = self.carry_gentle(gx, gy, rx, ry)
        for k in range(1, a.n_wp + 1 if a.carry_mode == "default" else 1):
            s = k / a.n_wp
            px, py = gx + s * (rx - gx), gy + s * (ry - gy)
            goal, z, tilt = self.ik_high(px, py, np.arange(a.carry_z, P.WASHER_H + 0.04, -0.05), LIFT_CARRY,
                                         tilts=(40, 50, 60))
            wps.append((round(float(px), 3), round(float(py), 3), round(float(z), 3), tilt))
            self.move(goal, a.carry_s / a.n_wp)
        res["carry_waypoints_xyz_tilt"] = wps
        self.hold(a.swing_s)
        x3 = self.cloth_pos()
        tip3 = self.tcp_world()
        tr = np.array(self.carry_track)
        self.carry_track = None
        res["carry_max_tip_to_nearest_vertex"] = float(tr[:, 0].max())
        res["carry_min_zmax_minus_tip"] = float((tr[:, 1] - tr[:, 2]).min())
        res["release_tip"] = [round(float(v), 3) for v in tip3]
        res["release_cloth_zmin"] = float(x3[:, 2].min())
        res["release_cloth_over_opening_frac"] = float(
            self.in_tub_xy(x3).mean())
        res["held_until_release"] = bool(tr[:, 0].max() < 0.03)
        res["pen_carry_end"] = self.penetration(x3)

        # release
        if self.T is None:
            self.gripper(OPEN, 0.6)
        else:
            self.p_grip("release", OPEN, 0.6)
        self.hold(2.0)
        x4 = self.cloth_pos()
        res["in_tub_frac_release+2s"] = float(self.in_tub(x4).mean())
        # retract the arm up/back towards the mast so the tub view is clear, then watch the tub
        if self.T is None:
            try:
                goal, _, _ = self.ik_high(-0.02, 0.10, (1.10, 1.0, 0.9), LIFT_CARRY, tilts=(0, 30, 50))
                self.move(goal, 1.5)
            except AssertionError:
                pass
        else:  # the old target (-0.02, 0.10) has no IK solution: planned way back to the standby fold
            try:
                self.plan_exec(self.T.to_standby, 1.5)
            except PlanningError as e:
                res["retract_plan_fail"] = str(e)[:300]
        self.view = self.tub_cam
        self.hold(a.post_s)
        x5 = self.cloth_pos()
        res["in_tub_frac_end"] = float(self.in_tub(x5).mean())
        res["in_basket_frac_end"] = float(self.in_basket(x5).mean())
        res["end_cloth_zmax"] = float(x5[:, 2].max())
        res["end_cloth_zmin_minus_tub_floor"] = float(x5[:, 2].min() - (P.WASHER_H - P.WASHER_TUB_DEPTH))
        res["end_cloth_centroid"] = [round(float(v), 3) for v in x5.mean(0)]
        res["pen_end"] = self.penetration(x5)
        res["success"] = bool(res["in_tub_frac_end"] >= 0.99)
        if self.proc:
            for _ in range(FPS):  # 1 s still of the tub view at the end
                self.frame()
        return res


def make_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="varies the drop pose -> the heap shape")
    ap.add_argument("--n", type=int, default=31, help="cloth grid vertices per side (31 -> 1 cm edges)")
    ap.add_argument("--mu", type=float, default=0.8, help="cloth friction_mu and robot (gripper, basket) friction")
    ap.add_argument("--washer-mu", type=float, default=0.5)
    ap.add_argument("--E", type=float, default=5e4)
    ap.add_argument("--rho", type=float, default=300.0)
    ap.add_argument("--bend", type=float, default=None, help="FEM.Cloth bending_stiffness (default None = no "
                    "bending term)")
    ap.add_argument("--thickness", type=float, default=0.001)
    ap.add_argument("--d-hat", type=float, default=0.001)
    ap.add_argument("--newton-iters", type=int, default=64)
    ap.add_argument("--newton-tol", type=float, default=None, help="IPC Newton velocity tolerance [m/s] "
                    "(libuipc default 0.05)")
    ap.add_argument("--eps-v", type=float, default=None, help="IPC friction eps_velocity [m/s]: slower relative "
                    "motion counts as static friction (libuipc default 0.01)")
    ap.add_argument("--semi-implicit", choices=["default", "on", "off"], default="default",
                    help="libuipc Newton semi-implicit termination (libuipc default on); off removes the towel drift "
                    "in a moving basket but makes moving steps about 4-5x slower")
    ap.add_argument("--semi-beta", type=float, default=None,
                    help="newton_semi_implicit_beta_tolerance (libuipc default 1e-3)")
    ap.add_argument("--cs-trans", type=float, default=None, help="IPC soft-constraint translation strength "
                    "(Genesis default 100)")
    ap.add_argument("--cs-rot", type=float, default=None, help="IPC soft-constraint rotation strength (default 100)")
    ap.add_argument("--contact-resistance", type=float, default=None, help="IPC contact stiffness (default 1e9)")
    ap.add_argument("--grip-kp", type=float, default=2000.0)
    ap.add_argument("--grip-kv-scale", type=float, default=1.0, help="joint7/8 kv multiplier (kv = scale * 2 sqrt(kp))")
    ap.add_argument("--arm-kv-scale", type=float, default=1.0, help="joint1-6 kv multiplier (kv = scale * 2 sqrt(kp))")
    ap.add_argument("--ff", choices=list(FF_MODES), default="none",
                    help="PD feed-forward on the lift + joint1-6 target (scripts/arm_control.py): grav = + tau_g / kp "
                         "(solver bias force, no contact force, no integral term; removes the ~12 mm static sag of the "
                         "finger bottom), grav+vel = also + kv / kp * planned velocity on joint1-6 (tracking lag)")
    ap.add_argument("--dt", type=float, default=0.01, help="sim dt [s] (move durations stay in seconds)")
    ap.add_argument("--grip-fmax", type=float, default=None, help="finger force limit [N] (default: URDF effort 10)")
    ap.add_argument("--yaw", type=float, default=90.0, help="gripper yaw [deg] for the vertical grasp")
    ap.add_argument("--yaw-mode", choices=["fixed", "ridge"], default="ridge",
                    help="ridge: closing axis perpendicular to the local fold at the grasp point")
    ap.add_argument("--grasp-r", type=float, default=0.05, help="grasp: highest vertex within this xy radius of "
                    "the heap centroid [m]")
    ap.add_argument("--depth", type=float, default=0.04, help="finger bottoms this far below the heap top [m]")
    ap.add_argument("--press", type=float, default=1.0, help="finger bottom target above the basket floor "
                    "[m]; >= 0.01 = use --depth below the heap top instead")
    ap.add_argument("--close-s", type=float, default=1.0)
    ap.add_argument("--pre-open", type=float, default=OPEN, help="joint7 opening before the descent (0.035 = full, "
                    "gap = 2 x this)")
    ap.add_argument("--descend-s", type=float, default=1.5, help="duration of the descent onto the heap [s]")
    ap.add_argument("--press-dz", type=float, default=0.0, help="> 0: descend this much further while closing "
                    "(over --close-s) [m]")
    ap.add_argument("--no-scoop", dest="scoop", action="store_false",
                    help="descend with the gripper fully open (pushes the cloth out of the gap, see docstring)")
    ap.add_argument("--scoop-close", type=float, default=0.008, help="joint7 reached at the end of the descent")
    ap.add_argument("--lift1-z", type=float, default=0.80, help="max TCP z for the vertical arm lift [m]")
    ap.add_argument("--lift1-s", type=float, default=2.0)
    ap.add_argument("--lift2-s", type=float, default=2.0)
    ap.add_argument("--carry-z", type=float, default=1.20, help="max TCP z tried at the carry waypoints [m]")
    ap.add_argument("--n-wp", type=int, default=3)
    ap.add_argument("--carry-s", type=float, default=3.0)
    ap.add_argument("--carry-mode", choices=["default", "gentle"], default="default",
                    help="gentle: straight up first, then the tilt grows gradually (--tilt-step) with the yaw kept")
    ap.add_argument("--raise-s", type=float, default=2.0, help="gentle carry: duration of the straight raise [s]")
    ap.add_argument("--tilt-step", type=float, default=12.0, help="gentle carry: planned tilt increase per waypoint "
                    "[deg]")
    ap.add_argument("--skip-carry-on-fail", action="store_true", help="end the run when the lift failed")
    ap.add_argument("--swing-s", type=float, default=1.0, help="hold over the opening before release [s]")
    ap.add_argument("--release-dx", type=float, default=0.0, help="release offset along the washer x (front) axis")
    ap.add_argument("--release-dy", type=float, default=0.0, help="release offset along the washer y axis")
    ap.add_argument("--settle", type=float, default=3.0, help="min settle time [s]")
    ap.add_argument("--settle-max", type=float, default=8.0, help="max settle time [s]")
    ap.add_argument("--still-tol", type=float, default=0.003, help="settled when no vertex moved more than this "
                    "in 0.5 s [m]")
    ap.add_argument("--drop-h", type=float, default=0.12, help="towel's lower edge height above the basket floor "
                    "at release [m]")
    ap.add_argument("--lean-max", type=float, default=8.0, help="max lean of the released towel [deg]")
    ap.add_argument("--post-s", type=float, default=2.5)
    ap.add_argument("--attempts", type=int, default=1, help="max pinch attempts (retry at another point after a "
                    "failed lift)")
    ap.add_argument("--miss-check", action="store_true", help="after closing: squeeze < --miss-n or a finger "
                    "without cloth contact = empty pinch -> open without lifting (retry with --attempts)")
    ap.add_argument("--miss-n", type=float, default=5.0, help="min IPC squeeze [N] of a real pinch")
    ap.add_argument("--floor-clear", type=float, default=0.003, help="finger bottom target at least this far above "
                    "the basket floor [m]")
    ap.add_argument("--min-bite", type=float, default=0.0, help="> 0: widen the grasp radius until the heap top is "
                    "floor-clear + this above the basket floor [m]")
    ap.add_argument("--stall-check", action="store_true", help="after closing: finger gap > --stall-gap and an "
                    "upward IPC force > --stall-fz (fingertips on the basket floor) -> treat as a miss, retry")
    ap.add_argument("--lift-slide-max", type=float, default=0.0, help="> 0: a lift whose pinched vertices slid more "
                    "than this [m] counts as failed (thin-edge hold) -> retry elsewhere")
    ap.add_argument("--stall-gap", type=float, default=0.005, help="stall: joint7 - joint8 after closing [m]")
    ap.add_argument("--stall-fz", type=float, default=5.0, help="stall: min upward IPC force on the fingers [N]")
    ap.add_argument("--retry-settle", type=float, default=1.5, help="wait before a retry [s]")
    ap.add_argument("--exclude-r", type=float, default=0.04, help="retry: min xy distance to earlier grasp points")
    ap.add_argument("--select", choices=["centroid", "hmap"], default="centroid",
                    help="centroid: highest vertex near the heap centroid; hmap: sharpest high ridge of the top "
                    "height map (closing axis across it)")
    ap.add_argument("--hmap-band", type=float, default=0.03, help="hmap: candidate cells within this of the "
                    "highest cell [m]")
    ap.add_argument("--tilt", type=float, default=0.0, help="tilt the gripper about the fold ridge [deg] (approach "
                    "the fold obliquely)")
    ap.add_argument("--stop-after", choices=["settle", "lift", "all"], default="all")
    ap.add_argument("--cam", choices=["wide", "close"], default="wide", help="close: basket close-up (debug)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None, help="result JSON path (default outputs/cloth_ipc_washer_s<seed>.json)")
    ap.add_argument("--save-obs", default=None, metavar="PATH", help="after the settle save an npz: cloth vertices, "
                    "basket camera rgb / depth / point cloud, intrinsics / extrinsics (adds a basket camera)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--planner", action="store_true", help="every arm / lift / gripper motion from a collision-free "
                    "path of scripts/arm_planner.py (see docstring); needs --no-scoop --carry-mode gentle")
    ap.add_argument("--planner-margin", type=float, default=0.01, help="--planner: arm-obstacle clearance [m]")
    ap.add_argument("--planner-room", action="store_true", help="--planner: also keep the arm clear of the house "
                    "(room collision boxes in the parked frame, arm_planner.ArmPlanner(with_room=True))")
    ap.add_argument("--planner-proc", choices=["subprocess", "inproc"], default="subprocess",
                    help="--planner: planning in a CPU-backend child process (default; GPU planning is 10-50x slower) "
                    "or in this process on the IPC backend (debugging)")
    ap.add_argument("--vmax", type=float, default=0.8, help="--planner: a planned segment lasts at least "
                    "joint arc / vmax [s] (rad/s, lift m/s)")
    ap.add_argument("--qlog-every", type=int, default=5, help="--planner: check the measured q every n steps")
    ap.add_argument("--grip-const", action="store_true", help="closing target 2 * grip-fmax / grip-kp beyond CLOSED: "
                    "the finger PD saturates at --grip-fmax (needs --grip-fmax)")
    return ap


def main(test_cls=None, ap=None):
    """Command line entry; a subclass script passes its Test subclass and its (extended) parser."""
    ap = ap or make_parser()
    args = ap.parse_args()
    if args.planner and (args.scoop or args.press_dz > 0 or args.tilt != 0 or args.carry_mode != "gentle"):
        ap.error("--planner plans the vertical grasp with the gripper fixed during the descent: use --no-scoop, "
                 "--press-dz 0, --tilt 0, --carry-mode gentle")
    if args.grip_const and not args.grip_fmax:
        ap.error("--grip-const needs --grip-fmax")
    global DT
    DT = args.dt

    out = args.out or str(ROOT / "outputs" / ("cloth_ipc_washer.mp4" if args.seed == 0 else
                                              f"cloth_ipc_washer_s{args.seed}.mp4"))
    t = (test_cls or Test)(args)
    if not args.no_video:
        t.start_video(out)
    t0 = time.time()
    t.res = {"seed": args.seed, "washer_yaw_deg": round(WASHER_YAW_DEG, 3)}
    try:
        t.run()
    except Exception as e:  # keep the partial numbers
        t.res["error"] = repr(e)[:300]
    finally:
        t.finish_video()
    res = t.res
    wall = time.time() - t0
    if t.T is not None:
        rows = t.plan_rows
        ok_rows = [r for r in rows if r.get("ok") and "min_clear_plan_mm" in r]
        res["planner"] = dict(build_s=round(t.planner_build_s, 1), standby=np.round(t.T.standby, 4).tolist(),
                              margin=args.planner_margin, plan_total_s=round(sum(r["plan_s"] for r in rows), 1),
                              plan_failed=[r["name"] for r in rows if not r.get("ok")],
                              plan_min_clear_mm=min((r["min_clear_plan_mm"] for r in ok_rows), default=None),
                              skipped=t.T.skipped)
        res["plan_segments"] = rows
        t1 = time.time()
        try:
            res["measured_segments"], worst = t.measured_stats()
        except RuntimeError as e:   # planner process gone: keep the rest of the numbers
            res["measured_error"] = repr(e)[:300]
            res["measured_segments"], worst = [], (0.0, "")
        finally:
            t.T.shutdown()
        res["planner"]["measured_check_s"] = round(time.time() - t1, 1)
        res["measured_max_pen_mm"] = round(worst[0] * 1e3, 2)
        res["measured_max_pen_pair"] = worst[1]
        res["measured_min_clear_mm"] = min((r["min_clear_mm"] for r in res["measured_segments"]
                                            if r["min_clear_mm"] is not None), default=None)
    res["grip_close_target"] = round(float(t.close_target), 4)
    res["nan"] = t.nan
    res["max_vertex_speed"] = t.max_speed
    res["max_edge_stretch_ratio"] = t.max_stretch
    res["build_s"] = round(t.build_s, 1)
    res["steps"] = t.nstep
    res["sim_steps_per_s"] = round(t.nstep / max(t.sim_time, 1e-9), 1)
    res["wall_steps_per_s_incl_render"] = round(t.nstep / wall, 1)
    res["gpu_mem_mib_process"] = gpu_mem_mib()
    print(f"=== IPC basket -> washer: {vars(args)} ===")
    for k, v in res.items():
        print(f"{k}: {round(v, 4) if isinstance(v, float) else v}")
    js = Path(args.json) if args.json else ROOT / "outputs" / f"cloth_ipc_washer_s{args.seed}.json"
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(res, indent=1, default=str))
    if not args.no_video:
        print(f"video: {out}")


if __name__ == "__main__":
    main()
