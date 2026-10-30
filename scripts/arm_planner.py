"""Collision-aware joint trajectories for the PiPER arm of house_robot (plain Genesis, no IPC, runs in .venv).

The planning scene holds only collision shapes: floor plane, (with_room=True) the room's collision boxes, the
robot (assets/robot/house_robot.urdf, fixed base; chassis, mast and basket are boxes of its `base_link`) and the
washer (assets/washer/washer.urdf, a separate fixed entity, its open lid merged into washer_body). It is built once
per process; the planner moves the robot kinematically
(set_qpos), it never steps the simulation.

Genesis 1.4.4 `RigidEntity.plan_path` (genesis/utils/path_planning.py, RRTConnect) as used here:
- Samples every DOF of the entity uniformly inside its DOF limits (`_get_dofs_limit`, `_kernel_rrt_connect_step1`),
  i.e. also base x / y (+-20 m) and yaw. There is no DOF subset argument, so DOFs that must stay put are frozen by
  narrowing their limits to the start value (`set_dofs_limit`) for the call and restoring them afterwards.
- A state is invalid if the full-scene narrow phase (`rigid_solver._kernel_detect_collision`) reports any contact
  with penetration >= 1e-5 m (f32) that involves a geom of the planned entity (`_func_check_collision`). Contacts with
  other entities (the washer) therefore count, self-collision counts if `enable_self_collision`.
- Geom pairs that are already in contact at the start OR the goal are ignored for the whole query
  (`get_exclude_geom_pairs`). A start inside an obstacle is silently accepted, so start and goal are validated here
  first.
- The collider drops self-collision pairs that overlap at qpos0 unless `enable_neutral_collision=True`
  (collider.py, warning "Filtered out geometry pairs ... (qpos0)"). For house_robot those are (2, 11) and (2, 12) =
  mast vs link2 / link3, because the PiPER zero pose lies inside the mast. This scene therefore enables neutral
  collision; the only pairs touching at rest are link6/7/8 (gripper), which are excluded explicitly.
- Edges are not swept: only the RRT nodes (steps of `resolution` per DOF, L-inf) and the final `num_waypoints`
  linear interpolation are checked. `smooth_path` = 10 random shortcut attempts. `timeout` is a wall-clock check per
  iteration, `max_nodes` caps the trees, `max_retry` re-plans failed queries. Returns [num_waypoints, n_qs].

On top of that, this module
- validates every path densely (joint step <= `check_step`) and requires a clearance `margin` between the arm links
  and the obstacle boxes (analytic box SDF of points sampled on the arm collision hulls); RRT paths that come closer
  are rejected and re-planned, then shortcut with the same check,
- tries the straight joint-space interpolation first (RRT only if it collides),
- solves IK from several initial values and drops colliding solutions.

DOFs: the base (x, y, yaw) is parked during manipulation and stays fixed in every query (navigation is a separate
problem; the room is modelled only with with_room=True). The lift is fixed per query by default: it is slow
(0.2 m/s), the task uses it only for a pure vertical move (lift segment), and leaving it out keeps the RRT space
6-D; `free_lift=True` adds it (inside its limits) as a fallback when the 6-D query fails.
The gripper is fixed per query.

TransferSeq (bottom of this file) plans the basket -> washer pick-and-place segment by segment (standby, pre-open,
hover, descent, close, lift, lift joint, raise, tilted carry waypoints, release, back to standby, and the retry moves
after an empty pinch or a failed lift). cloth_ipc_washer_test.py --planner and arm_planner_test.py both call it, so
the CPU check (arm_planner_test.py --grasp-set) exercises the planning code of the IPC run. The planner scene is
never stepped.

Planning is far slower on Genesis' GPU backend than on the CPU one (346-2068 s vs 14-59 s per grasp point, identical
plans), and gs.init runs once per process, so the IPC run (gs.gpu) must not plan in its own process. PlannerClient (the
bottom of this file) runs ArmPlanner + TransferSeq on the CPU backend in a child process (a subprocess, not a fork: the
IPC parent holds a CUDA context) behind a socketpair and mirrors the TransferSeq state the caller reads (segs, q,
standby, target, carry_wps, skipped); `python -c "import arm_planner; arm_planner.serve(fd, ...)"` is the child.
mode="inproc" keeps the planner in the calling process (debugging).
"""
import atexit
import math
import os
import pickle
import socket
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))
import params as PRM  # noqa: E402

NAMES = ["base_x_joint", "base_y_joint", "base_yaw_joint", "lift_joint"] + [f"joint{i}" for i in range(1, 9)]
I_LIFT, I_ARM, I_GRIP = 3, slice(4, 10), slice(10, 12)
ARM_LINKS = tuple(f"link{i}" for i in range(1, 9))
GRIPPER_LINKS = ("link6", "link7", "link8")
TCP = 0.15                        # link6 origin -> fingertip centre along link6 z [m]
# washer pose in the parked-base frame (= house layout, params.py): opening centre; yaw about z (front towards the
# robot, lid on the far side)
WASHER_POS = (*PRM.WASHER_BASE_XY, 0.003)
WASHER_YAW_DEG = PRM.WASHER_BASE_YAW_DEG
WASHER_PARTS = (["block"] + ["shell"] * 4 + ["tub"] * 4 + ["top"] * 4 + ["tub_floor", "lid"])  # gen_washer.py order


# ---------------- orientation helpers (same conventions as cloth_grasp_test / cloth_ipc_washer_test) ----------------
def wxyz(r):
    q = r.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def rot(q_wxyz):
    q = np.asarray(q_wxyz, float)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def quat_down(yaw_deg=0.0):
    """wxyz: tool axis (link6 z) pointing down, gripper yawed by yaw_deg about world z."""
    return wxyz(Rotation.from_euler("z", yaw_deg, degrees=True) * Rotation.from_euler("y", 180, degrees=True))


def quat_toward(az, tilt_deg):
    """wxyz: pointing down, then tilted by tilt_deg so the tool points outward along azimuth az [rad]."""
    return wxyz(Rotation.from_euler("z", az) * Rotation.from_euler("y", -tilt_deg, degrees=True)
                * Rotation.from_euler("y", 180, degrees=True))


def quat_tilt_keep_yaw(az, yaw_deg, tilt_deg):
    """wxyz: quat_down(yaw_deg) tilted by tilt_deg so the tool axis leans outward along azimuth az (gripper yaw
    kept), as cloth_ipc_washer_test.quat_tilt_keep_yaw."""
    u = np.array([-math.sin(az), math.cos(az), 0.0])
    return wxyz(Rotation.from_rotvec(-math.radians(tilt_deg) * u) * rot(quat_down(yaw_deg)))


def hover_point(g, quat, z):
    """Point on the approach line through g (along the tool axis) at height z."""
    d = rot(quat).apply([0, 0, 1.0])
    return np.asarray(g, float) - d * (z - g[2]) / -d[2]


# ---------------- results ----------------
@dataclass
class PlanResult:
    ok: bool
    path: np.ndarray = None           # [N, 12] joint positions (NAMES order), consecutive max step <= check_step
    method: str = "fail"              # direct | rrt | rrt_lift | cartesian | fail
    plan_s: float = 0.0
    rrt_calls: int = 0
    reason: str = ""
    min_clearance: float = float("nan")
    info: dict = field(default_factory=dict)


def densify(path, step):
    """Linear joint-space resampling so no DOF moves more than `step` between consecutive rows."""
    path = np.asarray(path, float)
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(math.ceil(np.abs(b - a).max() / step)))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out)


def retime(path, dur, dt):
    """Resample a path to a trajectory of duration `dur` [s] at `dt`, smoothstep in joint-space arc length."""
    path = np.asarray(path, float)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    n = max(1, int(round(dur / dt)))
    if s[-1] < 1e-9:
        return np.repeat(path[-1:], n, axis=0)
    a = np.arange(1, n + 1) / n
    a = a * a * (3 - 2 * a)
    return np.stack([np.interp(a * s[-1], s, path[:, j]) for j in range(path.shape[1])], axis=1)


def _parse_boxes(urdf, link_name):
    """Collision boxes (xyz, rpy, size) of `link_name` in a URDF, plus fixed-joint children merged into it."""
    root = ET.parse(urdf).getroot()
    out = []

    def boxes_of(ln, T):
        for lk in root.findall("link"):
            if lk.get("name") != ln:
                continue
            for c in lk.findall("collision"):
                b = c.find("geometry/box")
                if b is None:
                    continue
                o = c.find("origin")
                xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
                rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
                Tl = np.eye(4)
                Tl[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
                Tl[:3, 3] = xyz
                out.append((T @ Tl, np.array([float(v) for v in b.get("size").split()])))
        for j in root.findall("joint"):
            if j.get("type") == "fixed" and j.find("parent").get("link") == ln:
                o = j.find("origin")
                Tj = np.eye(4)
                Tj[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in o.get("rpy", "0 0 0").split()]).as_matrix()
                Tj[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
                boxes_of(j.find("child").get("link"), T @ Tj)

    boxes_of(link_name, np.eye(4))
    return out


class ArmPlanner:
    def __init__(self, washer_pos=WASHER_POS, washer_yaw_deg=WASHER_YAW_DEG, margin=0.01,
                 grip_basket_margin=0.002, rrt_inflate=0.015,
                 goal_margin=0.015, check_step=0.01, n_surface=800, backend="cpu", scene_hook=None,
                 init_genesis=True, vis_options=None, seed=0, with_room=False, room_margin=None, room_reach=2.0):
        """margin: required clearance [m] between arm links (link1-8) and the mast / chassis / basket / carriage /
        washer boxes. grip_basket_margin: clearance for the gripper links (link6/7/8) vs the basket boxes (the
        fingers enter the basket and go down to the heap near its floor). check_step: joint step [rad or m] of the
        dense path check. rrt_inflate: the washer boxes and the mast are duplicated, inflated by this much, as
        invisible collision-only boxes that only the RRT sees (its validity test is exact contact); it is larger
        than `margin` so the chords between RRT nodes stay outside `margin`. goal_margin: IK goals with at least
        this slack are preferred (a start/goal already touching an inflated box makes plan_path ignore that pair).
        scene_hook(scene) is called before build (e.g. to add a camera).
        with_room: also model the house (the collision boxes of assets/room/room.urdf: walls, fridge + microwave,
        wall cabinet, counter, ... in the parked-base frame via house_frame.T_room_to_park). Only the arm links
        (link1-8) are checked against it (SDF margin `room_margin`, default = margin; the RRT sees the same boxes
        inflated by rrt_inflate); the chassis / basket / mast stay ~5 cm from the corridor walls by design, and
        driving past the walls is a separate sweep (scripts/arm_drive_check.py). room_reach: only room boxes whose
        footprint is within this distance [m] of the base origin are added (None: all); floor slabs are skipped
        (the plane is the floor)."""
        import genesis as gs
        self.gs = gs
        if init_genesis:
            gs.init(backend=gs.cpu if backend == "cpu" else gs.gpu, logging_level="warning", seed=seed)
        self.margin, self.grip_basket_margin, self.check_step = margin, grip_basket_margin, check_step
        self.goal_margin = goal_margin
        self.room_margin = margin if room_margin is None else room_margin
        self.washer_xy, self.washer_yaw = tuple(washer_pos[:2]), math.radians(washer_yaw_deg)
        self.rng = np.random.default_rng(seed)
        kw = dict(vis_options=vis_options) if vis_options is not None else {}
        sc = self.scene = gs.Scene(
            rigid_options=gs.options.RigidOptions(enable_self_collision=True, enable_neutral_collision=True,
                                                  max_collision_pairs=400),
            show_viewer=False, **kw)
        sc.add_entity(gs.morphs.Plane())
        self.robot = sc.add_entity(gs.morphs.URDF(file=str(ROOT / "assets/robot/house_robot.urdf"), fixed=True))
        self.washer = sc.add_entity(gs.morphs.URDF(file=str(ROOT / "assets/washer/washer.urdf"),
                                                   pos=tuple(washer_pos), euler=(0, 0, washer_yaw_deg), fixed=True))
        # inflated copies (collision only, invisible) of the washer boxes and the mast: the RRT's own validity test
        # is exact contact, so these make it keep `margin` away; check() ignores them and uses the exact SDF margin
        self.infl = []
        if rrt_inflate > 0:
            Tw = np.eye(4)
            Tw[:3, :3] = Rotation.from_euler("z", washer_yaw_deg, degrees=True).as_matrix()
            Tw[:3, 3] = washer_pos
            parts = [("washer_" + WASHER_PARTS[i], Tw @ T, size) for i, (T, size) in
                     enumerate(_parse_boxes(ROOT / "assets/washer/washer.urdf", "washer_body"))]
            parts += [(n, T, size) for (T, size) in _parse_boxes(ROOT / "assets/robot/house_robot.urdf", "base_link")
                      for n in ["mast"] if size[2] > 0.6]   # base at the origin, yaw 0 (parked, frozen)
            for n, T, size in parts:
                e = sc.add_entity(gs.morphs.Box(pos=tuple(T[:3, 3]), quat=tuple(wxyz(Rotation.from_matrix(T[:3, :3]))),
                                                size=tuple(size + 2 * rrt_inflate), fixed=True, visualization=False))
                self.infl.append((e, "infl_" + n))
        self.room = []   # (entity, name) of the exact room boxes (collision only, invisible)
        if with_room:
            for k, (T, size) in enumerate(self._room_boxes(room_reach)):
                quat = tuple(wxyz(Rotation.from_matrix(T[:3, :3])))
                self.room.append((sc.add_entity(gs.morphs.Box(pos=tuple(T[:3, 3]), quat=quat, size=tuple(size),
                                                              fixed=True, visualization=False)), f"room_{k}"))
                if rrt_inflate > 0:
                    e = sc.add_entity(gs.morphs.Box(pos=tuple(T[:3, 3]), quat=quat, size=tuple(size + 2 * rrt_inflate),
                                                    fixed=True, visualization=False))
                    self.infl.append((e, f"infl_room_{k}"))
        if scene_hook is not None:
            scene_hook(sc)
        sc.build()
        self.rs = sc.sim.rigid_solver
        for i, n in enumerate(NAMES):  # qpos index == dof index == NAMES position (no free joint)
            j = self.robot.get_joint(n)
            assert j.dofs_idx_local[0] == i and j.qs_idx_local[0] == i, n
        self.link6 = self.robot.get_link("link6")
        lo, hi = self.robot.get_dofs_limit()
        self.lim_lo, self.lim_hi = lo.cpu().numpy().copy(), hi.cpu().numpy().copy()
        self._name_geoms()
        for e, n in self.infl:   # the parsed boxes must coincide with the Genesis geoms they inflate
            gp = e.geoms[0].get_pos().cpu().numpy().reshape(-1)
            src = [g for g in self.rs.geoms if self.gnames[g.idx] == n[5:]]
            assert min(np.linalg.norm(g.get_pos().cpu().numpy().reshape(-1) - gp) for g in src) < 1e-4, n
        self._prepare_sdf(n_surface)

    @staticmethod
    def _room_boxes(reach):
        """Collision boxes (T in the parked-base frame, size) of the room URDF; floor slabs dropped, boxes whose
        footprint is farther than `reach` from the base origin dropped."""
        from house_frame import T_room_to_park
        Tp = T_room_to_park()
        out = []
        for T, size in _parse_boxes(ROOT / "assets/room/room.urdf", "room"):
            T = Tp @ T
            hz = np.abs(T[:3, :3]) @ size / 2
            if T[2, 3] + hz[2] <= 1e-3:
                continue
            if reach is not None:   # distance origin -> the footprint's axis-aligned box
                d = np.maximum(np.abs(T[:2, 3]) - hz[:2], 0.0)
                if np.linalg.norm(d) > reach:
                    continue
            out.append((T, size))
        return out

    # ---------------- geometry bookkeeping ----------------
    def _name_geoms(self):
        names = []
        wgeoms = list(self.washer.geoms)
        assert len(wgeoms) == len(WASHER_PARTS), "washer.urdf changed: update WASHER_PARTS"
        infl = {id(e): n for e, n in self.infl}
        infl.update({id(e): n for e, n in self.room})
        for g in self.rs.geoms:
            if id(g.entity) in infl:
                names.append(infl[id(g.entity)])
                continue
            if g.entity is self.washer:
                names.append("washer_" + WASHER_PARTS[wgeoms.index(g)])
                continue
            n = g.link.name
            if n == "base_link":
                v = np.asarray(g.init_verts)
                h = (v.max(0) - v.min(0)) / 2
                n = "chassis" if h[0] > 0.2 and h[2] > 0.09 else ("mast" if h[2] > 0.3 else
                                                                    ("basket_floor" if h[2] < 0.01 else "basket_wall"))
            names.append(n)
        self.gnames = names
        self.arm_geoms = [g.idx for g in self.rs.geoms if names[g.idx] in ARM_LINKS]
        # analytic SDF only for box geoms (the PiPER base mesh merged into `carriage` is left to Genesis' contacts)
        self.obst_geoms = [g.idx for g in self.rs.geoms if g.type == self.gs.GEOM_TYPE.BOX and
                           (names[g.idx] in ("chassis", "mast", "basket_floor", "basket_wall", "carriage")
                            or names[g.idx].startswith(("washer", "room")))]

    def _prepare_sdf(self, n_surface):
        import trimesh
        self.arm_pts = {}
        for gi in self.arm_geoms:
            g = self.rs.geoms[gi]
            v, f = np.asarray(g.init_verts, float), np.asarray(g.init_faces)
            s = trimesh.sample.sample_surface(trimesh.Trimesh(v, f, process=False), n_surface, seed=0)[0]
            self.arm_pts[gi] = np.concatenate([v, s])
        self.obst_box = {}
        for gi in self.obst_geoms:
            v = np.asarray(self.rs.geoms[gi].init_verts, float)
            self.obst_box[gi] = ((v.max(0) - v.min(0)) / 2, (v.max(0) + v.min(0)) / 2)

    def pair_margin(self, link, obst):
        if link in GRIPPER_LINKS and obst.startswith("basket"):
            return self.grip_basket_margin
        if obst.startswith("room"):
            return self.room_margin
        return self.margin

    # ---------------- state checks ----------------
    def set_q(self, q):
        self.robot.set_qpos(np.asarray(q, float), zero_velocity=True)

    def contacts(self, q):
        """Genesis narrow-phase contacts involving the robot at q: list of (name_a, name_b, penetration [m]), the
        gripper-internal pairs (link6/7/8 touch at rest) left out."""
        self.set_q(q)
        self.rs._kernel_detect_collision()
        c = self.robot.get_contacts()
        out = {}
        for a, b, p in zip(c["geom_a"].cpu().numpy().ravel(), c["geom_b"].cpu().numpy().ravel(),
                           c["penetration"].cpu().numpy().ravel()):
            na, nb = self.gnames[a], self.gnames[b]
            if (na in GRIPPER_LINKS and nb in GRIPPER_LINKS) or na.startswith("infl_") or nb.startswith("infl_"):
                continue
            key = tuple(sorted((na, nb)))
            out[key] = max(out.get(key, 0.0), float(p))
        return out

    def sdf(self, q=None, reach=0.05):
        """Signed distance [m] (negative = depth inside) of the arm collision hulls (vertices + surface samples) to
        each obstacle box, for the (arm link, obstacle) pairs closer than `reach`; the robot must be at q."""
        if q is not None:
            self.set_q(q)
        gp = self.rs.get_geoms_pos().cpu().numpy().reshape(-1, 3)
        gq = self.rs.get_geoms_quat().cpu().numpy().reshape(-1, 4)
        gR = Rotation.from_quat(gq[:, [1, 2, 3, 0]]).as_matrix()
        out = {}
        obs = []
        for oi in self.obst_geoms:
            h, cc = self.obst_box[oi]
            c = gR[oi] @ cc + gp[oi]
            obs.append((oi, h, cc, c, np.linalg.norm(h)))
        for ai in self.arm_geoms:
            w = self.arm_pts[ai] @ gR[ai].T + gp[ai]
            wc = w.mean(0)
            wr = np.linalg.norm(w - wc, axis=1).max()
            ln = self.gnames[ai]
            for oi, h, cc, c, rr in obs:
                on = self.gnames[oi]
                if np.linalg.norm(wc - c) > wr + rr + reach:
                    continue
                loc = (w - gp[oi]) @ gR[oi] - cc
                d = np.abs(loc) - h
                s = float((np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(1), 0)).min())
                if s < reach:
                    key = (ln, on)
                    out[key] = min(out.get(key, np.inf), s)
        return out

    def check(self, q, margin=True):
        """(ok, reason, min clearance [m] within 5 cm or inf): no Genesis contact (except inside the gripper) and, if
        margin, every (arm link, obstacle) clearance >= its pair margin. self.last_slack = min over pairs of
        clearance - pair margin + self.margin (comparable to self.margin)."""
        con = {k: p for k, p in self.contacts(q).items() if p >= 1e-5}
        if con:
            k = max(con, key=con.get)
            self.last_slack = -con[k]
            return False, f"contact {k[0]}-{k[1]} {con[k] * 1e3:.1f}mm", -con[k]
        d = self.sdf(None, reach=0.05)
        cmin = min(d.values()) if d else np.inf
        self.last_slack = min([s - self.pair_margin(*k) + self.margin for k, s in d.items()] or [np.inf])
        if margin:
            for (ln, on), s in d.items():
                if s < self.pair_margin(ln, on):
                    return False, f"clearance {ln}-{on} {s * 1e3:.1f}mm", cmin
        return True, "", cmin

    def clearance_stats_many(self, qs):
        """clearance_stats for a list of joint vectors -> list of (penetration, clearance, worst pair)."""
        return [self.clearance_stats(q) for q in qs]

    def clearance_stats(self, q):
        """(penetration [m] >= 0, min clearance [m] within 5 cm or inf, worst pair) of the arm at q: max of the
        Genesis contact depths (gripper-internal and link1-carriage pairs left out, as arm_planner_test.measure) and
        of the box-SDF depths. For replaying measured joint positions."""
        con = {k: p for k, p in self.contacts(q).items() if p > 0 and k != ("carriage", "link1")}
        d = self.sdf(None, reach=0.05)
        pen, worst = 0.0, ""
        for (a, b), p in con.items():
            if p > pen:
                pen, worst = p, f"{a}-{b}"
        for (ln, on), s in d.items():
            if -s > pen:
                pen, worst = -s, f"{ln}-{on}"
        return pen, (min(d.values()) if d else np.inf), worst

    def path_valid(self, path, margin=True):
        """Dense check of a joint path (linear between rows). Returns (ok, index of the first bad dense row, reason,
        min clearance [m] over the path)."""
        cmin = np.inf
        for k, q in enumerate(densify(path, self.check_step)):
            ok, why, c = self.check(q, margin)
            cmin = min(cmin, c)
            if not ok:
                return False, k, why, cmin
        return True, -1, "", cmin

    def segment_valid(self, qa, qb, margin=True):
        """Dense check of the straight joint segment, samples in bisection order (early exit on the first bad one)."""
        dense = densify([qa, qb], self.check_step)[1:]
        n = len(dense)
        order, seen, step = [], set(), 1 << max(0, int(math.ceil(math.log2(max(n, 1)))))
        while step >= 1:
            for k in range(step - 1, n, step):
                if k not in seen:
                    seen.add(k)
                    order.append(k)
            step //= 2
        return all(self.check(dense[k], margin)[0] for k in order)

    # ---------------- kinematics ----------------
    def fk_tcp(self, q):
        self.set_q(q)
        p = self.link6.get_pos().cpu().numpy().reshape(-1)
        qw = self.link6.get_quat().cpu().numpy().reshape(-1)
        return p + rot(qw).apply([0, 0, TCP]), qw

    def link1_pos(self, q):
        self.set_q(q)
        return self.robot.get_link("link1").get_pos().cpu().numpy().reshape(-1)

    def _ik1(self, q_init, tcp, quat, seed, iters=200, max_samples=50):
        """One Genesis IK call. max_samples > 1 lets Genesis restart from random joint values when the (tight)
        tolerance is not met, which jumps IK branches; path following uses max_samples=1."""
        qs, err = self.robot.inverse_kinematics(
            link=self.link6, pos=np.asarray(tcp, float), quat=quat, local_point=(0, 0, TCP),
            dofs_idx_local=list(range(4, 10)), init_qpos=np.asarray(q_init, float), max_solver_iters=iters,
            max_samples=max_samples, seed=seed, return_error=True)
        q = np.asarray(q_init, float).copy()
        q[I_ARM] = qs.cpu().numpy().reshape(-1)[I_ARM]
        e = err.cpu().numpy().reshape(-1)
        return q, float(np.abs(e[:3]).max()), float(np.abs(e[3:]).max())

    def ik_solutions(self, tcp, quat, q_ref, lift=None, grip=None, n_random=16, pos_tol=0.003, rot_tol=0.03,
                     margin=True, max_keep=6):
        """IK of the fingertip centre (link6 + TCP) from several initial values (q_ref, joint1 aimed at the target
        with a few elbow shapes, random values inside the limits). Solutions that collide (or violate the margin)
        are dropped; the rest are sorted by joint distance to q_ref. Returns (list of q, stats)."""
        q_ref = np.asarray(q_ref, float).copy()
        if lift is not None:
            q_ref[I_LIFT] = lift
        if grip is not None:
            q_ref[I_GRIP] = (grip, -grip)
        b = self.link1_pos(q_ref)
        az = math.atan2(tcp[1] - b[1], tcp[0] - b[0])
        inits = [q_ref[I_ARM].copy()]
        for j1 in (az, az + math.pi, az - math.pi):
            if abs(j1) > 2.618:
                continue
            inits += [np.array([j1, 0.5, -0.8, 0, 0.5, 0]), np.array([j1, 1.0, -1.2, 0, 0.6, 0]),
                      np.array([j1, 1.5, -1.0, 0, 0.8, 0]), np.array([j1, 0.3, -0.3, 0, 0.3, 0]),
                      np.array([j1, 2.0, -1.8, 0, 0.0, 0])]
        lo, hi = self.lim_lo[I_ARM], self.lim_hi[I_ARM]
        inits += list(lo + self.rng.random((n_random, 6)) * (hi - lo))
        sols, n_conv, n_coll, why = [], 0, 0, {}
        for k, a in enumerate(inits):
            q0 = q_ref.copy()
            q0[I_ARM] = a
            q, ep, er = self._ik1(q0, tcp, quat, seed=k)
            if ep > pos_tol or er > rot_tol:
                continue
            n_conv += 1
            if any(np.abs(q - s).max() < 0.03 for s, _ in sols):
                continue
            ok, w, _ = self.check(q, margin)
            if not ok:
                n_coll += 1
                why[w.split(" ")[1]] = why.get(w.split(" ")[1], 0) + 1
                continue
            sols.append((q, self.last_slack))
        wts = np.r_[np.zeros(4), [1.5, 1.5, 1.0, 0.5, 0.5, 0.5], np.zeros(2)]
        # solutions with goal_margin slack first, each group by joint distance to q_ref
        sols.sort(key=lambda s: (s[1] < self.goal_margin, float(np.sum(wts * np.abs(s[0] - q_ref)))))
        return [q for q, _ in sols[:max_keep]], dict(n_init=len(inits), n_converged=n_conv, n_colliding=n_coll,
                                                      colliding=why,
                                                      slack_mm=[round(float(c) * 1e3, 1) for _, c in sols[:max_keep]])

    # ---------------- planning ----------------
    def _rrt(self, q_start, q_goal, free_lift, timeout, max_nodes, resolution, num_waypoints):
        """One plan_path call with the non-planned DOFs frozen through their limits."""
        free = np.zeros(12, bool)
        free[I_ARM] = True
        if free_lift:
            free[I_LIFT] = True
        lo = np.where(free, self.lim_lo, q_start - 1e-6)
        hi = np.where(free, self.lim_hi, q_start + 1e-6)
        q_goal = np.where(free, q_goal, q_start)
        self.robot.set_dofs_limit(lo, hi)
        try:
            path, ok = self.robot.plan_path(qpos_goal=q_goal, qpos_start=q_start, max_nodes=max_nodes,
                                            resolution=resolution, timeout=timeout, max_retry=0, smooth_path=False,
                                            num_waypoints=num_waypoints, planner="RRTConnect",
                                            return_valid_mask=True)
        finally:
            self.robot.set_dofs_limit(self.lim_lo, self.lim_hi)
        return path.cpu().numpy(), bool(ok)

    def shortcut(self, path, iters=15):
        """Greedy (farthest reachable knot first) then random shortcutting of a polyline with the dense margin check
        (the planner's own smoothing only knows exact contact)."""
        pts = [p for p in path]
        # drop collinear interior rows of the interpolated polyline
        keep = [pts[0]]
        for k in range(1, len(pts) - 1):
            d0, d1 = pts[k] - keep[-1], pts[k + 1] - pts[k]
            if np.linalg.norm(d0) < 1e-9:
                continue
            c = d0 @ d1 / (np.linalg.norm(d0) * max(np.linalg.norm(d1), 1e-12))
            if c < 0.999:
                keep.append(pts[k])
        keep.append(pts[-1])
        pts = keep
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            for j in range(len(pts) - 1, i, -1):
                if j == i + 1 or self.segment_valid(pts[i], pts[j]):
                    out.append(pts[j])
                    i = j
                    break
        pts = out
        for _ in range(iters):
            if len(pts) < 3:
                break
            i, j = sorted(self.rng.choice(len(pts), 2, replace=False))
            if j - i < 2:
                continue
            if self.segment_valid(pts[i], pts[j]):
                pts = pts[: i + 1] + pts[j:]
        return np.array(pts)

    def plan(self, q_start, q_goal, free_lift=False, timeout=5.0, tries=8, max_nodes=3000, resolution=0.1,
             num_waypoints=200, allow_direct=True):
        """Collision-free joint path q_start -> q_goal (both full 12-vectors). Direct interpolation if it is valid,
        else RRTConnect (6-D arm, + lift if free_lift) with the dense margin check and shortcutting."""
        t0 = time.time()
        q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
        for nm, q in (("start", q_start), ("goal", q_goal)):
            ok, why, _ = self.check(q)
            if not ok:
                return PlanResult(False, reason=f"{nm} invalid: {why}", plan_s=time.time() - t0)
        if not free_lift and (abs(q_goal[I_LIFT] - q_start[I_LIFT]) > 1e-6 or
                              np.abs(q_goal[:3] - q_start[:3]).max() > 1e-6):
            allow_rrt = False   # lift / base change with a frozen lift: only a direct move is possible
        else:
            allow_rrt = True
        if allow_direct:
            ok, k, why, cmin = self.path_valid([q_start, q_goal])
            if ok:
                return PlanResult(True, densify([q_start, q_goal], self.check_step), "direct", time.time() - t0, 0,
                                  min_clearance=cmin)
            direct_why = why
        else:
            direct_why = "skipped"
        if not allow_rrt:
            return PlanResult(False, reason=f"direct invalid ({direct_why}), lift/base change needs free_lift",
                              plan_s=time.time() - t0)
        fails = []
        for n in range(tries):
            t1 = time.time()
            path, ok = self._rrt(q_start, q_goal, free_lift, timeout, max_nodes, resolution, num_waypoints)
            if not ok:   # timeout / max_nodes / plan_path's own final check of the interpolated path
                fails.append(f"plan_path failed {time.time() - t1:.1f}s")
                continue
            t_rrt = time.time() - t1
            path[0], path[-1] = q_start, q_goal
            ok, k, why, _ = self.path_valid(path)
            if not ok:
                fails.append(why)
                continue
            t2 = time.time()
            sp = self.shortcut(path)
            t_sc = time.time() - t2
            ok, k, why, cmin = self.path_valid(sp)
            if not ok:   # shortcut is margin-checked, should not happen; keep the raw path then
                sp = path
                ok, k, why, cmin = self.path_valid(sp)
            return PlanResult(True, densify(sp, self.check_step), "rrt_lift" if free_lift else "rrt",
                              time.time() - t0, n + 1, min_clearance=cmin,
                              info=dict(direct=direct_why, rejected=fails, knots=len(sp), last_rrt_s=round(t_rrt, 2),
                                        shortcut_s=round(t_sc, 2)))
        return PlanResult(False, reason=f"direct: {direct_why}; rrt: {fails}", plan_s=time.time() - t0,
                          rrt_calls=tries)

    def plan_to_pose(self, q_start, tcp, quat, lift=None, grip=None, k_goals=4, goal_filter=None, **kw):
        """IK (several initial values, colliding solutions dropped) + plan() to the nearest goal that works.
        goal_filter(q_goal) -> bool can reject IK branches (e.g. one from which the next straight move fails)."""
        t0 = time.time()
        sols, st = self.ik_solutions(tcp, quat, q_start, lift, grip)
        if goal_filter is not None:
            n0 = len(sols)
            sols = [g for g in sols if goal_filter(g)]
            st["rejected_by_filter"] = n0 - len(sols)
        if not sols:
            return PlanResult(False, reason=f"no collision-free IK solution {st}", plan_s=time.time() - t0, info=st)
        last = None
        for i, g in enumerate(sols[:k_goals]):
            r = self.plan(q_start, g, **kw)
            if r.ok:
                r.plan_s = time.time() - t0
                r.info.update(ik=st, goal_rank=i)
                return r
            last = r
        last.plan_s = time.time() - t0
        last.info.update(ik=st)
        return last

    def cartesian(self, q_start, tcp_goal, quat_goal=None, lift=None, step=0.005, max_jump=0.08):
        """Straight fingertip line from FK(q_start) to tcp_goal (orientation slerped to quat_goal), IK per point
        seeded with the previous solution, each point and the dense path checked. Optionally the lift moves
        linearly along (lift = end value)."""
        t0 = time.time()
        q_start = np.asarray(q_start, float)
        p0, qw0 = self.fk_tcp(q_start)
        quat_goal = qw0 if quat_goal is None else quat_goal
        lift0 = q_start[I_LIFT]
        lift1 = lift0 if lift is None else lift
        n = max(1, int(math.ceil(max(np.linalg.norm(np.asarray(tcp_goal) - p0), abs(lift1 - lift0)) / step)))
        from scipy.spatial.transform import Slerp
        sl = Slerp([0, 1], Rotation.concatenate([rot(qw0), rot(quat_goal)]))
        path = [q_start]
        for k in range(1, n + 1):
            a = k / n
            q0 = path[-1].copy()
            q0[I_LIFT] = lift0 + a * (lift1 - lift0)
            q, ep, er = self._ik1(q0, p0 + a * (np.asarray(tcp_goal) - p0), wxyz(sl(a)), seed=0, iters=100,
                                  max_samples=1)
            if ep > 0.002 or er > 0.02:
                return PlanResult(False, reason=f"IK lost the line at {a:.2f} (err {ep * 1e3:.1f}mm {er:.3f}rad)",
                                  plan_s=time.time() - t0)
            if np.abs(q - path[-1]).max() > max_jump:
                return PlanResult(False, reason=f"joint jump {np.abs(q - path[-1]).max():.2f} at {a:.2f}",
                                  plan_s=time.time() - t0)
            path.append(q)
        ok, k, why, cmin = self.path_valid(path)
        if not ok:
            return PlanResult(False, reason=f"collision on the line: {why}", plan_s=time.time() - t0)
        return PlanResult(True, densify(path, self.check_step), "cartesian", time.time() - t0, 0, min_clearance=cmin)

    def keepout_dist(self, q, lo, hi, links=GRIPPER_LINKS):
        """Distance [m] from the collision hulls (vertices + surface samples) of `links` at q to the axis-aligned box
        lo..hi (world frame; 0 inside). The point-sample distance is a slight overestimate of the hull distance."""
        self.set_q(q)
        gp = self.rs.get_geoms_pos().cpu().numpy().reshape(-1, 3)
        gq = self.rs.get_geoms_quat().cpu().numpy().reshape(-1, 4)
        gR = Rotation.from_quat(gq[:, [1, 2, 3, 0]]).as_matrix()
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        dmin = np.inf
        for ai in self.arm_geoms:
            if self.gnames[ai] in links:
                w = self.arm_pts[ai] @ gR[ai].T + gp[ai]
                d = np.maximum(np.maximum(lo - w, w - hi), 0.0)
                dmin = min(dmin, float(np.linalg.norm(d, axis=1).min()))
        return dmin

    def find_standby(self, lift, grips=(0.0, 0.035), q_hint=None, min_clear=0.02, n=4000, keepout=None):
        """Collision-free folded pose nearest (L1 over joint1-6) to q_hint (default: the PiPER zero pose, which lies
        inside the mast), valid for every gripper opening in `grips`, clearance >= min_clear. Gripper = grips[0].
        keepout=(lo, hi, dist): additionally the gripper links (link6/7/8) stay >= dist from the box lo..hi."""
        q_hint = np.zeros(12) if q_hint is None else np.asarray(q_hint, float)
        base = q_hint.copy()
        base[I_LIFT] = lift

        def good(q):
            for g in grips:
                qq = q.copy()
                qq[I_GRIP] = (g, -g)
                ok, _, c = self.check(qq)
                if not ok or c < min_clear:
                    return False
                if keepout is not None and self.keepout_dist(qq, keepout[0], keepout[1]) < keepout[2]:
                    return False
            return True

        lo, hi = self.lim_lo[I_ARM], self.lim_hi[I_ARM]
        best = (np.inf, base) if good(base) else None
        for k in range(0 if best else n):
            sc = min(1.0, 0.1 + k / (n / 2))
            q = base.copy()
            q[I_ARM] = np.clip(base[I_ARM] + sc * self.rng.normal(0, 0.6, 6) * np.array([1, 1, 1, .3, .3, .3]),
                               lo, hi)
            if best is not None and np.abs(q[I_ARM] - base[I_ARM]).sum() >= best[0]:
                continue
            if good(q):
                best = (np.abs(q[I_ARM] - base[I_ARM]).sum(), q)
        if best is None:
            return None
        q = best[1].copy()
        q[I_GRIP] = (grips[0], -grips[0])
        return q


# ---------------- basket -> washer pick-and-place (cloth_ipc_washer_test --planner, arm_planner_test) ----------------
KEEPOUT_XY, KEEPOUT_DIST = 0.02, 0.03   # towel drop envelope: basket inner box + XY; the waiting gripper keeps DIST


def drop_keepout(drop_h=0.12, towel=0.30):
    """(lo, hi, dist) for find_standby / TransferSeq(keepout=): the towel drop envelope (basket inner xy box +
    KEEPOUT_XY, from the basket floor up to the towel top BASKET_FLOOR + drop_h + towel + KEEPOUT_DIST) and the
    distance KEEPOUT_DIST the gripper links keep from it."""
    bx, by = PRM.BASKET_CENTER_XY
    ix, iy = PRM.BASKET_INNER[0] / 2 + KEEPOUT_XY, PRM.BASKET_INNER[1] / 2 + KEEPOUT_XY
    fl = PRM.BASE_Z1 + PRM.BASKET_WALL
    return (bx - ix, by - iy, fl), (bx + ix, by + iy, fl + drop_h + towel + KEEPOUT_DIST), KEEPOUT_DIST


FINGER_BOTTOM_TO_TCP = -0.014    # vertical gripper: the TCP (link6 z 0.15) lies 1.4 cm below the finger bottoms


def set_grip(q, g):
    q = np.asarray(q, float).copy()
    q[I_GRIP] = (g, -g)
    return q


def seg_duration(path, nominal, vmax=0.8):
    """Execution time [s] of a planned path: `nominal`, longer if the lift or an arm joint would have to move faster
    than vmax [m/s, rad/s] on average (long RRT detours)."""
    p = np.asarray(path, float)
    arc = float(np.abs(np.diff(p[:, 3:10], axis=0)).max(1).sum()) if len(p) > 1 else 0.0
    return max(nominal, arc / vmax)


class TransferSeq:
    """Plans the basket -> washer segments one call at a time; every call appends dict(name, res=PlanResult, dur)
    to self.segs and, on success, moves self.q to the end of the path. self.q is the last *commanded* configuration:
    plan from it, not from measured joint positions (the kp 300 arm lags the command by up to 0.16 rad while moving).

    Call order of cloth_ipc_washer_test.py --planner (arm_planner_test.py plans the same order on the CPU):
      start() -> grip("pre_open", pre_open) -> approach(...) [-> approach("hover (re-sensed)", ...)] -> descend()
      -> close() -> lift1() -> lift_to("lift2 (lift joint)", lift_carry) -> raise_() -> carry(rx, ry) -> release()
      -> to_standby().
      Empty pinch:  close() -> grip("miss_open", pre_open) -> up_from_grasp("miss_up") -> next approach.
      Failed lift:  lift_to(...) -> grip("retry_open", open_) -> lift_to("retry_lower", lift_grasp) -> grip("pre_open")
                    -> next approach.
    Gripper: `hold` is the opening assumed for collision checks while holding the cloth (the fingers rest on the
    cloth, the commanded closing target in the IPC run is CLOSED or beyond it)."""

    def __init__(self, planner, lift_grasp=0.25, lift_carry=0.5, hold=0.004, open_=0.035, hover_dz=0.15,
                 lift1_z=0.80, carry_z=1.20, n_wp=6, tilt_step=12.0, wait_lift=None, keepout=None):
        """wait_lift / keepout: start() waits at this lift (default lift_grasp) with the gripper outside the box
        keepout=(lo, hi, dist) (the towel drop envelope); the caller then plans lift_to(lift_grasp) before the first
        approach."""
        self.wait_lift, self.keepout = wait_lift, keepout
        self.P = planner
        self.lift_grasp, self.lift_carry, self.hold, self.open = lift_grasp, lift_carry, hold, open_
        self.hover_dz, self.lift1_z, self.carry_z = hover_dz, lift1_z, carry_z
        self.n_wp, self.tilt_step = n_wp, tilt_step
        self.segs, self.skipped, self.carry_wps = [], [], []
        self.q = self.standby = self.target = None
        self.standby_s = 0.0
        self.grasp_standby = None

    def add(self, name, r, dur):
        self.segs.append(dict(name=name, res=r, dur=dur))
        if r.ok:
            self.q = r.path[-1].copy()
        return r.ok

    def start(self, q_hint=None):
        """Collision-free standby pose at wait_lift (default lift_grasp; the PiPER zero pose lies inside the mast),
        outside `keepout` if given; sets self.q."""
        t0 = time.time()
        lift = self.lift_grasp if self.wait_lift is None else self.wait_lift
        self.standby = self.P.find_standby(lift, q_hint=q_hint, keepout=self.keepout)
        if self.standby is not None and abs(lift - self.lift_grasp) > 1e-6:
            # the waiting arm pose does not stay valid when the lift is lowered (link2 meets the mast): the fold at
            # lift_grasp nearest to it is the goal of lower_to_grasp()
            self.grasp_standby = self.P.find_standby(self.lift_grasp, q_hint=self.standby)
            if self.grasp_standby is None:
                self.standby = None
        self.standby_s = time.time() - t0
        if self.standby is not None:
            self.q = self.standby.copy()
        return self.standby

    def lower_to_grasp(self, name, dur):
        """Planned move from the waiting pose (start(): lift wait_lift, gripper outside the keep-out box) to the fold
        at lift_grasp, lift and arm free (RRT), gripper opening as it is. No-op if already at lift_grasp."""
        if abs(self.q[I_LIFT] - self.lift_grasp) < 1e-6:
            return True
        if self.grasp_standby is None:   # start() was skipped (reset to a given standby)
            self.grasp_standby = self.P.find_standby(self.lift_grasp, q_hint=self.standby)
            if self.grasp_standby is None:
                return self.add(name, PlanResult(False, reason="no collision-free fold at lift_grasp"), dur)
        q = self.grasp_standby.copy()
        q[I_GRIP] = self.q[I_GRIP]
        return self.add(name, self.P.plan(self.q, q, free_lift=True), dur)

    def move_dofs(self, name, q_goal, dur):
        """Gripper or lift move (arm fixed): direct interpolation, densely checked (RRT with the lift free as the
        fallback for a lift move)."""
        r = self.P.plan(self.q, q_goal, free_lift=abs(q_goal[I_LIFT] - self.q[I_LIFT]) > 1e-6)
        return self.add(name, r, dur)

    def grip(self, name, g, dur):
        return self.move_dofs(name, set_grip(self.q, g), dur)

    def lift_to(self, name, lift, dur):
        q = self.q.copy()
        q[I_LIFT] = lift
        return self.move_dofs(name, q, dur)

    def approach(self, name, grasp, fb, yaws, dur):
        """Hover `hover_dz` above the grasp point (on the tool axis) with the first gripper yaw in `yaws` [deg] that
        plans, IK branch restricted to those from which the straight descent of the finger bottoms to `fb` works.
        Lift and gripper opening stay as they are (pre-open before). Sets self.target for the following moves."""
        t0 = time.time()
        gx, gy, gz = (float(v) for v in grasp)
        r, n = None, 0
        for yaw in yaws:
            n += 1
            down = quat_down(yaw)
            tcp = (gx, gy, fb + FINGER_BOTTOM_TO_TCP)
            hov = hover_point((gx, gy, gz), down, gz + self.hover_dz)
            r = self.P.plan_to_pose(self.q, hov, down, self.q[I_LIFT], self.q[I_GRIP][0],
                                    goal_filter=lambda g, tcp=tcp, down=down: self.P.cartesian(g, tcp, down).ok)
            if not r.ok:
                r = self._approach_bottom_up(hov, tcp, down, r)
            if r.ok:
                self.target = dict(grasp=(gx, gy, gz), fb=float(fb), yaw=float(yaw), quat=down, tcp=tcp)
                r.info["yaw"] = float(yaw)
                break
        if r is None:
            r = PlanResult(False, reason="no yaw candidates")
        r.plan_s = time.time() - t0
        r.info["yaws_tried"] = n
        return self.add(name, r, dur)

    def _approach_bottom_up(self, hov, tcp, down, r0, k_goals=4):
        """Fallback when no hover IK branch allows the descent (IK at the hover point often converges to a single
        branch): IK at the descent end, straight line up to the hover point, keep the hover ends from which the
        descent works, plan() to the nearest that plans. Returns r0 (with the fallback reason) if none does."""
        bots, st = self.P.ik_solutions(tcp, down, self.q)
        goals = []
        for qb in bots:
            up = self.P.cartesian(qb, hov, down)
            if up.ok and self.P.cartesian(up.path[-1], tcp, down).ok:
                goals.append(up.path[-1])
        for i, g in enumerate(goals[:k_goals]):
            r = self.P.plan(self.q, g)
            if r.ok:
                r.info.update(bottom_up=True, goal_rank=i, ik_bottom=st)
                return r
        r0.reason += f"; bottom-up: {len(bots)} IK at the bottom, {len(goals)} hover goals, none planned"
        return r0

    def descend(self, dur, name="descend"):
        t = self.target
        return self.add(name, self.P.cartesian(self.q, t["tcp"], t["quat"]), dur)

    def close(self, dur):
        return self.grip("close", self.hold, dur)

    def up_from_grasp(self, name, dur):
        """Empty pinch: straight back up to the TCP hover_dz above the finger-bottom target (gripper open)."""
        t = self.target
        gx, gy, _ = t["grasp"]
        up = hover_point((gx, gy, t["fb"]), t["quat"], t["fb"] + self.hover_dz)
        return self.add(name, self.P.cartesian(self.q, up, t["quat"]), dur)

    def vertical_high(self, name, zs, dur):
        """Straight fingertip line above the grasp point to the highest z in zs that works (grasp orientation)."""
        gx, gy, _ = self.target["grasp"]
        last = None
        for z in zs:
            r = self.P.cartesian(self.q, (gx, gy, z), self.target["quat"])
            if r.ok:
                r.info["z"] = round(float(z), 3)
                return self.add(name, r, dur)
            last = r
        return self.add(name, last if last is not None else PlanResult(False, reason="no z candidates"), dur)

    def lift1(self, dur):
        """Lift with the arm (lift joint at the grasp height), TCP up to lift1_z."""
        return self.vertical_high("lift1", np.arange(self.lift1_z, self.target["tcp"][2] + 0.1, -0.05), dur)

    def raise_(self, dur):
        """Highest vertical TCP above the grasp point (lift at lift_carry); a no-op if nothing higher is reachable
        (recorded in self.skipped, returns True)."""
        p, _ = self.P.fk_tcp(self.q)
        zs = [z for z in np.arange(self.carry_z, 0.6, -0.05) if z > p[2] + 0.01]
        gx, gy, _ = self.target["grasp"]
        last = None
        for z in zs:
            r = self.P.cartesian(self.q, (gx, gy, z), self.target["quat"])
            if r.ok:
                r.info["z"] = round(float(z), 3)
                return self.add("raise", r, dur)
            last = r
        self.skipped.append(("raise", f"no higher vertical TCP than {p[2]:.3f} m"
                             + (f" ({last.reason})" if last is not None else "")))
        return True

    def carry(self, rx, ry, dur_each, washer_xy=None):
        """n_wp waypoints from the grasp point to the release point (rx, ry): planned tilt min(60, k * tilt_step)
        (next larger in 5 deg steps if needed), gripper yaw kept ('keep') or the tool leaning toward the target
        ('toward'), as high as reachable (>= washer top + 4 cm over the washer footprint, else >= 0.85 m). Same pose
        candidates as cloth_ipc_washer_test.carry_gentle, but every waypoint is a planned segment.
        Route: on the straight xy line (as carry_gentle); if a waypoint has no collision-free solution there, it and
        all later ones are placed on the 'polar' route instead (azimuth and radius about the joint1 axis interpolated:
        the straight line passes within ~7 cm of the joint1 axis for grasp points at the back of the basket, where
        there is no IK solution, and its IK branches can swing the upper arm into the open lid)."""
        gx, gy, _ = self.target["grasp"]
        yaw = self.target["yaw"]
        wx, wy = self.P.washer_xy if washer_xy is None else washer_xy   # opening centre (base frame)
        b0 = self.P.link1_pos(self.q)[:2]
        a0, a1 = math.atan2(gy - b0[1], gx - b0[0]), math.atan2(ry - b0[1], rx - b0[0])
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        r0, r1 = math.hypot(gx - b0[0], gy - b0[1]), math.hypot(rx - b0[0], ry - b0[1])
        ok, polar = True, False
        self.carry_wps = []
        for k in range(1, self.n_wp + 1):
            s = k / self.n_wp
            t0 = time.time()
            r, found, tried = None, None, 0
            for route in (("polar",) if polar else ("line", "polar")):
                if route == "line":
                    px, py = gx + s * (rx - gx), gy + s * (ry - gy)
                else:
                    a, rr = a0 + s * da, r0 + s * (r1 - r0)
                    px, py = b0[0] + rr * math.cos(a), b0[1] + rr * math.sin(a)
                r, found, n = self._carry_wp(k, px, py, yaw, wx, wy)
                tried += n
                if found:
                    polar = route == "polar"
                    break
            if r is None:
                r = PlanResult(False, reason="no pose candidates")
            r.plan_s = time.time() - t0
            r.info["pose_candidates_tried"] = tried
            if found:
                r.info["route"] = route
            self.carry_wps.append(found)
            ok = self.add(f"carry{k}" + (f" z{found[2]:.2f} tilt{found[3]:.0f} {found[4]}" + (" polar" if polar else "")
                                         if found else ""), r, dur_each)
            if not ok:
                break
        return ok

    def _carry_wp(self, k, px, py, yaw, wx, wy):
        """Pose candidates of carry waypoint k at (px, py) -> (last PlanResult, found tuple or None, n tried)."""
        lx, ly = PRM.base_to_washer(px, py, (wx, wy), self.P.washer_yaw)   # footprint test in the washer frame
        over = abs(lx - PRM.WASHER_BODY_CX) < PRM.WASHER_W / 2 + 0.05 and abs(ly) < PRM.WASHER_D / 2 + 0.05
        zs = np.arange(self.carry_z, (PRM.WASHER_H + 0.04) if over else 0.85, -0.05)
        plan = min(60.0, k * self.tilt_step)
        b = self.P.link1_pos(self.q)
        az = math.atan2(py - b[1], px - b[0])
        r, tried = None, 0
        for t in np.arange(plan, 61.0, 5.0):
            for z in zs:
                for kind, quat in (("keep", quat_tilt_keep_yaw(az, yaw, t)), ("toward", quat_toward(az, t))):
                    tried += 1
                    r = self.P.plan_to_pose(self.q, (px, py, z), quat, self.q[I_LIFT], self.q[I_GRIP][0])
                    if r.ok:
                        return r, (round(float(px), 3), round(float(py), 3), round(float(z), 3), float(t), kind), tried
        return r, None, tried

    def release(self, dur):
        return self.grip("release", self.open, dur)

    def to_standby(self, dur):
        """Back to the standby arm pose at the current lift and opening (clears the opening for the tub view; the
        old IPC retract target above the mast has no IK solution)."""
        q = self.standby.copy()
        q[I_LIFT] = self.q[I_LIFT]
        q[I_GRIP] = self.q[I_GRIP]
        return self.add("->standby", self.P.plan(self.q, q), dur)


# ---------------- planner in a CPU child process (cloth_ipc_washer_test --planner, arm_planner_test --proxy) -------
SEQ_CALLS = ("start", "lower_to_grasp", "move_dofs", "grip", "lift_to", "approach", "descend", "close",
             "up_from_grasp", "vertical_high", "lift1", "raise_", "carry", "release", "to_standby")
SEQ_STATE = ("q", "standby", "standby_s", "target", "carry_wps", "skipped", "hold", "grasp_standby")


def _seq_state(T):
    return {k: getattr(T, k) for k in SEQ_STATE}


def serve(fd, ap_kwargs, seq_kwargs):
    """Child side of PlannerClient: builds the planner on the CPU backend (gs.init here) and answers requests on the
    connection `fd` until it is closed. A request is (op, ...); every reply is ("ok", payload) or ("err", traceback)."""
    from multiprocessing.connection import Connection
    conn = Connection(int(fd))
    try:
        t0 = time.time()
        AP = ArmPlanner(init_genesis=True, backend="cpu", **ap_kwargs)
        T = TransferSeq(AP, **seq_kwargs)
        conn.send(("ok", dict(build_s=time.time() - t0, state=_seq_state(T))))
    except BaseException:
        conn.send(("err", traceback.format_exc()))
        return
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):   # the parent is gone
            return
        op = msg[0]
        if op == "close":
            conn.send(("ok", None))
            return
        try:
            if op == "call":
                _, name, args, kw = msg
                if name not in SEQ_CALLS:
                    raise ValueError(f"not a TransferSeq call: {name}")
                n0 = len(T.segs)
                ret = getattr(T, name)(*args, **kw)
                out = dict(ret=ret, segs=T.segs[n0:], state=_seq_state(T))
            elif op == "reset":   # fresh TransferSeq (the planner scene and its rng stay), optionally given a standby
                T = TransferSeq(AP, **seq_kwargs)
                if msg[1] is not None:
                    T.standby, T.q = np.asarray(msg[1], float).copy(), np.asarray(msg[1], float).copy()
                out = dict(state=_seq_state(T))
            elif op == "seed":
                AP.rng = np.random.default_rng(msg[1])
                out = None
            elif op == "fk_tcp":
                out = AP.fk_tcp(msg[1])
            elif op == "clearance":
                out = AP.clearance_stats_many(msg[1])
            else:
                raise ValueError(f"unknown op {op}")
            conn.send(("ok", out))
        except BaseException:
            conn.send(("err", traceback.format_exc()))


class _APView:
    """The ArmPlanner calls the IPC test makes, forwarded to the planner (fk_tcp, clearance_stats_many)."""

    def __init__(self, client):
        self._c = client

    def fk_tcp(self, q):
        return self._c._request("fk_tcp", np.asarray(q, float))

    def clearance_stats_many(self, qs):
        return self._c._request("clearance", [np.asarray(q, float) for q in qs])


class PlannerClient:
    """TransferSeq (+ AP.fk_tcp / AP.clearance_stats_many) in a CPU-backend child process. Same interface as the
    TransferSeq the caller would build itself: the public calls (SEQ_CALLS) are forwarded and the state they change
    is mirrored after each reply (segs gets the segments the call added, as plain PlanResult data; the caller may
    edit its local copy). The child starts at construction and is awaited at the first request. A crashed child or
    a call longer than `timeout` s raises RuntimeError; shutdown() (also atexit) ends the child.
    mode="inproc": ArmPlanner + TransferSeq in this process on its current Genesis backend (init_genesis=False)."""

    def __init__(self, ap_kwargs, seq_kwargs, mode="subprocess", timeout=600.0, startup_timeout=900.0):
        self.mode, self.timeout, self.startup_timeout = mode, timeout, startup_timeout
        self.segs = []
        self.q = self.standby = self.target = None
        self.standby_s = 0.0
        self.carry_wps, self.skipped = [], []
        self._hold = seq_kwargs.get("hold", 0.004)
        self._seq_kwargs = seq_kwargs
        self.build_s = None
        self.AP = _APView(self)
        self.proc = self.conn = None
        self._ready = False
        if mode == "inproc":
            t0 = time.time()
            self._ap = ArmPlanner(init_genesis=False, **ap_kwargs)
            self._seq = TransferSeq(self._ap, **seq_kwargs)
            self.AP = self._ap
            self.build_s, self._ready = time.time() - t0, True
            return
        from multiprocessing.connection import Connection
        a, b = socket.socketpair()
        fd = b.fileno()
        code = ("import sys; sys.path.insert(0, %r); import pickle, arm_planner; "
                "arm_planner.serve(%d, *pickle.loads(bytes.fromhex(%r)))" % (
                    str(Path(__file__).resolve().parent), fd, pickle.dumps((ap_kwargs, seq_kwargs)).hex()))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")   # the planner never touches the GPU
        self.proc = subprocess.Popen([sys.executable, "-c", code], pass_fds=[fd], env=env)
        b.close()
        self.conn = Connection(a.detach())
        atexit.register(self.shutdown)

    @property
    def hold(self):
        return self._hold

    def _set_state(self, st):
        for k, v in st.items():
            setattr(self, "_hold" if k == "hold" else k, v)

    def _recv(self, timeout):
        t0 = time.time()
        while not self.conn.poll(1.0):
            if self.proc.poll() is not None and not self.conn.poll(0.5):
                raise RuntimeError(f"planner process died (exit code {self.proc.returncode})")
            if time.time() - t0 > timeout:
                self.shutdown()
                raise RuntimeError(f"planner process: no reply within {timeout:.0f} s")
        try:
            kind, payload = self.conn.recv()
        except (EOFError, OSError) as e:
            raise RuntimeError(f"planner process died (exit code {self.proc.poll()}): {e!r}") from e
        if kind == "err":
            raise RuntimeError("planner process:\n" + payload)
        return payload

    def _wait_ready(self):
        if not self._ready:
            info = self._recv(self.startup_timeout)
            self.build_s, self._ready = info["build_s"], True
            self._set_state(info["state"])

    def _request(self, *msg):
        if self.mode == "inproc":
            return {"fk_tcp": lambda q: self._ap.fk_tcp(q), "clearance": self._ap.clearance_stats_many}[msg[0]](msg[1])
        self._wait_ready()
        try:
            self.conn.send(msg)
        except (OSError, ValueError) as e:   # BrokenPipeError: the child is gone
            raise RuntimeError(f"planner process died (exit code {self.proc.poll()}): {e!r}") from e
        return self._recv(self.timeout)

    def wait_ready(self):
        if self.mode != "inproc":
            self._wait_ready()
        return self.build_s

    def reset(self, standby=None):
        """Fresh TransferSeq state (segs, skipped, ...), optionally at a given standby pose (no search)."""
        self.segs = []
        if self.mode == "inproc":
            self._seq = TransferSeq(self._ap, **self._seq_kwargs)
            if standby is not None:
                self._seq.standby, self._seq.q = np.array(standby, float), np.array(standby, float)
            self._set_state(_seq_state(self._seq))
            return
        self._set_state(self._request("reset", standby)["state"])

    def seed(self, seed):
        """Reseed the planner's rng."""
        if self.mode == "inproc":
            self._ap.rng = np.random.default_rng(seed)
        else:
            self._request("seed", seed)

    def _call(self, name, args, kw):
        if self.mode == "inproc":
            n0 = len(self._seq.segs)
            ret = getattr(self._seq, name)(*args, **kw)
            self.segs += self._seq.segs[n0:]
            self._set_state(_seq_state(self._seq))
            return ret
        out = self._request("call", name, args, kw)
        self.segs += out["segs"]
        self._set_state(out["state"])
        return out["ret"]

    def __getattr__(self, name):
        if name in SEQ_CALLS:
            return lambda *a, **kw: self._call(name, a, kw)
        raise AttributeError(name)

    def shutdown(self):
        if self.proc is None or self.conn is None:
            return
        proc, conn, self.conn = self.proc, self.conn, None
        try:
            conn.send(("close",))
            if conn.poll(5.0):
                conn.recv()
        except (EOFError, OSError, BrokenPipeError):
            pass
        conn.close()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
