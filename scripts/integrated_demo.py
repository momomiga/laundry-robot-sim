"""Integrated take (one continuous 1x video): the IPC towel transfer inside the house, with the VLM choosing the grasp.

Phases (captions are burnt into the frames):
  drop      the towel is dropped into the basket in the main room (robot at START, standby arm pose)
  settle    the towel slumps until still
  drive     the robot drives START -> corridor -> washer (base x / y translation only, 0.30 m/s, smooth start / stop)
  settle    the towel re-settles after the drive (the base then stands exactly at the parked origin = planner frame)
  vlm       the basket camera image goes to the VLM (qwen3-vl via ollama), the ranked pinch points are validated against
            the depth / cloth; a card (annotated image) is held for the inference time: the simulation is paused
  grasp     friction pinch (IPC only, no weld), lift, carry, release = the unchanged scripts/cloth_ipc_washer_test.py
            sequence (--planner required); then the tub camera; "in tub: NN %"

Frames: the parked-base frame of the IPC test (robot base at the washer = origin); the house is visual only
(scripts/house_frame.py). The robot is built at START (base joints 0 = the build pose, so IPC never sees a 5 m jump)
and Test._dq_off() maps the commanded world base position to the joint value. Cameras are given in the room frame
(`room_to_park`): a chase camera behind the robot that stays inside the house (below the 2.4 m ceiling, in front of the
walls), an entrance-side camera (room (1.0, 8.25, 1.5)) for the transfer, the tub camera of the test for the end.

  .venv-ipc/bin/python scripts/integrated_demo.py --planner --no-scoop --carry-mode gentle ... (see the GPU commands in
  the task notes / README of the run)
  .venv/bin/python scripts/integrated_demo.py --selftest              # CPU unit checks (no GPU)
  .venv/bin/python scripts/integrated_demo.py --render-check DIR      # CPU: still of each camera (house + robot)
"""
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))
import params as P  # noqa: E402
import cloth_ipc_washer_test as cwt  # noqa: E402
import house_frame  # noqa: E402
import robot_sensors  # noqa: E402
import vlm_grasp  # noqa: E402
from house_frame import park_to_room, room_to_park  # noqa: E402

SPEED = 0.30            # drive speed [m/s] (as scripts/demo_video.py)
RAMP_S = 0.6            # smoothstep ramp of the speed at the start and the end of the drive
START_ROOM = (1.35, 2.0)   # --start-room overrides it (set_start_room)
TRACE_DT = 0.25         # drive_trace sampling period [s of sim time]
TRACE_COLS = ["sim_t", "phase", "base_x", "base_y", "towel_cx", "towel_cy", "p90_z", "stretch", "acc_max"]
PARK_ROOM = tuple(P.PARK_POSE[:2])
CX = PARK_ROOM[0]       # corridor line (as demo_video.py)
WAYPOINTS_ROOM = [START_ROOM, (CX, 3.2), (CX, 4.6), (CX, 5.6), (CX, 6.6), PARK_ROOM]
ENTRANCE_CAM_ROOM = (1.0, 8.27, 1.85)
ENTRANCE_LOOK_ROOM = (0.66, 7.1, 0.55)
ENTRANCE_FOV = 62
CHASE_BACK, CHASE_H, CHASE_LOOK_AHEAD, CHASE_LOOK_Z = 1.7, 1.75, 0.20, 0.45
MOCK_ANSWER = '{"points": [{"x": 500, "y": 500}, {"x": 430, "y": 520}, {"x": 570, "y": 480}]}'
FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]


# ------------------------------------------------------------------ drive
def drive_waypoints_park():
    """START ... PARK in the parked-base frame (xy); the last point is exactly the origin (the planner frame)."""
    w = room_to_park(np.asarray(WAYPOINTS_ROOM, float))
    w[-1] = (0.0, 0.0)
    return w


def set_start_room(xy):
    """First waypoint (room frame); also the build pose of the robot (IntegratedTest.base_xy0, read at build time)."""
    global START_ROOM
    START_ROOM = (float(xy[0]), float(xy[1]))
    WAYPOINTS_ROOM[0] = START_ROOM
    IntegratedTest.base_xy0 = tuple(float(v) for v in drive_waypoints_park()[0])


def cmd_accel(path, p0, dt=cwt.DT):
    """|a| [m/s2] of the commanded xy along the path (second difference; the step before path[0] is at rest at p0)."""
    P3 = np.vstack([p0, p0, np.asarray(path, float)])
    return np.linalg.norm(np.diff(P3, 2, axis=0), axis=1) / dt ** 2


def drive_profile(length, v=SPEED, ramp=RAMP_S, dt=0.01):
    """Arc length s_k at t_k = k dt, k = 1..n: speed v with a smoothstep ramp up / down of `ramp` s (the distance of a
    ramp is v ramp / 2), so the duration is length / v + ramp; the last value is exactly `length`."""
    T = length / v + ramp
    n = int(math.ceil(T / dt - 1e-9))
    t = np.arange(1, n + 1) * dt

    def ramp_s(x):   # integral of the smoothstep 3x^2 - 2x^3 over x in [0, 1] (times v ramp)
        return v * ramp * (x ** 3 - x ** 4 / 2)

    s = np.where(t < ramp, ramp_s(np.clip(t / ramp, 0, 1)),
                 np.where(t < T - ramp, v * ramp / 2 + v * (t - ramp),
                          length - ramp_s(np.clip((T - t) / ramp, 0, 1))))
    s[-1] = length
    return np.minimum(s, length), T


def polyline_at(pts, s):
    pts = np.asarray(pts, float)
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], 1)


# ------------------------------------------------------------------ geometry / cameras
def room_boxes(min_top=0.12):
    """Axis-aligned boxes (cx, cy, cz, sx, sy, sz) of assets/room/room.urdf (collision + visual, room frame) that
    rise above min_top (floor slabs excluded)."""
    out = []
    root = ET.parse(ROOT / "assets/room/room.urdf").getroot()
    for node in list(root.iter("visual")) + list(root.iter("collision")):
        b = node.find("geometry/box")
        if b is None:
            continue
        o = [float(v) for v in node.find("origin").get("xyz").split()]
        sz = [float(v) for v in b.get("size").split()]
        if o[2] + sz[2] / 2 > min_top:
            out.append((*o, *sz))
    return out


def inside_box(p, box, margin=0.0):
    c, s = np.asarray(box[:3]), np.asarray(box[3:])
    return bool(np.all(np.abs(np.asarray(p, float) - c) < s / 2 + margin))


def chase_pose_room(base_room_xy):
    """(camera pos, lookat) in the room frame for a robot at base_room_xy: behind it (room -y, the drive direction is
    +y), 1.6 m high, kept inside the main room / corridor."""
    rx, ry = base_room_xy
    pos = np.array([min(max(rx, 0.3), 2.4), max(ry - CHASE_BACK, 0.25), CHASE_H])
    return pos, np.array([rx, ry + CHASE_LOOK_AHEAD, CHASE_LOOK_Z])


def chase_path_room(n_samples=400):
    """Chase camera positions along the drive (before the smoothing)."""
    pts = np.asarray(WAYPOINTS_ROOM, float)
    L = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    return [chase_pose_room(p)[0] for p in polyline_at(pts, np.linspace(0, L, n_samples))]


def segment_blocked(a, b, boxes, margin=-0.01, n=60, skip_near_end=0.25):
    """True if the open segment a -> b passes through one of the boxes (shrunk by -margin), ignoring the last
    `skip_near_end` metres at b (the robot / what is looked at)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.linalg.norm(b - a)
    for f in np.linspace(0, 1, n):
        if (1 - f) * d < skip_near_end:
            break
        p = a + f * (b - a)
        if any(inside_box(p, bx, margin) for bx in boxes):
            return True
    return False


# ------------------------------------------------------------------ image helpers
def _font(px):
    from PIL import ImageFont
    for f in FONT_PATHS:
        if os.path.exists(f):
            return ImageFont.truetype(f, px)
    return ImageFont.load_default(size=px)


def overlay(img, top=None, bottom=None):
    """Burn caption bands into an RGB uint8 frame (top-left label, bottom label), return a new array."""
    from PIL import Image, ImageDraw
    h, w = img.shape[:2]
    im = Image.fromarray(np.asarray(img, np.uint8)).convert("RGBA")
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    px = max(14, h // 26)
    f = _font(px)
    for text, anchor in ((top, "top"), (bottom, "bottom")):
        if not text:
            continue
        bb = d.textbbox((0, 0), text, font=f)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        pad = px // 2
        x0 = pad if anchor == "top" else (w - tw) // 2
        y0 = pad if anchor == "top" else h - th - 3 * pad
        d.rectangle([x0 - pad, y0 - pad // 2, x0 + tw + pad, y0 + th + pad + pad // 2], fill=(0, 0, 0, 150))
        d.text((x0, y0 - bb[1]), text, font=f, fill=(255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(im, lay).convert("RGB"))


def make_card(annotated, size, title, sub, note=None):
    """Card frame (W, H) for the paused simulation: the annotated basket image centred on a dark canvas."""
    from PIL import Image
    w, h = size
    canvas = np.full((h, w, 3), 22, np.uint8)
    im = Image.fromarray(np.asarray(annotated, np.uint8)).convert("RGB")
    k = 0.72 * h / im.height
    im = im.resize((int(im.width * k), int(im.height * k)))
    y0 = int(0.14 * h)
    canvas[y0:y0 + im.height, (w - im.width) // 2:(w - im.width) // 2 + im.width] = np.asarray(im)
    out = overlay(canvas, top=title, bottom=sub)
    if note:
        from PIL import ImageDraw
        pil = Image.fromarray(out)
        d = ImageDraw.Draw(pil)
        d.text(((w - im.width) // 2, y0 + im.height + 6), note, font=_font(max(12, h // 34)), fill=(210, 210, 210))
        out = np.asarray(pil)
    return out


def mark_point(rgb, uv, label, color=(255, 220, 0)):
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.asarray(rgb, np.uint8)).convert("RGB")
    d = ImageDraw.Draw(im)
    u, v = uv
    d.line([u - 12, v, u + 12, v], fill=color, width=3)
    d.line([u, v - 12, u, v + 12], fill=color, width=3)
    d.text((u + 14, v - 18), label, fill=color)
    return np.asarray(im)


# ------------------------------------------------------------------ the test
def room_cam_to_park(pos, look):
    return tuple(float(v) for v in room_to_park(np.asarray(pos, float))), \
        tuple(float(v) for v in room_to_park(np.asarray(look, float)))


def phase_durations(phases, key, t_end):
    """Total time per phase name from a phase log.

    phases: list of dicts with 'phase' and `key` (e.g. 'wall_t' or 'sim_t'), in order.
    Each phase lasts until the next entry's value; the last one ends at t_end.
    Repeated phases (e.g. 'retry') are summed.
    """
    out = {}
    for i, ph in enumerate(phases):
        end = phases[i + 1][key] if i + 1 < len(phases) else t_end
        out[ph["phase"]] = round(out.get(ph["phase"], 0.0) + (end - ph[key]), 1)
    return out


class IntegratedTest(cwt.Test):
    base_xy0 = tuple(float(v) for v in drive_waypoints_park()[0])

    def __init__(self, args):
        self._wall0 = time.perf_counter()   # origin for per-phase wall-clock times (set before gs.init)
        super().__init__(args)

    def _wall_now(self):
        return round(time.perf_counter() - self._wall0, 1)

    def scene_options(self):
        vis = house_frame.house_vis_options("park")
        lights = list(vis.lights) + [dict(type="directional", dir=(-0.4, 0.3, -1.0), color=(1.0, 1.0, 1.0),
                                          intensity=2.5)]
        return dict(vis_options=house_frame.house_vis_options("park", lights=lights))

    def plane_options(self):
        return dict(visualization=False)   # the house floor is the visible floor (the plane stays in IPC)

    def add_extra_entities(self, scene):
        house_frame.add_house(scene, frame="park", with_washer=False)

    def make_cameras(self, scene):
        super().make_cameras(scene)   # self.cam (reused as the chase camera), self.tub_cam, self.obs_cam if --save-obs
        if self.obs_cam is None:
            self.obs_cam = robot_sensors.add_basket_camera(scene, self.robot)
        pos, look = room_cam_to_park(ENTRANCE_CAM_ROOM, ENTRANCE_LOOK_ROOM)
        self.ent_cam = scene.add_camera(res=(cwt.W, cwt.H), pos=pos, lookat=look, fov=ENTRANCE_FOV, GUI=False)
        self.chase_state = None
        self.parked = False
        self.caption = "towel dropped into the basket"
        self.phase = "drop"
        self.phases = [dict(phase="drop", sim_t=0.0, video_t=0.0, wall_t=self._wall_now())]
        self.vlm_cands = None
        self.vlm = None
        self.vlm_attempts = []
        self.drive_info = None
        self.drive_trace = []     # rows of TRACE_COLS every TRACE_DT of sim time during drive + resettle
        self.trace_on = False
        self._cmd_hist = []       # last commanded base xy (3 steps) for the commanded acceleration
        self._acc_since = 0.0     # max commanded |a| since the last trace row
        self._tub_cache = (-1, 0.0)

    # ---------------- phases / frames ----------------
    def set_phase(self, name, caption):
        if name != self.phase:
            self.phases.append(dict(phase=name, sim_t=round(self.nstep * cwt.DT, 2),
                                    video_t=round(self.nframes / cwt.FPS, 2), wall_t=self._wall_now()))
        self.phase, self.caption = name, caption

    def follow(self, seg, row, grip_to=None):
        n = row["name"]
        if n.startswith(("miss", "retry")):
            self.set_phase("retry", "pinch missed / lift failed: retry at another point")
        elif n.startswith(("pre_open", "standby->hover", "hover", "descend", "close")):
            self.set_phase("grasp", "friction grasp (IPC contact + friction only)")
        elif n.startswith(("lower",)):
            self.set_phase("lower", "arm lowered into the basket")
        elif n.startswith(("lift",)):
            self.set_phase("lift", "lift the towel")
        elif n.startswith(("raise", "carry")):
            self.set_phase("carry", "carry over the washer")
        elif n.startswith("release"):
            self.set_phase("release", "release")
        elif n.startswith("->standby"):
            self.set_phase("retract", "retract the arm")
        super().follow(seg, row, grip_to)

    def in_tub_pct(self):
        if self._tub_cache[0] != self.nstep:
            self._tub_cache = (self.nstep, 100.0 * float(self.in_tub(self.cloth_pos()).mean()))
        return self._tub_cache[1]

    def _chase_update(self):
        rb = park_to_room(np.array([self.q[0], self.q[1], 0.0]))
        pos, look = chase_pose_room(rb[:2])
        if self.chase_state is None:
            self.chase_state = [pos, look]
        else:
            self.chase_state[0] = self.chase_state[0] + 0.1 * (pos - self.chase_state[0])
            self.chase_state[1] = self.chase_state[1] + 0.1 * (look - self.chase_state[1])
        p, lk = room_cam_to_park(*self.chase_state)
        self.cam.set_pose(pos=p, lookat=lk, up=(0.0, 0.0, 1.0))

    def frame(self):
        if self.view is self.tub_cam:
            cam = self.tub_cam
            if self.phase != "in tub":
                self.set_phase("in tub", "in tub")
        elif self.parked:
            cam = self.ent_cam
        else:
            self._chase_update()
            cam = self.cam
            if self.phase == "drop" and self.nstep * cwt.DT > 1.0:
                self.set_phase("settle", "towel settles in the basket")
        img = np.asarray(cam.render()[0]).astype(np.uint8)
        cap = self.caption
        if self.phase == "in tub":
            cap = f"in tub: {self.in_tub_pct():.0f} %"
        img = overlay(img, top=f"1x real time   t = {self.nstep * cwt.DT:5.1f} s", bottom=cap)
        self.proc.stdin.write(img.tobytes())

    def card_frames(self, img, seconds):
        """Identical frames that do not advance the simulation clock (nframes is the sim-time tick counter)."""
        if self.proc:
            data = np.asarray(img, np.uint8).tobytes()
            for _ in range(int(round(seconds * cwt.FPS))):
                self.proc.stdin.write(data)
        self.video_extra_s = getattr(self, "video_extra_s", 0.0) + seconds

    # ---------------- drive ----------------
    def step(self):
        super().step()
        if not self.trace_on:
            return
        h = self._cmd_hist
        h.append(np.array(self.cmd[:2], float))
        del h[:-3]
        if len(h) == 3:
            self._acc_since = max(self._acc_since, float(np.linalg.norm(h[2] - 2 * h[1] + h[0])) / cwt.DT ** 2)
        if self.nstep % int(round(TRACE_DT / cwt.DT)) == 0:
            self.trace_row()

    def trace_row(self):
        """GPU path (cloth state); not exercised on CPU. Towel centroid relative to the basket centre, p90 of the vertex
        height above the basket floor, the current (not running max) edge stretch, max commanded |a| since the last row."""
        x = self.cloth_pos()
        ln = np.linalg.norm(x[self.edges[:, 0]] - x[self.edges[:, 1]], axis=1)
        c = x.mean(0)[:2] - (np.asarray(self.q[:2]) + np.asarray(P.BASKET_CENTER_XY))
        self.drive_trace.append([round(self.nstep * cwt.DT, 2), self.phase, round(float(self.q[0]), 3),
                                 round(float(self.q[1]), 3), round(float(c[0]), 3), round(float(c[1]), 3),
                                 round(float(np.percentile(x[:, 2], 90) - cwt.BASKET_FLOOR), 3),
                                 round(float((ln / self.rest_len).max()), 3), round(self._acc_since, 2)])
        self._acc_since = 0.0

    def trace_summary(self, label, i0):
        r = self.drive_trace[i0:]
        if not r:
            return
        a = np.array([[v for v in row[2:]] for row in r], float)
        print(f"[trace {label}] {len(r)} rows t={r[0][0]}..{r[-1][0]}s  towel c(base-basket) start "
              f"({a[0, 2]:+.3f},{a[0, 3]:+.3f}) end ({a[-1, 2]:+.3f},{a[-1, 3]:+.3f})  p90_z end {a[-1, 4]:.3f} "
              f"max {a[:, 4].max():.3f}  stretch end {a[-1, 5]:.3f} max {a[:, 5].max():.3f}  "
              f"max cmd acc {a[:, 6].max():.2f} m/s2", flush=True)

    def drive(self):
        pts = drive_waypoints_park()
        L = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        s, T = drive_profile(L, SPEED, RAMP_S, cwt.DT)
        path = polyline_at(pts, s)
        path[-1] = pts[-1]
        self.set_phase("drive", "driving through the corridor to the washer")
        t0 = self.nstep * cwt.DT
        self.trace_on, self._cmd_hist, self._acc_since = True, [np.array(self.cmd[:2], float)] * 2, 0.0
        i0 = len(self.drive_trace)
        for xy in path:
            self.cmd[:2] = xy
            self.step()
        self.cmd[:2] = pts[-1]
        self.trace_summary("drive", i0)
        self._trace_i0 = len(self.drive_trace)
        self.drive_info = dict(length_m=round(L, 3), duration_s=round(T, 2), steps=len(path), speed_m_s=SPEED,
                               ramp_s=RAMP_S, sim_t0=round(t0, 2), sim_t1=round(self.nstep * cwt.DT, 2),
                               start_park=np.round(pts[0], 3).tolist(), end_park=pts[-1].tolist(),
                               waypoints_room=[list(map(float, w)) for w in WAYPOINTS_ROOM],
                               final_base_xy_mm_err=None)
        return pts

    def resettle(self):
        a = self.args
        t_end = self.nstep * cwt.DT + a.resettle_max
        self.hold(1.0)
        xp = self.cloth_pos()
        drift = float("nan")
        while self.nstep * cwt.DT < t_end:
            self.hold(0.5)
            xn = self.cloth_pos()
            drift = float(np.linalg.norm(xn - xp, axis=1).max())
            xp = xn
            if drift < a.still_tol:
                break
        self.trace_on = False
        self.trace_summary("resettle", self._trace_i0)
        return drift

    def after_settle(self, res):
        a = self.args
        res["settled_first_t"] = round(self.nstep * cwt.DT, 2)
        self.drive()
        self.parked = True
        self.base_xy0 = (0.0, 0.0)
        self.set_phase("settle2", "towel settles after the drive")
        t0 = self.nstep * cwt.DT
        drift = self.resettle()
        err = np.linalg.norm(self.q[:2])
        res["drive"] = dict(self.drive_info, final_base_xy_mm_err=round(float(err) * 1e3, 2),
                            resettle_s=round(self.nstep * cwt.DT - t0, 2), resettle_last_drift_mm=round(drift * 1e3, 2))
        res["drive_trace"] = dict(cols=TRACE_COLS, dt_s=TRACE_DT, rows=self.drive_trace)
        x = self.cloth_pos()
        res["settled2_in_basket_frac"] = float(self.in_basket(x).mean())
        res["settled2_zmax_above_floor"] = float(x[:, 2].max() - cwt.BASKET_FLOOR)
        if res["settled2_in_basket_frac"] < 0.99:
            raise RuntimeError("towel left the basket during the drive")
        self.qlog = [] if self.T is not None else None   # the measured-q check starts at the parked pose
        if a.vlm != "off":
            self.vlm_step(res, x)
        res["integrated"] = dict(phases=self.phases)
        if a.stop_after == "vlm":
            return True
        self.set_phase("lower", "arm lowered into the basket")
        return False

    # ---------------- VLM ----------------
    def vlm_step(self, res, x):
        a = self.args
        self.set_phase("vlm", "VLM picks the grasp point")
        out = Path(a.out or "outputs/integrated_demo.mp4")
        stem = str(out.with_suffix(""))
        obs_path = stem + "_vlm_obs.npz"
        res["vlm_obs"] = self.save_obs(obs_path, x)
        obs = vlm_grasp.load_obs(obs_path)
        mock = a.vlm == "mock"
        r = vlm_grasp.choose_center(obs, vlm=(lambda im: a.mock_answer) if mock else None, model=a.vlm_model,
                                    timeout=a.vlm_timeout)
        cands = []   # all valid ranked points (the first one is the choice; later attempts use the next ones)
        for rk, (u, v) in enumerate(r["pixels"], 1):
            p = vlm_grasp.pixel_to_point(u, v, obs)
            ok, why = vlm_grasp.validate(p, obs["verts"], (u, v), obs["rgb"])
            if ok and all(np.linalg.norm(np.array(p[:2]) - np.array(c["xy"])) > a.exclude_r for c in cands):
                cands.append(dict(rank=rk, xy=(float(p[0]), float(p[1])), point=[float(v_) for v_ in p]))
        r["valid_candidates"] = cands
        self.vlm_cands = cands
        card_png = stem + "_vlm_card.png"
        vlm_grasp.draw(obs, r, card_png)
        from PIL import Image
        annotated = np.asarray(Image.open(card_png).convert("RGB"))
        heur = None
        try:
            gx, gy, gz, *_ = self.select_grasp(x)
            heur = [round(gx, 3), round(gy, 3), round(gz, 3)]
        except AssertionError:
            pass
        fb = r["fallback"] or not cands
        if fb and heur is not None:
            uv, _ = vlm_grasp.project(obs, np.array([heur], float))
            annotated = mark_point(annotated, uv[0], "heuristic")
        lat = float(r["latency"] or 0.0)
        title = "VLM picks the grasp point" + (" (mock answer)" if mock else f" ({a.vlm_model})")
        sub = f"simulation paused during VLM inference ({lat:.1f} s)"
        if fb:
            note = f"VLM gave no valid point ({r['error']}): fallback = heuristic centroid grasp"
        else:
            note = (f"chosen: candidate {cands[0]['rank']} of {len(r['pixels'])} (green); yellow = backup for a retry; "
                    "red = rejected")
        card = make_card(annotated, (cwt.W, cwt.H), title, sub, note)
        Image.fromarray(card).save(stem + "_vlm_card_frame.png")
        self.card_frames(card, max(lat, a.card_s))
        r["card_png"] = card_png
        r["heuristic_point"] = heur
        r["mode"] = a.vlm
        r["model"] = None if mock else a.vlm_model
        self.vlm = r
        res["vlm"] = json.loads(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))

    def grasp_center(self, x, k):
        if self.vlm_cands is None:
            return None
        c = self.vlm_cands[k] if k < len(self.vlm_cands) else None
        self.vlm_attempts.append(dict(k=k, source="vlm" if c else "heuristic", rank=c["rank"] if c else None,
                                      xy=c["xy"] if c else None))
        if c is None:
            return None
        return (c["xy"], self.args.vlm_radius) if self.args.vlm_radius else c["xy"]

    def run(self):
        try:
            return super().run()
        finally:
            self.res["vlm_attempts"] = self.vlm_attempts
            self.res["phases"] = self.phases
            t_end_wall = self._wall_now()
            t_end_sim = round(self.nstep * cwt.DT, 2)
            self.res["phase_wall_s"] = phase_durations(self.phases, "wall_t", t_end_wall)
            self.res["phase_sim_s"] = phase_durations(self.phases, "sim_t", t_end_sim)
            self.res["video_extra_card_s"] = round(getattr(self, "video_extra_s", 0.0), 2)
            self.res["cameras"] = camera_report()


def camera_report():
    cp, cl = room_cam_to_park(ENTRANCE_CAM_ROOM, ENTRANCE_LOOK_ROOM)
    return dict(entrance_room=ENTRANCE_CAM_ROOM, entrance_park=np.round(cp, 3).tolist(),
                chase="behind the robot (room -y) %.2f m, %.2f m high" % (CHASE_BACK, CHASE_H))


# ------------------------------------------------------------------ CLI
def make_parser():
    ap = cwt.make_parser()
    for act in ap._actions:
        if act.dest == "stop_after":
            act.choices = list(act.choices) + ["vlm"]
    ap.add_argument("--vlm", choices=["ollama", "mock", "off"], default="ollama",
                    help="grasp point from the VLM (ollama), a canned answer (mock) or the heuristic only (off)")
    ap.add_argument("--vlm-model", default="qwen3-vl:8b-instruct")
    ap.add_argument("--mock-answer", default=MOCK_ANSWER, help="--vlm mock: the VLM reply text")
    ap.add_argument("--vlm-timeout", type=float, default=60.0)
    ap.add_argument("--vlm-radius", type=float, default=None, help="search radius around the VLM point (default: "
                    "--grasp-r)")
    ap.add_argument("--card-s", type=float, default=3.0, help="min seconds of the VLM card in the video")
    ap.add_argument("--resettle-max", type=float, default=8.0, help="max re-settle time after the drive [s]")
    ap.add_argument("--start-room", default=None, metavar="X,Y", help="first drive waypoint / build pose, room frame "
                    "(default %.3f,%.3f; %.3f,2.0 = on the corridor line, no corner)" % (*START_ROOM, CX))
    ap.add_argument("--res", default="1280x720", help="video size WxH")
    ap.add_argument("--selftest", action="store_true", help="CPU unit checks, no simulation")
    ap.add_argument("--render-check", metavar="DIR", default=None, help="CPU: render each camera (house + robot, no "
                    "IPC) to DIR and exit")
    ap.set_defaults(out="outputs/integrated_demo.mp4", json="outputs/integrated_demo.json", planner=True,
                    planner_room=True, scoop=False, carry_mode="gentle")
    return ap


def main():
    ap = make_parser()
    args = ap.parse_args()
    cwt.W, cwt.H = (int(v) for v in args.res.lower().split("x"))
    if args.start_room:
        set_start_room([float(v) for v in args.start_room.split(",")])
    if args.selftest:
        return selftest()
    if args.render_check:
        return render_check(args.render_check)
    if not args.planner:
        ap.error("the integrated demo needs --planner (the base is driven in world coordinates)")
    cwt.main(IntegratedTest, ap)


# ------------------------------------------------------------------ CPU checks
def selftest():
    ok = True

    def chk(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name, info)

    # waypoints
    w = drive_waypoints_park()
    chk("start in the parked frame", np.allclose(w[0], room_to_park(START_ROOM)[:2]), np.round(w[0], 3))
    chk("last waypoint exactly (0, 0)", w[-1][0] == 0.0 and w[-1][1] == 0.0)
    chk("PARK maps to ~0", np.abs(room_to_park(PARK_ROOM)[:2]).max() < 1e-9)
    chk("round trip", np.allclose(park_to_room(room_to_park(np.asarray(WAYPOINTS_ROOM, float))), WAYPOINTS_ROOM))
    chk("class base_xy0 == start", np.allclose(IntegratedTest.base_xy0, w[0]))
    # profile
    L = float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum())
    s, T = drive_profile(L)
    v = np.diff(np.r_[0.0, s]) / cwt.DT
    path = polyline_at(w, s)
    path[-1] = w[-1]
    vp = np.linalg.norm(np.diff(np.vstack([w[0], path]), axis=0), axis=1) / cwt.DT
    acc = np.diff(vp) / cwt.DT
    chk("length / duration", True, f"L={L:.3f} m  T={T:.2f} s ({len(s)} steps; {L / SPEED:.2f} s at constant speed)")
    chk("max speed <= 0.30", vp.max() <= SPEED + 1e-6, f"{vp.max():.4f}")
    chk("reaches the cruise speed", vp.max() > SPEED - 1e-3)
    chk("ends at rest", vp[-1] < 0.01 and vp[0] < 0.01, f"first {vp[0]:.4f} last {vp[-1]:.4f}")
    chk("final position exact", np.array_equal(path[-1], w[-1]))
    chk("monotone", bool((np.diff(s) >= -1e-12).all()))
    chk("max |accel| <= 1.6 v / ramp", np.abs(acc[1:-1]).max() <= 1.6 * SPEED / RAMP_S,
        f"{np.abs(acc[1:-1]).max():.3f} m/s2")
    del v
    # commanded 2D acceleration (second difference of the commanded xy): the corner of the default start vs the
    # corridor-line start
    amax = {}
    for tag, st in (("default start", (1.35, 2.0)), ("corridor start", (CX, 2.0))):
        wr = np.asarray([st] + list(WAYPOINTS_ROOM[1:]), float)
        wp_ = room_to_park(wr)[:, :2]
        wp_[-1] = (0.0, 0.0)
        Lw = float(np.linalg.norm(np.diff(wp_, axis=0), axis=1).sum())
        sw, _ = drive_profile(Lw)
        pw = polyline_at(wp_, sw)
        pw[-1] = wp_[-1]
        amax[tag] = float(cmd_accel(pw, wp_[0]).max())
        print(f"     max commanded |a| ({tag} {st}): {amax[tag]:.2f} m/s2")
    chk("corridor-line start: max commanded |a| <= 1.0 m/s2", amax["corridor start"] <= 1.0,
        f"{amax['corridor start']:.2f} (default start {amax['default start']:.2f})")
    # cameras vs the room boxes
    boxes = room_boxes()
    chk("room boxes parsed", len(boxes) > 20, f"{len(boxes)}")
    ex_lo, ex_hi = np.array([0.0, 0.0, 0.0]), np.array([P.ROOM_W, P.UPPER_Y[1], P.WALL_H])
    margin = 0.12

    def clear(p):
        p = np.asarray(p, float)
        return bool((p > ex_lo + margin).all() and (p < ex_hi - margin).all() and
                    not any(inside_box(p, b, margin) for b in boxes))

    ent = np.array(ENTRANCE_CAM_ROOM)
    chk("entrance camera inside the house, clear of all boxes (0.12 m)", clear(ent), ENTRANCE_CAM_ROOM)
    chk("entrance camera has a free line to the look point", not segment_blocked(ent, ENTRANCE_LOOK_ROOM, boxes))
    chk("entrance camera sees the washer opening",
        not segment_blocked(ent, (*P.WASHER_POS[:2], 0.9), boxes))
    cp = room_cam_to_park(ENTRANCE_CAM_ROOM, ENTRANCE_LOOK_ROOM)
    chk("entrance camera round trip", np.allclose(park_to_room(np.array(cp[0])), ENTRANCE_CAM_ROOM))
    bad = [p for p in chase_path_room() if not clear(p)]
    chk("chase camera clear along the drive", not bad, f"{len(bad)} bad of 400, first {bad[:1]}")
    blocked = 0
    for p in polyline_at(np.asarray(WAYPOINTS_ROOM), np.linspace(0, L, 120)):
        cpos, clook = chase_pose_room(p)
        blocked += segment_blocked(cpos, clook, boxes)
    chk("chase camera line of sight free", blocked == 0, f"{blocked} blocked of 120")
    # the smoothed camera (exp. filter 0.1 per frame) over the whole drive
    state, bad_s = None, 0
    for xy in path[::3]:
        pos, look = chase_pose_room(park_to_room(np.array([xy[0], xy[1], 0.0]))[:2])
        state = [pos, look] if state is None else [state[0] + 0.1 * (pos - state[0]),
                                                   state[1] + 0.1 * (look - state[1])]
        bad_s += not clear(state[0])
    chk("smoothed chase camera clear", bad_s == 0, f"{bad_s} bad frames")
    # tub camera: base frame, inside the house? (park frame -> room)
    wx, wy = cwt.WASHER_XY
    tub = park_to_room(np.array([wx + 0.20, wy - 0.12, 1.85]))
    chk("tub camera clear of the room boxes", clear(tub), np.round(tub, 3))
    # captions / cards
    img = np.zeros((cwt.H, cwt.W, 3), np.uint8) + 90
    o = overlay(img, top="1x real time   t =  12.3 s", bottom="friction grasp (IPC)")
    chk("caption changes pixels, same shape", o.shape == img.shape and (o != img).any())
    # VLM step on a real observation (mock answer)
    obs_p = ROOT / "outputs/remote/cloth_ipc_layout2/obs_s10.npz"
    if obs_p.exists():
        obs = vlm_grasp.load_obs(obs_p)
        r = vlm_grasp.choose_center(obs, vlm=lambda im: MOCK_ANSWER)
        chk("mock VLM -> valid centre", not r["fallback"] and r["center_xy"] is not None,
            f"rank {r['rank']} centre {np.round(r['center'], 3) if r['center'] else None} rejections {r['rejections']}")
        out = Path(os.environ.get("INTEGRATED_SELFTEST_DIR", "/tmp")) / "vlm_card_selftest.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        vlm_grasp.draw(obs, r, str(out))
        from PIL import Image
        card = make_card(np.asarray(Image.open(out).convert("RGB")), (cwt.W, cwt.H), "VLM picks the grasp point (mock)",
                         "simulation paused during VLM inference (0.0 s)", "chosen: candidate 1")
        Image.fromarray(card).save(out.with_name("vlm_card_frame_selftest.png"))
        chk("card frame", card.shape == (cwt.H, cwt.W, 3), str(out.with_name("vlm_card_frame_selftest.png")))
        fb = vlm_grasp.choose_center(obs, vlm=lambda im: "no idea")
        chk("fallback -> centre None", fb["fallback"] and fb["center_xy"] is None, fb["error"])
    else:
        chk("obs_s10.npz present", False, str(obs_p))
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


def render_check(outdir):
    """Rigid-only CPU scene (house, robot at a few poses, washer) -> a still of each camera."""
    import genesis as gs
    from PIL import Image
    Path(outdir).mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.cpu, logging_level="warning")
    kw = IntegratedTest.scene_options(IntegratedTest)
    sc = gs.Scene(show_viewer=False, **kw)
    sc.add_entity(gs.morphs.Plane(**IntegratedTest.plane_options(IntegratedTest)))
    robot = sc.add_entity(gs.morphs.URDF(file=str(ROOT / "assets/robot/house_robot.urdf"), fixed=True))
    wxy, wyaw = cwt.WASHER_XY, cwt.WASHER_YAW_DEG
    sc.add_entity(gs.morphs.URDF(file=str(ROOT / "assets/washer/washer.urdf"), pos=(*wxy, cwt.WASHER_Z),
                                 euler=(0, 0, wyaw), fixed=True))
    house_frame.add_house(sc, frame="park", with_washer=False)
    t = IntegratedTest.__new__(IntegratedTest)
    cams = {}
    cp, cl = room_cam_to_park(ENTRANCE_CAM_ROOM, ENTRANCE_LOOK_ROOM)
    cams["entrance"] = sc.add_camera(res=(cwt.W, cwt.H), pos=cp, lookat=cl, fov=ENTRANCE_FOV, GUI=False)
    cams["chase"] = sc.add_camera(res=(cwt.W, cwt.H), pos=(-1.7, 0, 1.75), lookat=(0, 0, 0.5), fov=40, GUI=False)
    wx, wy = wxy
    cams["tub"] = sc.add_camera(res=(cwt.W, cwt.H), pos=(wx + 0.20, wy - 0.12, 1.85),
                                lookat=(wx - 0.02, wy + 0.01, 0.40), fov=40, GUI=False)
    sc.build()
    names = ["base_x_joint", "base_y_joint", "base_yaw_joint", "lift_joint"] + [f"joint{i}" for i in range(1, 9)]
    dofs = [robot.get_joint(n).dofs_idx_local[0] for n in names]
    kp = np.array([8000, 8000, 3000, 8000] + [300] * 6 + [2000] * 2, dtype=float)
    robot.set_dofs_kp(kp, dofs)
    robot.set_dofs_kv(2 * np.sqrt(kp), dofs)
    standby = np.array([0, 0, 0, 0.5, -0.0291, 0.2192, 0.0, -0.0269, -0.0734, -0.0048, 0.0, -0.0])
    del t
    report = []
    for tag, room_xy, cam in (("start", START_ROOM, "chase"), ("room_mid", (CX, 3.6), "chase"),
                              ("door", (CX, 5.0), "chase"), ("corridor", (CX, 6.2), "chase"),
                              ("park", PARK_ROOM, "chase"), ("park", PARK_ROOM, "entrance"),
                              ("park", PARK_ROOM, "tub")):
        pk = room_to_park(np.array([*room_xy, 0.0]))
        q = standby.copy()
        q[:2] = pk[:2]
        robot.set_dofs_position(q, dofs)
        robot.control_dofs_position(q, dofs)
        sc.step()   # a plain set_dofs_position is not picked up by the rasterizer after the first render
        if cam == "chase":
            pos, look = chase_pose_room(room_xy)
            p, lk = room_cam_to_park(pos, look)
            cams[cam].set_pose(pos=p, lookat=lk, up=(0.0, 0.0, 1.0))
        img = np.asarray(cams[cam].render()[0]).astype(np.uint8)
        img = overlay(img, top="1x real time   t = 0.0 s", bottom=f"{tag} / {cam}")
        f = Path(outdir) / f"{tag}_{cam}.png"
        Image.fromarray(img).save(f)
        report.append(str(f))
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
