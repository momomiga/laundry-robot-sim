"""Multi-garment integrated take: towel + T-shirt + sock in the robot's basket, picked one by one into the washer.

One cycle = observe the basket (basket camera, every cycle) -> VLM boxes + recheck (scripts/vlm_multi.py) -> pick the
highest confirmed garment that is not given up -> planned lowering -> friction pinch (retries at other points) -> lift ->
carry over the washer -> release (+ wrist wiggle if either finger still touches cloth) -> retract -> look at the tub 1 s
-> re-settle.  Cycles repeat until nothing is left, everything left is given up, --max-cycles or --sim-budget.
Frames, planner, IPC settings and the per-phase wall / sim times are those of scripts/integrated_demo.py (this class is
IntegratedTest + the multi-garment scene of scripts/multi_cloth_scene.py); the planned arm motions are
scripts/arm_planner.TransferSeq on the CPU planner child process, unchanged.

WHAT DECIDES (perception only; no simulator ground truth picks a target, a point or a yaw)
  * Which garment is grasped, in which order, and when to stop: the VLM boxes (+ recheck of unexplained cloth-coloured
    regions), the colour / point-cloud segmentation of vlm_multi.py, the height ordering of the visible surface.  Only
    detections with status ok and source vlm / recheck are targets; unconfirmed leftovers are reported, not grasped.
  * The grasp point and the gripper yaw: the finite points of the target's mask (eroded --erode px) in the basket-camera
    point cloud are the "heap" handed to cloth_ipc_washer_test.select_grasp (highest point near the detection's 3-D
    centroid, 5 cm inside the basket walls, radius grows for --min-bite, always inside the mask points) and ridge_yaws
    (PCA of the visible surface around the point).  Retries exclude earlier points (--exclude-r) on the same observation.
  * Per attempt (the "simulated tactile / force sensing" of the gripper, aggregated over ALL garments, no object
    identity): the IPC squeeze / vertical coupling force and the finger gap (empty pinch, floor stall), the number of
    vertices touching each finger after the lift (lift_ok = not miss and both fingers touch and not weak_hold), the
    slide of the pinched vertices (--lift-slide-max, off by default), the vertices still touching either finger after
    the release (wrist wiggle, then open + wiggle again), plus the robot's own joint positions.
  * Attempt bookkeeping: failed attempts per object (association across cycles by VLM label + centroid < 6 cm) ->
    --attempts-per-object, then the object is given up.
SCORING ONLY (never read by a decision; written to the result JSON / used for captions and the final verdict)
  * Everything per object through obj_slices: which objects rose during the lift (double grasp), held until release,
    over the opening at release, in_tub / in_basket fractions, max edge stretch, penetration, final location, the
    cycle in which an object first counts as in the washer; the "in washer n/3 (sim)" caption; the seg_gt match of the
    target mask; the sim count `n_in_tub_after` of each cycle.

Result JSON additions (besides everything cloth_ipc_washer_test / integrated_demo write)
  cycles[]   per cycle: cycle, t0_sim, t1_sim, obs, vlm_json, n_detections, n_confirmed, order (labels), recheck,
             vlm_latency_s, target {label, label_raw, source, obj_id, centroid_3d, centroid_px, mask_px, n_points,
             seg_gt_index, seg_gt_name}, attempts[] (copy of the attempt numbers + lift_ok, lifted_objects, grasp_px),
             lift_ok, carried, release_check {...}, objects (per-object scoring), n_in_tub_after, end (loaded |
             lift_failed | given_up | planning_error | ...), card {...}
  cards[]    [{cycle, frame0, frame1 (mp4 frame indices, end exclusive), sim_t, obs, vlm_json, png}]
  objects_final[]  [{name, label_by_vlm, where (tub|basket|floor|robot), cycle_loaded}]
  also stop_reason, n_confirmed_first, objects_tracked (attempts per perceived object), phases carry `cycle`, `target`.

  flock /tmp/house_robot-gpu.lock .venv-ipc/bin/python scripts/multi_integrated_demo.py --seed 0 --no-drive --no-video \\
      --max-cycles 1 <production flags> --vlm ollama
  uv run --no-sync python scripts/multi_integrated_demo.py --selftest     # CPU, no Genesis scene
  uv run --no-sync python scripts/multi_integrated_demo.py --offline-check outputs/multi/obs_s0.npz \\
      --offline-vlm outputs/multi/vlm2_s0.json                             # CPU planner on a saved observation
"""
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))
import params as P  # noqa: E402
import cloth_ipc_washer_test as cwt  # noqa: E402
import garments as G  # noqa: E402
import integrated_demo as idm  # noqa: E402
import vlm_grasp  # noqa: E402
import vlm_multi  # noqa: E402
from arm_planner import NAMES as ARM_NAMES, drop_keepout  # noqa: E402
from integrated_demo import IntegratedTest, make_card, overlay  # noqa: E402
from multi_cloth_scene import MultiCloth  # noqa: E402

DROP_STACK_TOP = 0.65     # keep-out box top above the basket floor [m]: the 3 garments dropped stacked reach <= 0.627 m
#                           (200 seeds, --drop-h 0.12 --garment-gap 0.04); the single towel envelope is 0.42 m
ASSOC_R = 0.06            # same object across cycles: same VLM label and centroid within this [m]
TUB_FRAC = 0.90           # an object counts as in the washer when this fraction of its vertices is in the tub volume
BASKET_FRAC = 0.50        # ... as in the basket at this fraction
NEAR_TIP = 0.05           # not in tub / basket and a vertex within this of the fingertip centre: "robot"
LIFT_RISE = 0.03          # an object whose highest vertex rose by more than this during the lift counts as lifted [m]
HELD_DIST = 0.03          # held until release: the nearest vertex stayed within this of the fingertip centre [m]
WIGGLE_DEG = 30.0         # release check: wrist (joint6) +- this
J6 = ARM_NAMES.index("joint6")
CLOSE_AXIS_L6 = np.array([0.0, 1.0, 0.0])   # finger closing axis in the link6 frame (joint7 / joint8 axes, URDF)
CONFIRMED_SOURCES = ("vlm", "recheck")
ACTIONS = dict(vlm="VLM looks at the basket", lower="lower the arm", grasp="friction grasp", retry="missed, retry",
               lift="lift", carry="carry to the washer", release="release", retract="retract the arm",
               settle2="garments settle", **{"in tub": "check the washer"})
CYCLE_KEYS = ("cycle", "t0_sim", "t1_sim", "obs", "vlm_json", "n_detections", "n_confirmed", "order", "recheck",
              "vlm_latency_s", "target", "attempts", "lift_ok", "carried", "release_check", "objects",
              "n_in_tub_after", "end", "card")


# ------------------------------------------------------------------ pure helpers (selftest)
def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def clean(o):
    """Round-trip through JSON: numpy scalars / arrays become plain python values."""
    return json.loads(json.dumps(o, default=_jsonable))


def is_confirmed(d):
    """A grasp target: a detection the VLM confirmed (first answer or recheck) whose mask segmented (status ok)."""
    return d.get("status") == "ok" and d.get("source") in CONFIRMED_SOURCES


def centroid_xy(d):
    return (float(d["centroid_3d"][0]), float(d["centroid_3d"][1]))


def associate(objs, label, xy, radius=ASSOC_R):
    """The tracked object with the same VLM label whose last centroid is nearest and within `radius` (else None)."""
    best = None
    for o in objs:
        dist = math.hypot(o["xy"][0] - xy[0], o["xy"][1] - xy[1])
        if o["label"] == label and dist < radius and (best is None or dist < best[0]):
            best = (dist, o)
    return None if best is None else best[1]


def track(objs, label, xy, cycle):
    o = associate(objs, label, xy)
    if o is None:
        o = dict(id=len(objs) + 1, label=label, xy=xy, failed=0, given_up=False, reason=None, first_cycle=cycle,
                 cycles=[])
        objs.append(o)
    o["xy"] = xy
    o["cycles"].append(cycle)
    return o


def choose_target(dets, order, objs, per_obj_max, cycle):
    """Pick the first confirmed detection (pick order: highest first) whose tracked object is not given up.
    Returns (det index | None, object | None, no_target_reason | None).  Every confirmed detection is tracked."""
    conf = [i for i in order if is_confirmed(dets[i])]
    tracked = {i: track(objs, dets[i]["label"], centroid_xy(dets[i]), cycle) for i in conf}
    for i in conf:
        o = tracked[i]
        if o["failed"] >= per_obj_max:
            o["given_up"], o["reason"] = True, o["reason"] or "attempts"
        if not o["given_up"]:
            return i, o, None
    return None, None, ("all_given_up" if conf else "no_confirmed_detection")


def stop_reason(k, max_cycles, sim_t, budget):
    """Reason not to start cycle k (1-based), else None."""
    if k > max_cycles:
        return "max_cycles"
    if budget and sim_t >= budget:
        return "sim_budget"
    return None


def is_empty_reply(text):
    """True if the raw VLM reply is a valid empty answer: an empty JSON list (after <think> blocks and code fences), or
    a dict whose values are all empty lists ({"objects": []}).  Prose / broken JSON is not."""
    import re
    t = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S)
    t = re.sub(r"```[a-zA-Z]*", "", t).strip()
    try:
        obj = json.loads(t)
    except ValueError:
        return False
    if isinstance(obj, list):
        return len(obj) == 0
    return isinstance(obj, dict) and bool(obj) and all(isinstance(v, list) and not v for v in obj.values())


def empty_result(obs, raw, latency, recheck=None):
    """vlm_multi.run's result for a VLM that answered "no garment": zero detections, but the garment-pixel count and
    the recheck of unexplained cloth-coloured regions still run (vlm_multi.run itself reports it as parse_failure)."""
    res = dict(raw=raw, latency=latency, error=None, detections=[], order=[], evaluation=None, recheck=[],
               empty_reply=True)
    dets, info = vlm_multi.segment_detections(obs, [])
    garment = info.pop("_garment")
    res["segmentation"] = info
    if recheck is not None:
        res["recheck"] = vlm_multi.recheck_residuals(obs, dets, garment, recheck)
    for d in dets:
        if d["status"] == "ok":
            vlm_multi.describe_detection(obs, d, vlm_multi.get_points(obs))
    res["order"] = vlm_multi.pick_order(dets)
    for rank, i in enumerate(res["order"], 1):
        dets[i]["order"] = rank
    for d in dets:
        d["area_px"] = int(d["mask"].sum())
    res["detections"] = [{k: v for k, v in d.items() if k != "mask"} for d in dets]
    return res, dets


def no_target_stop(reason, garment_px, min_px=None):
    """Stop reason when a cycle has no target: nothing left (no confirmed detection and (almost) no garment pixels),
    or cloth-coloured pixels the VLM did not confirm remain (reported as missed, not grasped)."""
    min_px = vlm_multi.MIN_RESIDUAL_PX if min_px is None else min_px
    if reason == "all_given_up":
        return "all_given_up"
    if garment_px is None:
        return "vlm_error"
    return "basket_empty" if garment_px < min_px else "unconfirmed_garment_pixels_left"


def lift_ok(res):
    """Gripper-only verdict of one attempt (aggregated tactile signals): not an empty pinch, both fingers still touch
    cloth after the lift, no weak hold.  (The single-towel `lift_success` needs >= 50 % of ALL vertices above the rim.)"""
    if res.get("miss") or res.get("touch_lifted") is None:
        return False
    return bool(min(res["touch_lifted"]) > 0 and not res.get("weak_hold"))


def caption_fmt(k, n, label, action):
    return f"[{k}/{n}] {label}: {action}" if label else f"cycle {k}: {action}"


def classify_where(frac_tub, frac_basket, near_tip):
    if frac_tub >= TUB_FRAC:
        return "tub"
    if frac_basket >= BASKET_FRAC:
        return "basket"
    return "robot" if near_tip else "floor"


def lifted_flags(zmax_before, zmax_after, rise=LIFT_RISE):
    return [bool(a - b > rise) for b, a in zip(zmax_before, zmax_after)]


def object_rows(x, slices, in_tub, in_basket, tip=None):
    """Per-object scoring rows from the vertex array x (scoring only)."""
    rows = []
    for s0, s1 in slices:
        xo = x[s0:s1]
        rows.append(dict(in_tub_frac=round(float(in_tub(xo).mean()), 4), in_basket_frac=round(float(in_basket(xo).mean()), 4),
                         zmax=round(float(xo[:, 2].max()), 4),
                         tip_dist=None if tip is None else round(float(np.linalg.norm(xo - tip, axis=1).min()), 4)))
    return rows


def n_in_tub(rows):
    return int(sum(r["in_tub_frac"] >= TUB_FRAC for r in rows))


def obj_track_sample(x, slices, tip):
    """(n_obj, 2): nearest vertex to the fingertip centre and highest vertex, per object (one sample while carrying)."""
    return np.array([[np.linalg.norm(x[s0:s1] - tip, axis=1).min(), x[s0:s1, 2].max()] for s0, s1 in slices])


def mask_points(pts, mask, erode=2, min_px=60):
    """Finite points (M, 3) and their pixels (M, 2) of the target mask eroded by up to `erode` px (the edge pixels mix
    in the surface below; falls back to fewer iterations if the mask would shrink under min_px), and the iterations used."""
    from scipy import ndimage
    m, used = np.asarray(mask, bool), 0
    for it in range(erode, 0, -1):
        e = ndimage.binary_erosion(m, structure=np.ones((3, 3)), iterations=it)
        if e.sum() >= min_px:
            m, used = e, it
            break
    vs, us = np.nonzero(m & np.isfinite(pts).all(-1))
    return pts[vs, us].astype(float), np.stack([us, vs], 1), used


def seg_gt_match(obs, mask):
    """(index, share) of the seg_gt object that covers most of the mask (scoring only), or (None, 0)."""
    seg = obs.get("seg_gt")
    if seg is None:
        return None, 0.0
    v = np.asarray(seg)[np.asarray(mask, bool)]
    v = v[v >= 0]
    if not len(v):
        return None, 0.0
    cnt = np.bincount(v.astype(int))
    return int(cnt.argmax()), float(cnt.max() / len(v))


def new_cycle(k, t0):
    c = {key: None for key in CYCLE_KEYS}
    c.update(cycle=k, t0_sim=round(t0, 2), attempts=[], carried=False, lift_ok=False, order=[], recheck=[])
    return c


def pixel_of(pts, pix, p):
    """Pixel (u, v) of the mask point nearest to the 3-D point p."""
    j = int(np.argmin(np.linalg.norm(pts - np.asarray(p, float), axis=1)))
    return [int(pix[j][0]), int(pix[j][1])]


def grasp_candidate(t, pts, exclude=()):
    """The demo's grasp choice on target points `pts` (N,3): (gx, gy, gz, fb, i, centre) and the yaw order, using the
    Test methods (select_grasp, ridge_yaws) exactly as attempt() does."""
    g = t.select_grasp(pts, exclude, center=t.target_xy)
    return g, t.yaw_candidates(pts, g[4])


# ------------------------------------------------------------------ the test
class _MultiRun(MultiCloth):
    """Sits below IntegratedTest in the MRO so that IntegratedTest.run() (its finally block: phases, phase_wall_s,
    phase_sim_s, cameras) wraps the multi-garment scenario instead of the single-towel Test.run."""

    def run(self):
        return self.run_multi()


class MultiIntegratedTest(IntegratedTest, _MultiRun):
    resense = False         # the heap handed to attempt() is the target's mask points, not the cloth entity's vertices
    cur = None              # dict(k, n, label) of the running cycle (captions, phase stamps)
    target_xy = None

    def drop_envelope(self):
        return drop_keepout(DROP_STACK_TOP, 0.0)

    # ---------------- setup ----------------
    def make_cameras(self, scene):
        super().make_cameras(scene)
        self.parked = bool(self.args.no_drive)
        self.caption = "garments dropped into the basket"
        n = len(self.obj_slices)
        self.cycles, self.cards, self.objs, self.loaded = [], [], [], {}
        self.n_first = None
        self.card_n = 0                    # card frames written so far
        self.obj_track = []                # per carry sample: (n_obj, 2) nearest-to-tip, zmax
        self.obj_max_stretch = np.zeros(n)
        starts = np.array([s0 for s0, _ in self.obj_slices])
        self.edge_obj = np.searchsorted(starts, self.edges[:, 0], side="right") - 1
        self.label_by_gt = {}
        self._tub_n_cache = (-1, 0)

    # ---------------- phases / captions ----------------
    def set_phase(self, name, caption):
        cur = self.cur
        if cur and name in ACTIONS:
            caption = caption_fmt(cur["k"], cur["n"], cur.get("label"), ACTIONS[name])
        n0 = len(self.phases)
        super().set_phase(name, caption)
        if cur and len(self.phases) > n0:
            self.phases[-1].update(cycle=cur["k"], target=cur.get("label"))

    def n_in_tub_now(self):
        if self._tub_n_cache[0] != self.nstep:
            rows = object_rows(self.cloth_pos(), self.obj_slices, self.in_tub, self.in_basket)
            self._tub_n_cache = (self.nstep, n_in_tub(rows))
        return self._tub_n_cache[1]

    def in_tub_pct(self):   # IntegratedTest.in_tub_pct is the single-cloth percentage: aggregate = share of vertices
        return 100.0 * float(self.in_tub(self.cloth_pos()).mean())

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
                self.set_phase("settle", "garments settle in the basket")
        img = np.asarray(cam.render()[0]).astype(np.uint8)
        cap = self.caption
        if self.phase == "in tub":
            cur = self.cur or {}
            cap = f"in washer {self.n_in_tub_now()}/{len(self.obj_slices)} (sim)"
            if cur:
                cap = caption_fmt(cur["k"], cur["n"], cur.get("label"), cap)
        img = overlay(img, top=f"1x real time   t = {self.nstep * cwt.DT:5.1f} s", bottom=cap)
        self.proc.stdin.write(img.tobytes())

    def show_card(self, img, seconds):
        """Card frames (simulation paused); returns the mp4 frame range [frame0, frame1) of the card."""
        f0 = self.nframes + self.card_n
        self.card_frames(img, seconds)
        n = int(round(seconds * cwt.FPS))
        self.card_n += n
        return f0, f0 + n

    # ---------------- scoring in step() ----------------
    def step(self):
        super().step()
        if self.nstep % 5 or self.prev_x is None:
            return
        x = self.prev_x
        ln = np.linalg.norm(x[self.edges[:, 0]] - x[self.edges[:, 1]], axis=1) / self.rest_len
        np.maximum.at(self.obj_max_stretch, self.edge_obj, ln)
        ct = self.carry_track
        if ct is not None and len(ct):
            if len(ct) == 1:          # the first sample of a new lift: restart the per-object track
                self.obj_track = []
            self.obj_track.append(obj_track_sample(x, self.obj_slices, self.tcp_world()))

    # ---------------- drive ----------------
    def after_settle(self, res):
        a = self.args
        res["settled_first_t"] = round(self.nstep * cwt.DT, 2)
        if a.no_drive:     # the robot was built at the parked origin (main sets base_xy0): nothing to drive
            res["drive"] = None
            self.parked = True
            return False
        self.drive()
        self.parked = True
        self.base_xy0 = (0.0, 0.0)
        self.set_phase("settle2", "garments settle after the drive")
        t0 = self.nstep * cwt.DT
        drift = self.resettle()
        err = np.linalg.norm(self.q[:2])
        res["drive"] = dict(self.drive_info, final_base_xy_mm_err=round(float(err) * 1e3, 2),
                            resettle_s=round(self.nstep * cwt.DT - t0, 2), resettle_last_drift_mm=round(drift * 1e3, 2))
        res["drive_trace"] = dict(cols=idm.TRACE_COLS, dt_s=idm.TRACE_DT, rows=self.drive_trace)
        x = self.cloth_pos()
        res["settled2_in_basket_frac"] = float(self.in_basket(x).mean())
        res["settled2_zmax_above_floor"] = float(x[:, 2].max() - cwt.BASKET_FLOOR)
        if res["settled2_in_basket_frac"] < 0.99:
            raise RuntimeError("garments left the basket during the drive")
        self.qlog = [] if self.T is not None else None   # the measured-q check starts at the parked pose
        return False

    def grasp_center(self, x, k):
        return self.target_xy        # the detection's 3-D centroid (xy): the search centre of select_grasp

    def wait_still(self, max_s):
        t_end = self.nstep * cwt.DT + max_s
        self.hold(0.5)
        xp = self.cloth_pos()
        drift = float("nan")
        while self.nstep * cwt.DT < t_end:
            self.hold(0.5)
            xn = self.cloth_pos()
            drift = float(np.linalg.norm(xn - xp, axis=1).max())
            xp = xn
            if drift < self.args.still_tol:
                break
        return drift

    # ---------------- the scenario ----------------
    def run_multi(self):
        a, res = self.args, self.res
        res["drop"] = self.drop
        res["garments"] = list(self.keys)
        self.stem = str(Path(a.out or "outputs/multi/multi_demo.mp4").with_suffix(""))
        os.makedirs(os.path.dirname(os.path.abspath(self.stem)), exist_ok=True)
        res["cycles"], res["cards"], res["objects_tracked"] = self.cycles, self.cards, self.objs
        self.hold(a.settle)       # initial settle (as Test.run)
        xp = self.cloth_pos()
        drift = float("nan")
        while self.nstep * cwt.DT < a.settle_max:
            self.hold(0.5)
            xn = self.cloth_pos()
            drift = float(np.linalg.norm(xn - xp, axis=1).max())
            xp = xn
            if drift < a.still_tol:
                break
        res["settle_s"] = round(self.nstep * cwt.DT, 2)
        res["settle_last_0.5s_max_drift_mm"] = round(drift * 1e3, 2)
        x0 = self.cloth_pos()
        res["n_vertices"] = len(x0)
        res["settled_in_basket_frac"] = float(self.in_basket(x0).mean())
        res["settled_zmax_above_floor"] = float(x0[:, 2].max() - cwt.BASKET_FLOOR)
        res["settled_zmin_minus_floor"] = float(x0[:, 2].min() - cwt.BASKET_FLOOR)
        res["pen_settled"] = self.penetration(x0)
        if a.save_obs:
            res["save_obs"] = self.save_obs(a.save_obs, x0)
        if res["settled_in_basket_frac"] < 0.99:
            raise RuntimeError("garments did not settle inside the basket")
        if a.stop_after == "settle":
            return res
        self.after_settle(res)
        stop, k = None, 0
        while stop is None:
            k += 1
            stop = stop_reason(k, a.max_cycles, self.nstep * cwt.DT, a.sim_budget)
            if stop:
                break
            try:
                stop = self.run_cycle(k)
            except cwt.PlanningError as e:
                self.cycles[-1]["end"], self.cycles[-1]["error"] = "planning_error", str(e)[:300]
                stop = "planning_error"
            self.cycles[-1] = clean(self.cycles[-1])
        res["stop_reason"] = stop
        self.cur = None
        res["objects_tracked"] = clean(self.objs)
        self.finish(stop)
        return res

    def finish(self, stop):
        a, res = self.args, self.res
        if a.stop_after in ("vlm", "lift") or stop == "stop_after":
            watch = False
        else:
            watch = True
        if watch:
            self.view = self.tub_cam
            self.hold(a.post_s)
        x = self.cloth_pos()
        tip = self.tcp_world()
        rows = object_rows(x, self.obj_slices, self.in_tub, self.in_basket, tip)
        out = []
        for i, (key, r) in enumerate(zip(self.keys, rows)):
            out.append(dict(name=G.NAMES[key], label_by_vlm=self.label_by_gt.get(i),
                            where=classify_where(r["in_tub_frac"], r["in_basket_frac"],
                                                 r["tip_dist"] is not None and r["tip_dist"] < NEAR_TIP),
                            cycle_loaded=self.loaded.get(i), in_tub_frac=r["in_tub_frac"],
                            in_basket_frac=r["in_basket_frac"],
                            max_edge_stretch=round(float(self.obj_max_stretch[i]), 3)))
        res["objects_final"] = out
        res["n_objects"] = len(out)
        res["n_in_tub_final"] = sum(o["where"] == "tub" for o in out)
        res["success"] = bool(out and res["n_in_tub_final"] == len(out))
        res["cards"] = self.cards
        res["pen_end"] = self.penetration(x)

    # ---------------- perception ----------------
    def run_vlm(self, obs):
        a = self.args
        mock = a.vlm == "mock"
        res_v, dets = {}, []
        rk = (lambda im: a.mock_recheck) if mock and a.mock_recheck is not None else None
        for _ in range(2):        # one retry on a VLM / parse error (e.g. the model still loading)
            t0 = time.time()
            res_v, dets = vlm_multi.run(obs, vlm=(lambda im: a.mock_answer) if mock else None, model=a.vlm_model,
                                        timeout=a.vlm_timeout, recheck=rk)
            if res_v["error"] == "parse_failure" and is_empty_reply(res_v["raw"]):
                # a valid "no garment" answer, not a failure: go on to the basket-empty check (+ residual recheck)
                if rk is None and not mock:
                    rk = lambda im: vlm_grasp.ask(im, a.vlm_model, vlm_grasp.OLLAMA, a.vlm_timeout,
                                                  prompt=vlm_multi.RECHECK_PROMPT)
                return empty_result(obs, res_v["raw"], time.time() - t0, rk)
            if not res_v["error"]:
                break
        return res_v, dets

    def perceive(self, k, x, cyc):
        self.set_phase("vlm", ACTIONS["vlm"])
        obs_path = f"{self.stem}_obs_c{k}.npz"
        self.save_obs(obs_path, x)
        cyc["obs"] = obs_path
        cyc["objects_at_obs"] = clean(self.res.get("objects"))
        obs = vlm_grasp.load_obs(obs_path)
        t0 = time.time()
        res_v, dets = self.run_vlm(obs)
        lat = time.time() - t0
        res_v["prompt"], res_v["recheck_prompt"] = vlm_multi.PROMPT, vlm_multi.RECHECK_PROMPT
        vlm_path = f"{self.stem}_vlm_c{k}.json"
        Path(vlm_path).write_text(json.dumps(res_v, indent=1, default=vlm_multi._jsonable))
        cyc.update(vlm_json=vlm_path, vlm_latency_s=round(lat, 2), n_detections=len(dets),
                   order=[dets[i]["label"] for i in res_v["order"]],
                   recheck=clean([dict(e, raw=None) for e in res_v.get("recheck", [])]), vlm_error=res_v["error"])
        ev = res_v.get("evaluation")
        if ev:
            for o in ev["objects"]:          # scoring only: which simulator object each detection matched
                if o["label"] != "unknown":
                    self.label_by_gt[o["gt_index"]] = o["label"]
        return obs, res_v, dets, lat

    def vlm_card(self, k, obs, res_v, dets, lat, cyc, note):
        from PIL import Image
        a = self.args
        card_png = f"{self.stem}_card_c{k}.png"
        if dets:
            vlm_multi.draw(obs, res_v, dets, card_png)
        else:
            Image.fromarray(np.asarray(obs["rgb"], np.uint8)).save(card_png)
        annotated = np.asarray(Image.open(card_png).convert("RGB"))
        title = "VLM finds the garments" + (" (mock answer)" if a.vlm == "mock" else f" ({a.vlm_model})")
        card = make_card(annotated, (cwt.W, cwt.H), title, f"simulation paused during VLM inference ({lat:.1f} s)", note)
        f0, f1 = self.show_card(card, max(lat, a.card_s))
        info = dict(cycle=k, frame0=f0, frame1=f1, sim_t=round(self.nstep * cwt.DT, 2), obs=cyc["obs"],
                    vlm_json=cyc["vlm_json"], png=card_png)
        self.cards.append(info)
        cyc["card"] = info

    # ---------------- one cycle ----------------
    def run_cycle(self, k):
        """Returns a stop reason (the loop ends) or None (next cycle)."""
        a, res = self.args, self.res
        cyc = new_cycle(k, self.nstep * cwt.DT)
        self.cycles.append(cyc)
        self.cur = dict(k=k, n=self.n_first if self.n_first is not None else "?", label=None)
        x = self.cloth_pos()
        obs, res_v, dets, lat = self.perceive(k, x, cyc)
        conf = [i for i in res_v["order"] if is_confirmed(dets[i])]
        cyc["n_confirmed"] = len(conf)
        if self.n_first is None:
            self.n_first = len(conf)
            res["n_confirmed_first"] = self.n_first
        self.cur["n"] = self.n_first
        i, obj, why = choose_target(dets, res_v["order"], self.objs, a.attempts_per_object, k)
        if i is None:
            garment_px = (res_v.get("segmentation") or {}).get("garment_px")
            stop = no_target_stop(why, garment_px)
            self.vlm_card(k, obs, res_v, dets, lat, cyc, f"no garment to pick ({stop})")
            cyc["end"] = stop
            cyc["n_in_tub_after"] = n_in_tub(object_rows(self.cloth_pos(), self.obj_slices, self.in_tub, self.in_basket))
            cyc["t1_sim"] = round(self.nstep * cwt.DT, 2)
            return stop
        d = dets[i]
        self.cur["label"] = d["label"]
        pts_img = vlm_multi.get_points(obs)
        pts, pix, erode_used = mask_points(pts_img, d["mask"], a.erode)
        gt_i, gt_share = seg_gt_match(obs, d["mask"])
        cyc["target"] = dict(label=d["label"], label_raw=d.get("label_raw"), source=d["source"], obj_id=obj["id"],
                             centroid_3d=d["centroid_3d"], centroid_px=d["centroid_px"], mask_px=int(d["mask"].sum()),
                             n_points=len(pts), erode_px=erode_used, height_q=d["height_q"], seg_gt_index=gt_i,
                             seg_gt_share=round(gt_share, 3),
                             seg_gt_name=None if gt_i is None else G.NAMES[self.keys[gt_i]])
        self.target_xy = centroid_xy(d)
        self.vlm_card(k, obs, res_v, dets, lat, cyc,
                      f"cycle {k}: {len(conf)} garment(s) confirmed; next: {d['label']} (highest first)")
        if a.stop_after == "vlm":
            cyc["end"] = "stop_after_vlm"
            return "stop_after"
        self.plan_exec(self.T.lower_to_grasp, "lower to grasp lift", 1.5)
        gx = gy = None
        tried, ok, plan_errs = [], False, 0
        for j in range(max(1, min(a.attempts, a.attempts_per_object - obj["failed"]))):
            if j > 0:
                last = cyc["attempts"][-1]
                if not last.get("miss"):   # lift failed: open, lower the arm back into the basket, let things settle
                    self.p_grip("retry_open", cwt.OPEN, 0.5)
                    self.plan_exec(self.T.lift_to, "retry_lower", cwt.LIFT_GRASP, 1.5)
                self.hold(a.retry_settle)
            xb = self.cloth_pos()
            zb = [float(xb[s0:s1, 2].max()) for s0, s1 in self.obj_slices]
            try:
                gx, gy, fb, off = self.attempt(pts, tried)
            except AssertionError:       # select_grasp: no candidate left (all excluded / inside the wall band)
                cyc["attempts"].append(dict(error="no_grasp_candidate", miss=True, lift_ok=False))
                obj["given_up"], obj["reason"] = True, "no_grasp_candidate"
                break
            except cwt.PlanningError as e:
                plan_errs += 1
                cyc["attempts"].append(dict(error=f"planning: {str(e)[:200]}", miss=True, lift_ok=False))
                obj["failed"] += 1
                if plan_errs >= 2:
                    raise
                continue
            plan_errs = 0
            tried.append((gx, gy))
            xa = self.cloth_pos()
            lifted = lifted_flags(zb, [float(xa[s0:s1, 2].max()) for s0, s1 in self.obj_slices])
            lo = lift_ok(res)
            rec = {key: res.get(key) for key in (
                "miss", "stall", "weak_hold", "lift_slide_mm", "ipc_squeeze_N_closed", "ipc_fz_N_closed",
                "joint7_8_closed", "grasp_point", "grasp_yaw", "ridge_anisotropy", "finger_bottom_at_grasp_minus_floor",
                "verts_between_fingers_open", "verts_between_fingers_closed", "touch_closed", "touch_lifted",
                "lift_lost_at_tip_dz", "lift_dzmean", "lifted_tip_to_nearest_vertex", "plan_fail")}
            rec.update(lift_ok=lo, lifted_objects=[G.NAMES[self.keys[o]] for o, f in enumerate(lifted) if f],
                       double_grasp=sum(lifted) > 1, grasp_px=pixel_of(pts, pix, res["grasp_point"]) if res.get("grasp_point") else None,
                       t_sim=round(self.nstep * cwt.DT, 2))
            cyc["attempts"].append(clean(rec))
            if lo:
                ok = True
                break
            obj["failed"] += 1
        cyc["lift_ok"] = ok
        if not ok and obj["failed"] >= a.attempts_per_object:
            obj["given_up"], obj["reason"] = True, obj["reason"] or "attempts"
        if a.stop_after == "lift":
            cyc["end"] = "stop_after_lift"
            return "stop_after"
        if ok:
            stop = self.carry_release_retract(cyc, obj, gx, gy)
        else:
            cyc["end"] = "given_up" if obj["given_up"] else "lift_failed"
            self.recover_to_standby("cycle failed")
            stop = None
        x_end = self.cloth_pos()
        rows = object_rows(x_end, self.obj_slices, self.in_tub, self.in_basket)
        if cyc["objects"] is None:
            cyc["objects"] = dict(after=rows)
        cyc["n_in_tub_after"] = cyc.get("n_in_tub_after") if cyc.get("n_in_tub_after") is not None else n_in_tub(rows)
        for o, r in enumerate(rows):
            if r["in_tub_frac"] >= TUB_FRAC and o not in self.loaded:
                self.loaded[o] = k
        if cyc.get("carried") and cyc["end"] is None:
            cyc["end"] = "loaded" if any(self.loaded.get(o) == k for o in range(len(rows))) else "released_not_in_tub"
        if stop is None:
            self.wait_still(3.0)
        cyc["t1_sim"] = round(self.nstep * cwt.DT, 2)
        return stop

    def recover_to_standby(self, tag):
        """No carry: open the fingers if closed, lift joint to LIFT_CARRY, planned way back to the standby fold."""
        self.carry_track = None
        if self.cmd[10] < self.args.pre_open - 1e-3:
            self.p_grip(f"retry_open ({tag})", cwt.OPEN, 0.5)
        if abs(self.T.q[3] - cwt.LIFT_CARRY) > 1e-6:
            self.plan_exec(self.T.lift_to, f"retry_raise ({tag})", cwt.LIFT_CARRY, 1.5)
        self.plan_exec(self.T.to_standby, 1.5)

    def wiggle_wrist(self):
        base = np.array(self.T.q, float)
        for s, tag in ((1, "+"), (-1, "-"), (0, "0")):
            q = base.copy()
            q[J6] += s * math.radians(WIGGLE_DEG)
            self.plan_exec(self.T.move_dofs, f"release_wiggle{tag}", q, 0.4)
        self.hold(1.0)

    def release_check(self, x4):
        """After opening: if EITHER finger still touches cloth (aggregated contacts), wiggle the wrist and look again;
        if still touching, open to the maximum width and wiggle once more.  Every step is recorded."""
        rc = dict(enabled=bool(self.args.release_check), touch_after_open=self.touching(x4), wiggled=False, steps=[])
        if not (rc["enabled"] and max(rc["touch_after_open"]) > 0):
            return rc
        try:
            for n in (1, 2):
                if n == 2:
                    self.p_grip("release_open_max", cwt.OPEN, 0.6)    # the joint7 upper limit
                    rc["steps"].append(dict(step="open_max", opening=cwt.OPEN))
                self.wiggle_wrist()
                t = self.touching(self.cloth_pos())
                rc["wiggled"] = True
                rc["steps"].append(dict(step=f"wiggle{n}", touch_after=t))
                rc["touch_after_wiggle"] = t
                if max(t) == 0:
                    break
        except cwt.PlanningError as e:
            rc["wiggle_error"] = str(e)[:200]
        return rc

    def carry_release_retract(self, cyc, obj, gx, gy):
        """Carry over the opening, release, release check, retract, tub check.  Returns a stop reason or None."""
        a, res = self.args, self.res
        cyc["carried"] = True
        try:
            rx, ry = P.washer_to_base(a.release_dx, a.release_dy, cwt.WASHER_XY, math.radians(cwt.WASHER_YAW_DEG))
            self.plan_exec(self.T.raise_, a.raise_s)
            self.plan_exec(self.T.carry, rx, ry, a.carry_s / a.n_wp)
        except cwt.PlanningError as e:   # stuck above the basket with the garment: drop it and go home
            cyc["carry_plan_fail"] = str(e)[:300]
            cyc["end"] = "carry_plan_failed"
            self.recover_to_standby("carry planning failed")
            return None
        self.hold(a.swing_s)
        x3 = self.cloth_pos()
        tip3 = self.tcp_world()
        tr = np.array(self.carry_track)
        otr = np.array(self.obj_track)
        self.carry_track = None
        rows3 = object_rows(x3, self.obj_slices, self.in_tub, self.in_basket, tip3)
        lifted = [bool(otr[:, o, 1].max() - otr[0, o, 1] > LIFT_RISE) if len(otr) else False
                  for o in range(len(self.obj_slices))]
        per = []
        for o, (s0, s1) in enumerate(self.obj_slices):
            per.append(dict(name=G.NAMES[self.keys[o]], lifted=lifted[o],
                            held_until_release=bool(len(otr) and otr[:, o, 0].max() < HELD_DIST),
                            over_opening_frac=round(float(self.in_tub_xy(x3[s0:s1]).mean()), 4),
                            max_edge_stretch=round(float(self.obj_max_stretch[o]), 3),
                            pen_carry_end=self.penetration(x3[s0:s1])))
        res["carry_max_tip_to_nearest_vertex"] = float(tr[:, 0].max()) if len(tr) else None
        res["release_tip"] = [round(float(v), 3) for v in tip3]
        res["pen_carry_end"] = self.penetration(x3)
        self.p_grip("release", cwt.OPEN, 0.6)
        self.hold(2.0)
        x4 = self.cloth_pos()
        cyc["release_check"] = self.release_check(x4)
        x4 = self.cloth_pos()
        for o, (s0, s1) in enumerate(self.obj_slices):
            per[o]["in_tub_frac_release+2s"] = round(float(self.in_tub(x4[s0:s1]).mean()), 4)
        stop = None
        try:
            self.plan_exec(self.T.to_standby, 1.5)
        except cwt.PlanningError as e:
            cyc["retract_plan_fail"] = str(e)[:300]
            stop = "retract_plan_fail"
        self.view = self.tub_cam
        self.hold(1.0)                  # look at the tub
        x5 = self.cloth_pos()
        rows5 = object_rows(x5, self.obj_slices, self.in_tub, self.in_basket)
        for o in range(len(per)):
            per[o].update(in_tub_frac_end=rows5[o]["in_tub_frac"], in_basket_frac_end=rows5[o]["in_basket_frac"],
                          pen_end=self.penetration(x5[self.obj_slices[o][0]:self.obj_slices[o][1]]))
        cyc["objects"] = dict(carry=rows3, per_object=per, after=rows5)
        cyc["n_in_tub_after"] = n_in_tub(rows5)
        cyc["lifted_objects"] = [p["name"] for p in per if p["lifted"]]
        self.view = self.cam            # back to the entrance camera
        return stop


# ------------------------------------------------------------------ CLI
def make_parser():
    ap = idm.make_parser()
    ap.add_argument("--garments", default="towel,tshirt,sock", help="comma list of %s, drop order (lowest first)"
                    % "/".join(G.GARMENTS))
    ap.add_argument("--garment-gap", type=float, default=0.04, help="initial z gap between a garment's highest vertex "
                    "and the next one's lowest [m]")
    ap.add_argument("--tilt-max", type=float, default=10.0, help="max random tilt of a flat garment about x / y [deg]")
    ap.add_argument("--no-drive", action="store_true", help="skip the drive: the robot is built at the parked origin "
                    "and the scenario starts at the washer (R1 / R2 tests; the house is still in the scene)")
    ap.add_argument("--max-cycles", type=int, default=5, help="stop after this many pick cycles")
    ap.add_argument("--sim-budget", type=float, default=140.0, help="do not start a cycle when the simulated time is "
                    "beyond this [s] (0 = unlimited)")
    ap.add_argument("--attempts-per-object", type=int, default=3, help="failed pinches per perceived object before it "
                    "is given up (a cycle makes at most min(--attempts, what is left of this) attempts)")
    ap.add_argument("--no-release-check", dest="release_check", action="store_false",
                    help="do not wiggle the wrist when a finger still touches cloth after opening")
    ap.add_argument("--erode", type=int, default=2, help="erode the target mask by this many px before taking its "
                    "points (edge pixels mix in the surface below)")
    ap.add_argument("--mock-recheck", default=None, help="--vlm mock: reply text of the leftover-region question")
    ap.add_argument("--offline-check", default=None, metavar="OBS[,OBS...]", help="CPU: grasp candidates + planner "
                    "check on saved observations (with --offline-vlm), then exit")
    ap.add_argument("--offline-vlm", default=None, metavar="JSON[,JSON...]", help="vlm_multi result JSONs (their raw "
                    "answers are replayed as the VLM) matching --offline-check")
    ap.set_defaults(out="outputs/multi/multi_demo.mp4", json="outputs/multi/multi_demo.json", semi_implicit="off")
    return ap


def main():
    ap = make_parser()
    args = ap.parse_args()
    cwt.W, cwt.H = (int(v) for v in args.res.lower().split("x"))
    if args.selftest:
        return selftest()
    if args.offline_check:
        return offline_check(args)
    if args.vlm == "off":
        ap.error("the multi-garment demo needs the VLM (--vlm ollama, or --vlm mock with --mock-answer boxes)")
    if not args.planner:
        ap.error("the multi-garment demo needs --planner")
    if args.no_drive:
        MultiIntegratedTest.base_xy0 = (0.0, 0.0)       # built at the parked origin = planner frame
    elif args.start_room:
        idm.set_start_room([float(v) for v in args.start_room.split(",")])   # sets IntegratedTest.base_xy0 (inherited)
    cwt.main(MultiIntegratedTest, ap)


# ------------------------------------------------------------------ offline check (CPU planner on saved observations)
def bare_test(args):
    """A MultiIntegratedTest without a simulation (no gs.init / scene): enough for select_grasp / ridge_yaws."""
    t = MultiIntegratedTest.__new__(MultiIntegratedTest)
    t.args, t.res = args, {}
    t.base_xy0 = (0.0, 0.0)
    t.close_axis_l6 = CLOSE_AXIS_L6
    t.T = object()          # yaw_candidates returns all 24 yaws when a planner exists
    return t


def replay_vlm(obs, vlm_json):
    """vlm_multi.run on `obs` replaying the raw first answer and the recheck answers (in order) of a saved result."""
    saved = json.loads(Path(vlm_json).read_text())
    rk = iter([e.get("raw") for e in saved.get("recheck", [])])
    return vlm_multi.run(obs, vlm=lambda im: saved["raw"], recheck=lambda im: next(rk))


def offline_check(args):
    obs_paths = args.offline_check.split(",")
    vlm_paths = (args.offline_vlm or "").split(",")
    if len(obs_paths) != len(vlm_paths):
        raise SystemExit("--offline-vlm needs one JSON per --offline-check observation")
    ix, iy, _ = P.BASKET_INNER
    bx, by = P.BASKET_CENTER_XY
    t = bare_test(args)
    rows, first = [], None
    for op, vp in zip(obs_paths, vlm_paths):
        obs = vlm_grasp.load_obs(op)
        res_v, dets = replay_vlm(obs, vp)
        pts_img = vlm_multi.get_points(obs)
        names = [str(n) for n in obs["obj_names"]] if "obj_names" in obs else None
        for i in res_v["order"]:
            d = dets[i]
            if not is_confirmed(d):
                continue
            pts, pix, er = mask_points(pts_img, d["mask"], args.erode)
            t.target_xy = centroid_xy(d)
            row = dict(obs=Path(op).name, label=d["label"], source=d["source"], n_points=len(pts), erode=er)
            try:
                g, yaws = grasp_candidate(t, pts)
            except AssertionError as e:
                row.update(error=str(e))
                rows.append(row)
                continue
            gx, gy, gz, fb, j, _ = g
            u, v = pix[j]
            seg = obs.get("seg_gt")
            gt = None if seg is None else int(seg[v, u])
            mgt, share = seg_gt_match(obs, d["mask"])
            row.update(grasp=[round(gx, 3), round(gy, 3), round(gz, 3)], fb=round(fb, 3), px=[int(u), int(v)],
                       in_own_gt=None if gt is None else bool(gt == mgt and gt >= 0), gt_at_point=None if gt is None or gt < 0 or not names else names[gt],
                       outside_wall_band=bool(abs(gx - bx) < ix / 2 - vlm_grasp.WALL_BAND and abs(gy - by) < iy / 2 - vlm_grasp.WALL_BAND),
                       yaws_first=[round(y) for y in yaws[:4]], mask_gt=None if mgt is None else names[mgt] if names else mgt,
                       mask_gt_share=round(share, 3))
            row["_cand"] = (gx, gy, gz, fb, yaws)
            rows.append(row)
    print("candidates (perception only: mask points of the point cloud):")
    for r in rows:
        print("  " + json.dumps({k: v for k, v in r.items() if k != "_cand"}))
    # planner (CPU child process, as in the demo): approach + descend per candidate, then consecutive full cycles
    from arm_planner import PlannerClient
    t0 = time.time()
    T = PlannerClient(dict(washer_pos=(*cwt.WASHER_XY, cwt.WASHER_Z), washer_yaw_deg=cwt.WASHER_YAW_DEG,
                           margin=args.planner_margin, with_room=args.planner_room, seed=args.seed),
                      dict(lift_grasp=cwt.LIFT_GRASP, lift_carry=cwt.LIFT_CARRY, open_=cwt.OPEN, lift1_z=args.lift1_z,
                           carry_z=args.carry_z, n_wp=args.n_wp, tilt_step=args.tilt_step, wait_lift=cwt.LIFT_CARRY,
                           keepout=drop_keepout(DROP_STACK_TOP, 0.0)))
    try:
        st = T.start()
        print(f"planner built {T.build_s:.1f} s, standby at lift {cwt.LIFT_CARRY}: {'ok' if st is not None else 'NONE'} "
              f"({T.standby_s:.1f} s)")
        for r in rows:
            c = r.get("_cand")
            if c is None:
                continue
            gx, gy, gz, fb, yaws = c
            T.reset(T.standby)
            tt = time.time()
            ok = T.lower_to_grasp("lower", 1.5) and T.grip("pre_open", args.pre_open, 0.3)
            ok = ok and T.approach("hover", (gx, gy, gz), fb, yaws, 2.0)
            hov_ok = ok
            ok = ok and T.descend(args.descend_s)
            seg = [s for s in T.segs if s["name"] == "hover"]
            r["plan"] = dict(approach=bool(hov_ok), descend=bool(ok), yaw=T.target["yaw"] if ok else None,
                             yaws_tried=seg[0]["res"].info.get("yaws_tried") if seg else None,
                             plan_s=round(time.time() - tt, 1))
            print(f"  {r['obs']} {r['label']:8s} grasp {r['grasp']} -> {r['plan']}")
        # three consecutive full cycles (first observation's targets, highest first)
        print("three consecutive cycles (lower -> pre_open -> approach -> descend -> close -> lift -> carry -> release "
              "-> standby):")
        T.reset(T.standby)
        cyc_rows = [r for r in rows if r.get("_cand") and r["obs"] == rows[0]["obs"]][:3]
        rx, ry = P.washer_to_base(args.release_dx, args.release_dy, cwt.WASHER_XY, math.radians(cwt.WASHER_YAW_DEG))
        for n, r in enumerate(cyc_rows, 1):
            gx, gy, gz, fb, yaws = r["_cand"]
            tt = time.time()
            steps = []
            ok = True
            for name, fn in (("lower", lambda: T.lower_to_grasp("lower", 1.5)),
                             ("pre_open", lambda: T.grip("pre_open", args.pre_open, 0.3)),
                             ("approach", lambda: T.approach("hover", (gx, gy, gz), fb, yaws, 2.0)),
                             ("descend", lambda: T.descend(args.descend_s)),
                             ("close", lambda: T.close(args.close_s)),
                             ("lift1", lambda: T.lift1(args.lift1_s)),
                             ("lift2", lambda: T.lift_to("lift2", cwt.LIFT_CARRY, args.lift2_s)),
                             ("raise", lambda: T.raise_(args.raise_s)),
                             ("carry", lambda: T.carry(rx, ry, args.carry_s / args.n_wp)),
                             ("release", lambda: T.release(0.6)),
                             ("standby", lambda: T.to_standby(1.5))):
                t1 = time.time()
                ok = bool(fn())
                steps.append(f"{name}:{'ok' if ok else 'FAIL'}:{time.time() - t1:.1f}s")
                if not ok:
                    print("    reason:", T.segs[-1]["res"].reason[:200])
                    break
            print(f"  cycle {n} ({r['label']}): {' '.join(steps)}  total {time.time() - tt:.1f} s")
            r["cycle_ok"] = ok
            if not ok:
                break
    finally:
        T.shutdown()
    print(f"offline check done in {time.time() - t0:.0f} s")
    bad = [r for r in rows if "error" in r or not r.get("outside_wall_band", False) or r.get("in_own_gt") is False
           or not r.get("plan", {}).get("descend", False)]
    print("ALL OK" if not bad and all(r.get("cycle_ok", True) for r in rows) else f"{len(bad)} candidate(s) need a look")
    return 0


# ------------------------------------------------------------------ selftest (CPU, no Genesis scene)
def selftest():
    ok = True

    def chk(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name, info)

    ap = make_parser()
    args = ap.parse_args([])
    chk("parser: integrated + multi flags", args.no_drive is False and args.max_cycles == 5 and args.attempts_per_object == 3
        and args.release_check is True and args.garments == "towel,tshirt,sock" and args.sim_budget == 140.0
        and args.semi_implicit == "off" and args.planner is True)
    chk("--no-release-check", ap.parse_args(["--no-release-check"]).release_check is False)
    # drop keep-out covers the stacked drop of every seed
    import multi_cloth_scene as M
    fl = cwt.BASKET_FLOOR
    top = max(M.plan_drops(["towel", "tshirt", "sock"], np.random.default_rng(s), cwt._basket_xy(MultiIntegratedTest),
                           fl, args.drop_h, args.garment_gap, args.lean_max, args.tilt_max)[-1]["top"] - fl
              for s in range(200))
    lo, hi, dist = drop_keepout(DROP_STACK_TOP, 0.0)
    chk("drop keep-out above the stacked drop (200 seeds)", hi[2] - fl >= top + dist - 1e-9,
        f"top {top:.3f} m, box top {hi[2] - fl:.3f} m (single towel {drop_keepout(0.12, 0.30)[1][2] - fl:.3f} m)")
    t0 = cwt.Test.__new__(cwt.Test)
    t0.args = args
    chk("hook: default drop_envelope = the single-towel one", t0.drop_envelope() == drop_keepout(args.drop_h, cwt.TOWEL))
    chk("hook: multi drop_envelope taller", MultiIntegratedTest.__new__(MultiIntegratedTest).__class__.drop_envelope(
        types.SimpleNamespace(args=args))[1][2] > t0.drop_envelope()[1][2] and cwt.Test.resense is True
        and MultiIntegratedTest.resense is False)
    # MRO: IntegratedTest.run wraps the multi scenario, MultiCloth parts come from the scene class
    mro = [c.__name__ for c in MultiIntegratedTest.__mro__]
    chk("MRO", mro[:4] == ["MultiIntegratedTest", "IntegratedTest", "_MultiRun", "MultiCloth"], mro[:5])
    chk("run chain", MultiIntegratedTest.run is IntegratedTest.run and IntegratedTest.run is not cwt.Test.run
        and MultiIntegratedTest.add_cloth is MultiCloth.add_cloth and MultiIntegratedTest.save_obs is MultiCloth.save_obs)
    # ---- association / target selection / given_up / stop
    objs = []
    det = lambda label, xy, src="vlm", st="ok", h=0.1: dict(label=label, centroid_3d=[xy[0], xy[1], 0.3], source=src, status=st, height_q=h)
    dets = [det("t-shirt", (0.05, -0.05)), det("towel", (0.0, 0.0), "recheck"), det("sock", (0.1, 0.0), "vlm", "rejected"),
            det("sock", (0.1, 0.02), "geometry")]
    i, o, why = choose_target(dets, [0, 1], objs, 3, 1)
    chk("choose: highest confirmed first", i == 0 and o["id"] == 1 and why is None and len(objs) == 2)
    chk("confirmed = ok + vlm/recheck only", [is_confirmed(d) for d in dets] == [True, True, False, False])
    chk("associate: same label within 6 cm", associate(objs, "t-shirt", (0.08, -0.05)) is objs[0]
        and associate(objs, "t-shirt", (0.2, -0.05)) is None and associate(objs, "sock", (0.05, -0.05)) is None)
    objs[0]["failed"] = 3
    i, o, why = choose_target(dets, [0, 1], objs, 3, 2)
    chk("choose: failed >= max -> given up, next object", i == 1 and objs[0]["given_up"] and objs[0]["reason"] == "attempts")
    objs[1]["failed"] = 3
    i, o, why = choose_target(dets, [0, 1], objs, 3, 3)
    chk("choose: all given up", i is None and why == "all_given_up" and no_target_stop(why, 5000) == "all_given_up")
    chk("no detections: nothing left vs unconfirmed pixels", no_target_stop("no_confirmed_detection", 100) == "basket_empty"
        and no_target_stop("no_confirmed_detection", 5000) == "unconfirmed_garment_pixels_left"
        and no_target_stop("no_confirmed_detection", None) == "vlm_error")
    chk("stop: max cycles / budget", stop_reason(6, 5, 10, 140) == "max_cycles" and stop_reason(2, 5, 141, 140) == "sim_budget"
        and stop_reason(2, 5, 90, 140) is None and stop_reason(2, 5, 1e9, 0) is None)
    chk("lift_ok", lift_ok(dict(touch_lifted=[3, 5])) and not lift_ok(dict(touch_lifted=[3, 0]))
        and not lift_ok(dict(touch_lifted=[3, 5], miss=True)) and not lift_ok(dict(touch_lifted=[3, 5], weak_hold=True))
        and not lift_ok(dict(miss=False)) and not lift_ok(dict(touch_lifted=[3, 5], plan_fail="x", miss=True)))
    chk("caption", caption_fmt(2, 3, "t-shirt", "lift") == "[2/3] t-shirt: lift" and caption_fmt(1, 3, None, "x") == "cycle 1: x")
    chk("where", classify_where(0.95, 0, False) == "tub" and classify_where(0.3, 0.6, False) == "basket"
        and classify_where(0.1, 0.1, True) == "robot" and classify_where(0.1, 0.1, False) == "floor")
    chk("lifted flags", lifted_flags([0.1, 0.1, 0.1], [0.2, 0.11, 0.14]) == [True, False, True])
    # ---- mask points / seg match / candidates on the synthetic observation
    obs = vlm_multi._synthetic()
    res_v, dets_s = vlm_multi.run(obs, vlm=lambda im: json.dumps(vlm_multi._gt_boxes(obs, jitter=4)))
    pts_img = vlm_multi.get_points(obs)
    t = bare_test(args)
    t.res = {}
    ix, iy, _ = P.BASKET_INNER
    bx, by = P.BASKET_CENTER_XY
    good = True
    for i in res_v["order"]:
        d = dets_s[i]
        pts, pix, er = mask_points(pts_img, d["mask"], 2)
        t.target_xy = centroid_xy(d)
        g, yaws = grasp_candidate(t, pts)
        gx, gy, gz, fb, j, c = g
        gt = int(obs["seg_gt"][pix[j][1], pix[j][0]])
        gm, share = seg_gt_match(obs, d["mask"])
        wall = abs(gx - bx) < ix / 2 - 0.05 and abs(gy - by) < iy / 2 - 0.05
        good &= (gt == gm >= 0) and wall and len(yaws) == 24 and len(pts) > 100 and er == 2 and np.isfinite(pts).all()
        good &= bool(np.allclose(pts[j], [gx, gy, gz])) and fb >= cwt.BASKET_FLOOR + args.floor_clear - 1e-9
        print(f"     {d['label']:8s} pts {len(pts):5d} grasp ({gx:+.3f},{gy:+.3f},{gz:.3f}) in own gt {gt == gm} yaws {[round(y) for y in yaws[:3]]}")
    chk("synthetic: grasp point in its own mask / gt, 5 cm off the walls, 24 yaws", good)
    pts_small, _, er_small = mask_points(pts_img, dets_s[0]["mask"], 2, min_px=10 ** 9)
    chk("mask_points: erosion falls back to the raw mask", er_small == 0 and len(pts_small) > 0)
    t.target_xy = (float(bx + 0.5), float(by))      # far outside: select_grasp has no candidate inside the points
    try:
        t.select_grasp(np.zeros((0, 3)), center=t.target_xy)
        chk("select_grasp without points raises AssertionError", False)
    except AssertionError:
        chk("select_grasp without points raises AssertionError", True)
    vlm_cases(chk, obs, args)
    # ---- end to end with a scripted fake (no Genesis): cycles, cards, captions, per-object scoring, stops
    chk("loop with a scripted sim", fake_loop_test(chk))
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


def vlm_cases(chk, obs, args):
    """run_vlm: empty list -> zero detections (no error) -> basket_empty; prose / broken JSON -> one retry -> vlm_error."""
    a = make_parser().parse_args(["--vlm", "mock"])
    t = bare_test(a)
    empty = {k: np.copy(v) for k, v in obs.items()}
    for g in range(3):                      # all garments gone from the basket
        m = empty["seg_gt"] == g
        empty["rgb"][m] = (46, 110, 150)
        empty["points"][m, 2] = cwt.BASKET_FLOOR
        empty["seg_gt"][m] = -1
    calls = []

    def ask(text):
        def f(im):
            calls.append(1)
            return text
        return f
    for txt in ("[]", "```json\n[]\n```", " [ ] ", '{"objects": []}'):
        a.mock_answer = txt
        calls.clear()
        res_v, dets = t.run_vlm(empty)
        chk(f"empty reply {txt!r}: no error, zero detections, basket_empty",
            res_v["error"] is None and dets == [] and res_v["order"] == [] and len(calls) == 0
            and no_target_stop("no_confirmed_detection", res_v["segmentation"]["garment_px"]) == "basket_empty",
            res_v["segmentation"]["garment_px"])
    a.mock_answer = "[]"
    res_v, dets = t.run_vlm(obs)               # basket not empty but the VLM says []: unexplained pixels, no target
    chk("empty reply on a full basket: unconfirmed_garment_pixels_left",
        res_v["error"] is None and no_target_stop("no_confirmed_detection", res_v["segmentation"]["garment_px"])
        == "unconfirmed_garment_pixels_left", res_v["segmentation"]["garment_px"])
    a.mock_recheck = '{"garment": true, "label": "sock"}'     # residual recheck still runs after an empty answer
    res_v, dets = t.run_vlm(obs)
    chk("empty reply + recheck confirms: a recheck detection becomes a target", len(dets) >= 1
        and all(d["source"] == "recheck" for d in dets) and is_confirmed(dets[0]) and res_v["order"], [d["source"] for d in dets])
    a.mock_recheck = None
    n_calls = {"n": 0}
    for txt in ("I see no garments in this picture.", '[{"bbox_2d": [10, 10, 50 "label": "sock"'):
        n = [0]
        orig = vlm_multi.run

        def counting(obs_, vlm=None, **kw):
            n[0] += 1
            return orig(obs_, vlm=vlm, **kw)
        vlm_multi.run = counting
        try:
            a.mock_answer = txt
            res_v, dets = t.run_vlm(empty)
        finally:
            vlm_multi.run = orig
        chk(f"unparsable reply {txt[:20]!r}: one VLM retry, then vlm_error", n[0] == 2 and res_v["error"] == "parse_failure"
            and no_target_stop("no_confirmed_detection", (res_v.get("segmentation") or {}).get("garment_px")) == "vlm_error", n[0])
    chk("is_empty_reply", is_empty_reply("[]") and not is_empty_reply("nothing") and not is_empty_reply("[{}]")
        and not is_empty_reply("") and not is_empty_reply('{"a": 1}'))


class _FakeT:
    """The TransferSeq surface the cycle loop touches (state only; plan_exec of FakeSim records the calls)."""

    def __init__(self):
        self.q = np.zeros(12)
        self.q[3] = cwt.LIFT_CARRY
        self.standby = self.q.copy()
        self.segs = []

    def _n(self, name):
        return name

    def lower_to_grasp(self, name, dur): self.q[3] = cwt.LIFT_GRASP
    def grip(self, name, g, dur): self.q[10:12] = (g, -g)
    def lift_to(self, name, lift, dur): self.q[3] = lift
    def move_dofs(self, name, q, dur): self.q = np.array(q, float)
    def raise_(self, dur): pass
    def carry(self, rx, ry, dur): pass
    def to_standby(self, dur): self.q = self.standby.copy()


def fake_loop_test(chk):
    """MultiIntegratedTest.run_multi on three scripted garments: cycle 1 (t-shirt) loads at once, cycle 2 (towel) fails
    twice then loads, cycle 3 (sock) fails three times and is given up; then nothing confirmed is left."""
    import tempfile
    args = make_parser().parse_args(["--vlm", "mock", "--attempts", "5", "--no-drive", "--sim-budget", "0"])
    args.out = os.path.join(tempfile.mkdtemp(prefix="multi_selftest_"), "m.mp4")
    base_obs = vlm_multi._synthetic()
    boxes = vlm_multi._gt_boxes(base_obs, jitter=3)          # gt index order: towel, t-shirt, sock
    bx, by = P.BASKET_CENTER_XY
    fl = cwt.BASKET_FLOOR
    obj_slices = [(0, 20), (20, 40), (40, 60)]
    X = np.zeros((60, 3))
    for o, (s0, s1) in enumerate(obj_slices):
        X[s0:s1] = [bx + 0.02 * o - 0.02, by, fl + 0.02 * (o + 1)]
    outcomes = {"t-shirt": ["ok"], "towel": ["miss", "lift_fail", "ok"], "sock": ["miss", "miss", "miss"]}

    class Sim(MultiIntegratedTest):
        def __init__(self):
            self.args, self.res = args, {}
            self.keys = ["towel", "tshirt", "sock"]
            self.obj_slices = obj_slices
            self.edges = np.array([[0, 1], [20, 21], [40, 41]])
            self.rest_len = np.ones(3) * 0.01
            self.nstep = self.nframes = 0
            self.proc = None
            self.view = "cam"
            self.cam, self.tub_cam = "cam", "tub"
            self.prev_x = None
            self.parked = True
            self.phase, self.caption = "settle", ""
            self.phases = [dict(phase="settle", sim_t=0.0, video_t=0.0, wall_t=0.0)]
            self._wall0 = time.perf_counter()
            self.T = _FakeT()
            self.AP = types.SimpleNamespace(fk_tcp=lambda q: (np.array([0, 0, 0.9]), None))
            self.cmd = np.zeros(12)
            self.close_axis_l6 = CLOSE_AXIS_L6
            self.base_xy0 = (0.0, 0.0)
            self.drop, self.drive_info, self.qlog, self.carry_track = {}, None, None, None
            self.cycles, self.cards, self.objs, self.loaded, self.n_first = [], [], [], {}, None
            self.card_n, self.obj_track = 0, []
            self.obj_max_stretch = np.zeros(3)
            self.edge_obj = np.array([0, 1, 2])
            self.label_by_gt = {}
            self._tub_n_cache = (-1, 0)
            self.x = X.copy()
            self.removed, self.script, self.log = [], {k: list(v) for k, v in outcomes.items()}, []
            self.held = None
            self.video_extra_s = 0.0
            self.touch_after_open = [2, 3]
            self.caps = []
            self.touch_script = [[2, 0], [2, 0], [0, 0]]    # cycle 1: one finger holds on, the wiggle does not help

        # --- simulator stand-ins
        def hold(self, dur): self.nstep += int(round(dur / cwt.DT))
        def cloth_pos(self): return self.x.copy()
        def tcp_world(self): return np.array([bx, by, 0.9])
        def touching(self, x, tol=0.003):
            return list(self.touch_script.pop(0)) if self.touch_script else list(self.touch_after_open)
        def penetration(self, x): return dict(fingers=0, washer_basket=0)
        def p_grip(self, name, v, dur, plan_v=None):
            self.plan_exec(self.T.grip, name, v, dur)
            self.cmd[10:12] = (v, -v)

        def plan_exec(self, fn, *a, grip_to=None):
            names = dict(lower_to_grasp=None, raise_="raise", carry="carry1", to_standby="->standby")
            name = a[0] if a and isinstance(a[0], str) else names[fn.__name__]
            fn(*a)
            self.log.append(name)
            IntegratedTest.follow(self, None, dict(name=name))   # phase / caption mapping, then (patched) Test.follow
            if name == "release":
                self.x[self.held[0]:self.held[1]] = [P.WASHER_BASE_XY[0], P.WASHER_BASE_XY[1], 0.5]
                self.removed.append(self.held_obj)

        def save_obs(self, path, x):
            o = {k: np.copy(v) for k, v in base_obs.items() if k in ("rgb", "points", "K", "E", "seg_gt", "obj_names")}
            for r in self.removed:              # a loaded garment is gone from the basket image
                m = o["seg_gt"] == r
                o["rgb"][m] = (46, 110, 150)
                o["points"][m, 2] = fl
                o["seg_gt"][m] = -1
            o["E_mode"] = np.array("world2cam")
            o["verts"] = x.astype(np.float32)
            np.savez_compressed(path, **o)
            self.res["objects"] = [dict(name=n) for n in o["obj_names"]]
            return dict(path=path)

        def run_vlm(self, obs):     # the mock VLM sees the remaining garments (scripted boxes, no network)
            ans = json.dumps([b for i, b in enumerate(boxes) if i not in self.removed])
            return vlm_multi.run(obs, vlm=lambda im: ans)

        def attempt(self, x0, exclude):
            label = self.cur["label"]
            g = self.select_grasp(x0, exclude, center=self.target_xy)
            gx, gy, gz, fb_t, i, c = g
            yaws = self.yaw_candidates(x0, i)
            out = self.script[label].pop(0)
            r = self.res
            for k in ("miss", "weak_hold", "touch_lifted", "plan_fail"):
                r.pop(k, None)
            r.update(grasp_point=[round(gx, 3), round(gy, 3), round(gz, 3)], grasp_yaw=yaws[0], miss=out == "miss",
                     weak_hold=False)
            if out != "miss":
                r["touch_lifted"] = [4, 6] if out == "ok" else [4, 0]
                o = ["towel", "t-shirt", "sock"].index(label)
                self.held = obj_slices[o]
                self.held_obj = o
                self.x[obj_slices[o][0]:obj_slices[o][1], 2] += 0.2
                self.carry_track = [(0.01, 0.8, 0.8, 30, 4, 6, 0.0)]
                samples = []
                for dz in (0.0, 0.1, 0.2):      # the held object follows the tip and rises, the others stay put
                    smp = np.array([[1.0, 0.3]] * 3)
                    smp[o] = (0.01, 0.3 + dz)
                    samples.append(smp)
                self.obj_track = samples
            self.hold(3.0)
            return gx, gy, fb_t, 0.014

        def drive(self): pass

        def set_phase(self, name, caption):
            super().set_phase(name, caption)
            self.caps.append(self.caption)

    saved = cwt.Test.follow
    cwt.Test.follow = lambda self, seg, row, grip_to=None: self.hold(1.5)
    try:
        s = Sim()
        s.run_multi = types.MethodType(MultiIntegratedTest.run_multi, s)
        # the settle / save_obs prologue of run_multi runs on the fake as well
        res = s.res
        try:
            s.run_multi()
            res["phases"] = s.phases
        except Exception as e:                                           # report instead of crashing the selftest
            import traceback
            traceback.print_exc()
            chk("fake loop ran", False, repr(e))
            return False
    finally:
        cwt.Test.follow = saved
    c = res["cycles"]
    chk("fake loop: cycles", [x["target"]["label"] for x in c[:3]] == ["t-shirt", "towel", "sock"], [x["end"] for x in c])
    return check_fake_result(chk, res, s)


def check_fake_result(chk, res, s):
    c = res["cycles"]
    good = True
    chk("fake: JSON-serialisable result", bool(json.dumps(res, default=str)))
    chk("fake: cycle keys", all(set(CYCLE_KEYS) <= set(x) for x in c), [sorted(set(CYCLE_KEYS) - set(x)) for x in c if set(CYCLE_KEYS) - set(x)])
    chk("fake: cards one per cycle with frame ranges", [k["cycle"] for k in res["cards"]] == [x["cycle"] for x in c]
        and all(k["frame1"] > k["frame0"] and Path(k["obs"]).exists() and Path(k["vlm_json"]).exists() for k in res["cards"])
        and all(a["frame1"] <= b["frame0"] for a, b in zip(res["cards"], res["cards"][1:])), [(k["frame0"], k["frame1"]) for k in res["cards"]])
    chk("fake: N = confirmed at cycle 1", res["n_confirmed_first"] == 3)
    caps = [(p["cycle"], p["target"]) for p in res["phases"] if "cycle" in p]
    chk("fake: phases carry cycle + target", {k for k, _ in caps} >= {1, 2, 3} and (1, "t-shirt") in caps
        and (2, "towel") in caps and (3, "sock") in caps, caps[:3])
    chk("fake: captions [k/N] label: action", "[1/3] t-shirt: carry to the washer" in s.caps and "[2/3] towel: lower the arm" in s.caps
        and "[2/3] towel: missed, retry" in s.caps and "[3/3] sock: carry to the washer" not in s.caps
        and "[1/3] t-shirt: release" in s.caps and "[1/3] t-shirt: retract the arm" in s.caps, sorted(set(s.caps))[:6])
    chk("fake: attempts per cycle", [len(x["attempts"]) for x in c[:3]] == [1, 3, 3], [len(x["attempts"]) for x in c])
    chk("fake: lift_ok verdicts", [x["lift_ok"] for x in c[:3]] == [True, True, False])
    chk("fake: sock given up, then stop", res["stop_reason"] in ("all_given_up", "basket_empty", "unconfirmed_garment_pixels_left"),
        res["stop_reason"])
    chk("fake: n_in_tub_after counts the sim", [x["n_in_tub_after"] for x in c[:3]] == [1, 2, 2], [x["n_in_tub_after"] for x in c])
    chk("fake: objects_final", [o["where"] for o in res["objects_final"]] == ["tub", "tub", "basket"]
        and [o["cycle_loaded"] for o in res["objects_final"]] == [2, 1, None] and res["n_in_tub_final"] == 2 and not res["success"],
        [(o["name"], o["where"], o["cycle_loaded"], o["label_by_vlm"]) for o in res["objects_final"]])
    rc0, rc1 = c[0]["release_check"], c[1]["release_check"]
    chk("fake: one-finger touch -> wiggle; still touching -> open max + second wiggle; steps recorded",
        rc0["touch_after_open"] == [2, 0] and rc0["wiggled"] and [x["step"] for x in rc0["steps"]] == ["wiggle1", "open_max", "wiggle2"]
        and rc0["touch_after_wiggle"] == [0, 0] and s.log.count("release_open_max") == 2 and s.log.count("release_wiggle+") == 4,
        (rc0["steps"], s.log.count("release_wiggle+")))
    chk("fake: touching at the second wiggle too is just recorded", [x["step"] for x in rc1["steps"]] == ["wiggle1", "open_max", "wiggle2"]
        and rc1["touch_after_wiggle"] == [2, 3])
    chk("fake: sock failed attempts bookkeeping", res["objects_tracked"][-1]["given_up"] is True
        or any(o["given_up"] for o in res["objects_tracked"]), [(o["label"], o["failed"], o["given_up"]) for o in res["objects_tracked"]])
    return True


if __name__ == "__main__":
    sys.exit(main() or 0)
