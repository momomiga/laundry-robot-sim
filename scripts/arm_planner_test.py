"""Plan the basket -> washer pose sequence of cloth_ipc_washer_test.py (seed 0 grasp point) with scripts/arm_planner.py
and replay it kinematically in a plain Genesis scene, measuring arm-obstacle penetration (Genesis narrow phase and an
analytic box SDF of the arm hulls, as scripts/arm_collision_replay.py does).

Pose sequence (robot parked at the origin, washer at arm_planner.WASHER_POS, as in the IPC test):
  standby (collision-free fold; the PiPER zero pose used by the IPC test lies inside the mast) -> pre-open gripper
  -> hover 15 cm above the grasp point -> straight descent into the basket -> close -> straight lift (lift 0.25)
  -> lift joint 0.25 -> 0.5 -> raise -> 6 carry waypoints towards the opening, tilt +12 deg each (max 60, gripper
  yaw kept), heights as high as reachable -> open -> back to standby (lift 0.5; the IPC test's old retract target
  above the mast has no IK solution).
The segments come from arm_planner.TransferSeq in the call order of cloth_ipc_washer_test.py --planner (run_chain),
so this script also checks the planning code of the IPC run: --grasp-set plans that order for 7 grasp points of
earlier IPC runs spread over the basket (CHECK_POINTS, yaw candidates as --yaw-mode ridge) and for the retry chain
(empty pinch -> failed lift -> success), replays each commanded trajectory and passes if every segment planned with
0 mm penetration (exit code 1 otherwise).
Fingers inside the basket: the gripper links (link6/7/8) enter the basket's open volume to reach the heap (finger
bottom ~6 mm above the floor at the grasp); that is not penetration. Penetration of the basket *boxes* (floor, walls)
is measured for every link including the fingers; for the gripper links the planner requires only 2 mm clearance to
the basket boxes (10 mm for every other link-obstacle pair). The cloth itself is not in the planning scene.

CPU backend for planning; the video renders with the rasterizer, so take the GPU lock:
  flock /tmp/house_robot-gpu.lock .venv/bin/python scripts/arm_planner_test.py            # plan + replay + video
  .venv/bin/python scripts/arm_planner_test.py --trials 10 --no-video                    # success rate / timing
  .venv/bin/python scripts/arm_planner_test.py --grasp-set          # IPC --planner check -> *_grasp_set.json
  (--grasp-set --backend gpu under the GPU lock: same check on the backend the planner gets inside the IPC process)
  .venv-ipc/bin/python scripts/arm_planner_test.py --grasp-set --proxy
  (same through arm_planner.PlannerClient, the CPU planner child process of cloth_ipc_washer_test --planner;
  replay check = clearance_stats of the commanded q)
Outputs: outputs/arm_planner_test.mp4 (640x480), outputs/arm_planner_test.json, outputs/arm_planner_test_traj.npz/.json
(the latter in the input format of arm_collision_replay.py: `python scripts/arm_collision_replay.py
outputs/arm_planner_test_traj --every 1`).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))
import params as P  # noqa: E402
from arm_planner import (I_ARM, I_LIFT, WASHER_POS, ArmPlanner, PlannerClient, TransferSeq,  # noqa: E402
                         drop_keepout, retime, seg_duration)

DT = 0.01
W, H, FPS = 640, 480, 30
BASKET_FLOOR = P.BASE_Z1 + P.BASKET_WALL
# seed 0 of the IPC through run (outputs/ipc_diag/full_base.json): grasp vertex, gripper yaw, finger-bottom target
GRASP = (0.0, -0.071, 0.307)
GRASP_YAW = -15.0
DEPTH = 0.03                     # --depth of the best IPC args
LIFT_GRASP, LIFT_CARRY = 0.25, 0.5
PRE_OPEN, HOLD, OPEN = 0.015, 0.004, 0.035
WAIT = dict(wait_lift=LIFT_CARRY, keepout=drop_keepout())   # wait above the towel drop envelope (--drop-h 0.12)


# grasp points of earlier IPC runs (outputs/cloth_ipc_full, outputs/ipc_diag; grasp vertex, ridge-preferred yaw)
# spread over the area select_grasp allows (5 cm inside the basket walls: x -0.085..0.165, y -0.15..0.05), heap tops
# 2.4-15.5 cm above the basket floor; index 0 = the seed-0 point above. In the 5 cm band along a wall only the yaws
# with the gripper body along the wall are collision-free (about half), so the preferred yaws below include ones
# that need the yaw fallback.
CHECK_POINTS = [
    (GRASP, GRASP_YAW),
    ((0.037, -0.148, 0.318), -165.0),    # left edge band (y wall)
    ((-0.026, 0.048, 0.298), -30.0),     # back-right edge band, low heap; straight carry line passes the joint1 axis
    ((0.144, 0.008, 0.294), -45.0),      # front, right of centre
    ((0.064, 0.047, 0.415), -15.0),      # right edge band, highest heap
    ((0.090, -0.120, 0.284), -165.0),    # front-left, lowest heap; straight carry line: upper arm hits the open lid
    ((-0.072, -0.102, 0.359), -105.0),   # back, near the x wall
    ((0.136, -0.001, 0.374), -75.0),     # front, high heap
]
# grasp points in the basket corners (5 cm from two walls; earlier IPC runs picked such points): the gripper body
# (link6) cuts 4-5 mm into a wall at the descent end for every yaw, so the approach must fail without any motion
# (the IPC run then retries elsewhere)
CORNER_POINTS = [
    ((-0.081, -0.150, 0.399), -150.0),
    ((0.160, -0.144, 0.299), -135.0),
    ((0.164, 0.044, 0.312), -150.0),
]
# retry chain of the IPC attempts loop: empty pinch at point 0 -> corner point (approach does not plan, no motion)
# -> point 1, lift fails -> point 4 -> carry
CHAIN = [(0, "miss"), ("c0", "plan_fail"), (1, "lift_fail"), (4, "ok")]


def yaw_order(pref, n=24):
    """Gripper yaw candidates like cloth_ipc_washer_test --planner --yaw-mode ridge: the 15 deg grid ordered by the
    distance (mod 180: both finger orders close across the same ridge) to the preferred yaw, first n (all 24)."""
    c = np.arange(-180, 180, 15.0)
    d = np.abs((c - pref + 90) % 180 - 90)
    return [float(v) for v in c[np.argsort(d, kind="stable")][:n]]


def fb_target(gz):
    """Finger-bottom target of cloth_ipc_washer_test.select_grasp (--depth 0.03 below the heap top)."""
    return max(BASKET_FLOOR + 0.003, gz - DEPTH)


def run_chain(T, chain, release_xy=WASHER_POS[:2]):
    """Plan the IPC run's motion order (see arm_planner.TransferSeq) for a list of (grasp, yaws, outcome); outcome
    'miss' (empty pinch: open, back up, next point), 'lift_fail' (lifted, cloth lost: open, lift joint down, next
    point), 'plan_fail' (the approach must not plan; its failed segment is moved to T.plan_fails, no motion, next
    point) or 'ok' (lift, carry, release, back to standby). Durations as in cloth_ipc_washer_test (best args)."""
    ok = True
    T.plan_fails = []
    ok = T.lower_to_grasp("lower to grasp lift", 1.5)   # waited at LIFT_CARRY above the towel drop envelope
    for k, (grasp, yaws, outcome) in enumerate(chain):
        ok = ok and T.grip("pre_open", PRE_OPEN, 0.3)
        if not ok:
            break
        ok = T.approach("standby->hover" if k == 0 else "hover", grasp, fb_target(grasp[2]), yaws, 2.0)
        if outcome == "plan_fail":
            if ok:   # expected to be infeasible: a plan here is a check failure
                T.segs[-1]["name"] += " (expected no plan)"
                T.segs[-1]["res"].ok = False
                return False
            T.plan_fails.append(T.segs.pop())
            ok = True
            continue
        ok = ok and T.descend(1.5)
        ok = ok and T.close(1.0)
        if outcome == "miss":
            ok = ok and T.grip("miss_open", PRE_OPEN, 0.4)
            ok = ok and T.up_from_grasp("miss_up", 1.0)
            continue
        ok = ok and T.lift1(2.0)
        ok = ok and T.lift_to("lift2 (lift joint)", LIFT_CARRY, 2.0)
        if outcome == "lift_fail":
            ok = ok and T.grip("retry_open", OPEN, 0.5)
            ok = ok and T.lift_to("retry_lower", LIFT_GRASP, 1.5)
            continue
        ok = ok and T.raise_(2.0)
        ok = ok and T.carry(*release_xy, 9.0 / 6)
        ok = ok and T.release(0.6)
        ok = ok and T.to_standby(1.5)
    return ok


class Sequence(TransferSeq):
    """Single grasp at the seed-0 point (yaw fixed), shared standby optional."""

    def __init__(self, P_, chain=None, standby=None):
        super().__init__(P_, lift_grasp=LIFT_GRASP, lift_carry=LIFT_CARRY, hold=HOLD, open_=OPEN, **WAIT)
        self.chain = chain or [(GRASP, [GRASP_YAW], "ok")]
        self.given_standby = standby

    def run(self):
        if self.given_standby is not None:
            self.standby, self.q = self.given_standby.copy(), self.given_standby.copy()
        else:
            assert self.start() is not None, "no collision-free standby pose"
        return run_chain(self, self.chain)


def trajectory(segs, vmax=0.8):
    """Concatenate the segment paths into a dt-sampled trajectory; duration = max(nominal, arc / vmax) (as
    cloth_ipc_washer_test --planner executes them)."""
    Q, seg_rows = [], []
    for s in segs:
        p = s["res"].path
        dur = seg_duration(p, s["dur"], vmax)
        seg_rows.append([len(Q), s["name"], 0, s["res"].method])
        Q += list(retime(p, dur, DT)) if len(p) > 1 else [p[-1]] * int(dur / DT)
    return np.array(Q), seg_rows


def measure(P_, Q, seg_rows, every=1):
    """Per step (every `every`-th): Genesis contacts (all pairs, neutral pairs kept) and box-SDF depth per (arm link,
    obstacle)."""
    starts = [s[0] for s in seg_rows]
    per_seg = [dict(gen_pen_mm=0.0, sdf_depth_mm=0.0, min_clear_mm=np.inf, worst="", fb_min_mm=np.inf)
               for _ in seg_rows]
    for S in per_seg:
        S.update(room_clear_mm=np.inf, room_pair="")   # closest approach to a room box within 5 cm (inf: > 50 mm)
    pairs = {}
    for k in range(0, len(Q), every):
        q = Q[k]
        i = max(j for j, s in enumerate(starts) if s <= k)
        con = P_.contacts(q)
        d = P_.sdf(None, reach=0.05)
        fb = min(float(P_.robot.get_link(n).get_verts().cpu().numpy()[:, 2].min()) for n in ("link7", "link8"))
        S = per_seg[i]
        S["fb_min_mm"] = min(S["fb_min_mm"], (fb - BASKET_FLOOR) * 1e3)
        for (a, b), p in con.items():
            if (a in ("link1",) and b == "carriage") or p <= 0:
                continue
            pairs.setdefault((a, b), [0, 0.0, 0.0])
            pairs[(a, b)][0] += 1
            pairs[(a, b)][1] = max(pairs[(a, b)][1], p * 1e3)
            if p * 1e3 > S["gen_pen_mm"]:
                S["gen_pen_mm"], S["worst"] = p * 1e3, f"{a}-{b}"
        for (ln, on), s in d.items():
            S["min_clear_mm"] = min(S["min_clear_mm"], s * 1e3)
            if on.startswith("room") and s * 1e3 < S["room_clear_mm"]:
                S["room_clear_mm"], S["room_pair"] = s * 1e3, f"{ln}-{on}"
            if s < 0:
                pairs.setdefault((ln, on), [0, 0.0, 0.0])
                pairs[(ln, on)][2] = max(pairs[(ln, on)][2], -s * 1e3)
                if -s * 1e3 > S["sdf_depth_mm"]:
                    S["sdf_depth_mm"] = -s * 1e3
                    S["worst"] = S["worst"] or f"{ln}-{on}"
    return per_seg, {f"{a}|{b}": dict(steps=v[0], gen_max_mm=round(v[1], 2), sdf_max_mm=round(v[2], 2))
                     for (a, b), v in pairs.items()}


def render_video(P_, cams, Q, seg_rows, path):
    """640x480 wide view (same camera as cloth_ipc_washer_test) with a rear inset (mast / lid side) and the segment
    name as caption."""
    from PIL import Image, ImageDraw
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                             "-crf", "22", str(path)], stdin=subprocess.PIPE)
    starts = [r[0] for r in seg_rows]
    nxt = 0.0
    for k, q in enumerate(Q):
        if k * DT + 1e-9 < nxt:
            continue
        nxt += 1.0 / FPS
        P_.set_q(q)
        img = Image.fromarray(np.asarray(cams["wide"].render(force_render=True)[0]).astype(np.uint8))
        if "rear" in cams:
            ins = Image.fromarray(np.asarray(cams["rear"].render(force_render=True)[0]).astype(np.uint8)).resize((240, 180))
            img.paste(ins, (W - 244, 4))
        i = max(j for j, s0 in enumerate(starts) if s0 <= k)
        d = ImageDraw.Draw(img)
        d.rectangle((0, H - 22, W, H), fill=(0, 0, 0))
        d.text((6, H - 18), f"t={k * DT:5.2f}s  {seg_rows[i][1]}  [{seg_rows[i][3]}]", fill=(255, 255, 255))
        proc.stdin.write(np.asarray(img).tobytes())
    proc.stdin.close()
    proc.wait()


def seg_json(segs):
    return [dict(name=s["name"], ok=s["res"].ok, method=s["res"].method, plan_s=round(s["res"].plan_s, 3),
                 rrt_calls=s["res"].rrt_calls, min_clear_plan_mm=round(float(s["res"].min_clearance) * 1e3, 1),
                 reason=s["res"].reason[:200],
                 info={k: v for k, v in s["res"].info.items() if k in ("ik", "goal_rank", "rejected", "knots", "z",
                                                                       "pose_candidates_tried", "direct", "last_rrt_s",
                                                                       "shortcut_s", "yaw", "yaws_tried", "bottom_up",
                                                                       "route")})
            for s in segs]


def grasp_set_check_proxy(a):
    """grasp_set_check through PlannerClient (the planner child process the IPC test uses): same runs and TransferSeq
    calls, results (ok, segment counts, planning time) compared with the in-process check; the commanded trajectory
    is checked with one clearance_stats request (max of Genesis contact and box-SDF depth, 0 mm required)."""
    t0 = time.time()
    C = PlannerClient(dict(margin=a.margin, seed=0, with_room=a.room),
                      dict(lift_grasp=LIFT_GRASP, lift_carry=LIFT_CARRY, hold=HOLD, open_=OPEN, **WAIT))
    C.seed(0)
    standby_s = time.time()
    standby = C.start()
    assert standby is not None, "no collision-free standby pose"
    build_s, standby_s = C.build_s, time.time() - standby_s
    print(f"planner process ready ({time.time() - t0:.1f}s, scene build {build_s:.1f}s)", flush=True)
    runs = [(f"p{i} {pt} yaw{yaw:.0f}", [(pt, yaw_order(yaw), "ok")]) for i, (pt, yaw) in enumerate(CHECK_POINTS)]
    runs += [(f"c{i} {pt} yaw{yaw:.0f} (corner)", [(pt, yaw_order(yaw), "plan_fail")])
             for i, (pt, yaw) in enumerate(CORNER_POINTS)]
    pts = {**{i: p for i, p in enumerate(CHECK_POINTS)}, **{f"c{i}": p for i, p in enumerate(CORNER_POINTS)}}
    runs.append(("chain " + " -> ".join(f"{'' if isinstance(i, str) else 'p'}{i}:{o}" for i, o in CHAIN),
                 [(pts[i][0], yaw_order(pts[i][1]), o) for i, o in CHAIN]))
    out = []
    for name, chain in runs:
        C.reset(standby)
        t1 = time.time()
        ok = run_chain(C, chain)
        plan_s = time.time() - t1
        row = dict(run=name, ok=ok, plan_total_s=round(plan_s, 1), segs=seg_json(C.segs), skipped=C.skipped,
                   carry_wps=C.carry_wps, expected_plan_fails=seg_json(C.plan_fails))
        if ok and len(C.segs) > 1:
            Q, _ = trajectory(C.segs)
            st = C.AP.clearance_stats_many(Q[::a.every])
            row.update(replay_steps=len(Q), max_pen_mm=round(max(s[0] for s in st) * 1e3, 2),
                       min_clear_replay_mm=round(min(s[1] for s in st) * 1e3, 1))
        row["min_clear_plan_mm"] = min((r["min_clear_plan_mm"] for r in row["segs"] if r["ok"]), default=None)
        print(f"{name}: ok={ok} plan {plan_s:.1f}s segs={len(row['segs'])} min clear plan/replay "
              f"{row['min_clear_plan_mm']}/{row.get('min_clear_replay_mm')} mm, max pen {row.get('max_pen_mm')} mm",
              flush=True)
        out.append(row)
    C.shutdown()
    passed = all(r["ok"] and r.get("max_pen_mm", 0) == 0 for r in out)
    res = dict(passed=passed, margin=a.margin, backend="cpu (PlannerClient)", standby=np.round(standby, 4).tolist(),
               standby_s=round(standby_s, 2), runs=out)
    Path(a.out + "_grasp_set_proxy.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"grasp set (proxy): passed={passed} ({sum(r['ok'] for r in out)}/{len(out)} runs planned)  json: "
          f"{a.out}_grasp_set_proxy.json")
    return passed


def grasp_set_check(P_, a):
    """CPU acceptance check of cloth_ipc_washer_test --planner: the IPC motion order (run_chain, same TransferSeq
    calls and parameters) for every CHECK_POINTS entry (fresh from standby, yaw candidates as --yaw-mode ridge) and
    for the retry chain CHAIN (empty pinch, failed lift, success). Each plan is replayed at DT along the commanded
    (retimed) trajectory: Genesis contacts + box-SDF depth. Pass = every segment planned and 0 mm penetration.
    Exit code 1 if not."""
    P_.rng = np.random.default_rng(0)
    T0 = TransferSeq(P_, lift_grasp=LIFT_GRASP, **WAIT)
    t0 = time.time()
    standby = T0.start()
    assert standby is not None, "no collision-free standby pose"
    standby_s = time.time() - t0
    runs = [(f"p{i} {pt} yaw{yaw:.0f}", [(pt, yaw_order(yaw), "ok")]) for i, (pt, yaw) in enumerate(CHECK_POINTS)]
    runs += [(f"c{i} {pt} yaw{yaw:.0f} (corner)", [(pt, yaw_order(yaw), "plan_fail")])
             for i, (pt, yaw) in enumerate(CORNER_POINTS)]
    pts = {**{i: p for i, p in enumerate(CHECK_POINTS)}, **{f"c{i}": p for i, p in enumerate(CORNER_POINTS)}}
    runs.append(("chain " + " -> ".join(f"{'' if isinstance(i, str) else 'p'}{i}:{o}" for i, o in CHAIN),
                 [(pts[i][0], yaw_order(pts[i][1]), o) for i, o in CHAIN]))
    out = []
    for name, chain in runs:
        seq = Sequence(P_, chain, standby)
        t1 = time.time()
        ok = seq.run()
        plan_s = time.time() - t1
        row = dict(run=name, ok=ok, plan_total_s=round(plan_s, 1), segs=seg_json(seq.segs), skipped=seq.skipped,
                   carry_wps=seq.carry_wps, expected_plan_fails=seg_json(seq.plan_fails))
        if ok and len(seq.segs) > 1:
            Q, seg_rows = trajectory(seq.segs)
            per_seg, pairs = measure(P_, Q, seg_rows, every=a.every)
            row.update(replay_steps=len(Q), max_gen_pen_mm=round(max(m["gen_pen_mm"] for m in per_seg), 2),
                       max_sdf_depth_mm=round(max(m["sdf_depth_mm"] for m in per_seg), 2),
                       min_clear_replay_mm=round(min(m["min_clear_mm"] for m in per_seg), 1),
                       min_room_clear_mm=round(min(m["room_clear_mm"] for m in per_seg), 1),
                       pairs=pairs, replay=[dict(name=r[1], method=r[3], **{k: (round(v, 2) if isinstance(v, float)
                                                                                 else v) for k, v in m.items()})
                                            for r, m in zip(seg_rows, per_seg)])
        row["min_clear_plan_mm"] = min((r["min_clear_plan_mm"] for r in row["segs"] if r["ok"]), default=None)
        bad = [r["name"] for r in row["segs"] if not r["ok"]]
        if a.room and "replay" in row:
            near = [(r["room_clear_mm"], r["name"], r["room_pair"]) for r in row["replay"] if r["room_pair"]]
            print("  room: closest " + ("; ".join(f"{c:.1f} mm {n} {p}" for c, n, p in sorted(near)[:3])
                                        if near else "none within 50 mm"), flush=True)
        print(f"{name}: ok={ok} plan {plan_s:.1f}s segs={len(row['segs'])} min clear plan/replay "
              f"{row['min_clear_plan_mm']}/{row.get('min_clear_replay_mm')} mm, max pen Genesis/SDF "
              f"{row.get('max_gen_pen_mm')}/{row.get('max_sdf_depth_mm')} mm" + (f" FAILED {bad}: "
              f"{row['segs'][-1]['reason'][:200]}" if bad else ""), flush=True)
        out.append(row)
    passed = all(r["ok"] and r.get("max_gen_pen_mm", 0) == 0 and r.get("max_sdf_depth_mm", 0) == 0 for r in out)
    res = dict(passed=passed, margin=a.margin, backend=a.backend, standby=np.round(standby, 4).tolist(),
               standby_s=round(standby_s, 2), points=CHECK_POINTS, chain=CHAIN, runs=out)
    Path(a.out + "_grasp_set.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"grasp set: passed={passed} ({sum(r['ok'] for r in out)}/{len(out)} runs planned)  json: "
          f"{a.out}_grasp_set.json")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1, help="plan the whole sequence this many times (rng seeds 0..)")
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--rrt-only", action="store_true", help="skip the direct-interpolation shortcut for free-space "
                    "segments (measures the RRTConnect success rate)")
    ap.add_argument("--grasp-set", action="store_true", help="acceptance check of cloth_ipc_washer_test --planner: "
                    "plan + replay the IPC motion order for CHECK_POINTS and the retry chain (no video)")
    ap.add_argument("--proxy", action="store_true", help="--grasp-set through arm_planner.PlannerClient (planner in a "
                    "CPU child process, as in cloth_ipc_washer_test --planner)")
    ap.add_argument("--room", action="store_true", help="plan with the house (room.urdf collision boxes in the parked "
                    "frame, arm links only) in the scene; --grasp-set also reports the clearance to each room part")
    ap.add_argument("--every", type=int, default=2, help="--grasp-set: replay check every n-th step")
    ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu", help="gpu: the backend the planner gets "
                    "inside the IPC process (gs.gpu); needs the GPU lock")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "arm_planner_test"))
    a = ap.parse_args()
    if a.grasp_set:
        a.no_video = True

    cams = {}

    def hook(sc):
        if not a.no_video:
            cams["wide"] = sc.add_camera(res=(W, H), pos=(1.15, -0.45, 1.75), lookat=(0.12, 0.30, 0.60), fov=55,
                                         GUI=False)
            cams["rear"] = sc.add_camera(res=(W, H), pos=(-1.05, -0.35, 1.45), lookat=(-0.05, 0.30, 0.75), fov=50,
                                         GUI=False)

    if a.proxy:
        if not a.grasp_set:
            ap.error("--proxy needs --grasp-set")
        sys.exit(0 if grasp_set_check_proxy(a) else 1)
    import genesis as gs
    vis = gs.options.VisOptions(show_world_frame=False, ambient_light=(0.4, 0.4, 0.4),
                                lights=[dict(type="directional", dir=(-0.4, 0.3, -1.0), color=(1, 1, 1),
                                             intensity=5.0)]) if not a.no_video else None
    t0 = time.time()
    P_ = ArmPlanner(margin=a.margin, scene_hook=hook, vis_options=vis, backend=a.backend, with_room=a.room)
    build_s = time.time() - t0
    if a.grasp_set:
        print(f"planner scene build {build_s:.1f}s ({a.backend})", flush=True)
        sys.exit(0 if grasp_set_check(P_, a) else 1)
    if a.rrt_only:
        orig = P_.plan
        P_.plan = lambda qs, qg, **kw: orig(qs, qg, **{**kw, "allow_direct": kw.get("allow_direct", False)
                                                      if abs(qg[I_LIFT] - qs[I_LIFT]) < 1e-6 and
                                                      np.abs(qg[I_ARM] - qs[I_ARM]).max() > 1e-6 else True})
    trials = []
    first = None
    for tr in range(a.trials):
        P_.rng = np.random.default_rng(tr)
        seq = Sequence(P_)
        t1 = time.time()
        ok = seq.run()
        rows = seg_json(seq.segs)
        trials.append(dict(trial=tr, ok=ok, total_s=round(time.time() - t1, 2), segs=rows))
        print(f"trial {tr}: ok={ok} {time.time() - t1:.1f}s " +
              " | ".join(f"{r['name']}:{r['method']}:{r['plan_s']:.2f}s" for r in rows), flush=True)
        if first is None:
            first = seq
    res = dict(build_s=round(build_s, 1), margin=a.margin, grasp=GRASP, grasp_yaw=GRASP_YAW,
               standby=np.round(first.standby, 4).tolist(), standby_s=round(first.standby_s, 2),
               carry_wps=first.carry_wps, skipped=first.skipped, trials=trials)
    # per-segment summary over trials
    names = [r["name"].split(" ")[0] for r in trials[0]["segs"]]
    summ = {}
    for t in trials:
        for r in t["segs"]:
            n = r["name"].split(" ")[0]
            s = summ.setdefault(n, dict(n=0, ok=0, times=[], methods={}))
            s["n"] += 1
            s["ok"] += int(r["ok"])
            s["times"].append(r["plan_s"])
            s["methods"][r["method"]] = s["methods"].get(r["method"], 0) + 1
    res["summary"] = {n: dict(ok=f"{s['ok']}/{s['n']}", plan_s_median=round(float(np.median(s["times"])), 3),
                              plan_s_max=round(float(np.max(s["times"])), 3), methods=s["methods"])
                      for n, s in summ.items()}
    print("segment | success | plan s median / max | methods")
    for n in names:
        s = res["summary"][n]
        print(f"{n} | {s['ok']} | {s['plan_s_median']} / {s['plan_s_max']} | {s['methods']}")

    if all(s["res"].ok for s in first.segs):
        Q, seg_rows = trajectory(first.segs)
        t2 = time.time()
        per_seg, pairs = measure(P_, Q, seg_rows)
        res["replay"] = dict(steps=len(Q), duration_s=round(len(Q) * DT, 2), measure_s=round(time.time() - t2, 1),
                             pairs=pairs, segments=[dict(name=r[1], t0=round(r[0] * DT, 2), method=r[3],
                                                         **{k: (round(v, 2) if isinstance(v, float) else v)
                                                            for k, v in m.items()})
                                                    for r, m in zip(seg_rows, per_seg)])
        print("replay: segment | t0 | method | max Genesis pen mm | max SDF depth mm | min clearance mm | "
              "finger bottom - basket floor mm")
        for s in res["replay"]["segments"]:
            print(f"{s['name']} | {s['t0']} | {s['method']} | {s['gen_pen_mm']} | {s['sdf_depth_mm']} | "
                  f"{s['min_clear_mm']} | {s['fb_min_mm']}" + (f" | {s['worst']}" if s["worst"] else ""))
        print("pairs with penetration > 0:", pairs)
        np.savez(a.out + "_traj.npz", q=Q)
        Path(a.out + "_traj.json").write_text(json.dumps(
            dict(seg=seg_rows, diag=dict(dt=DT)), indent=1))
        if not a.no_video:
            render_video(P_, cams, Q, seg_rows, a.out + ".mp4")
            print("video:", a.out + ".mp4")
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, default=str))
    print("json:", a.out + ".json")


if __name__ == "__main__":
    main()
