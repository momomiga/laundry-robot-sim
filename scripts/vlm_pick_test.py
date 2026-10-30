"""VLM pick-point test: can a VLM (ollama, Qwen3-VL) point at the next laundry item in the basket?

Substitute laundry (4 coloured boxes, each < 7 cm wide) is dropped into the basket at random
(seeded) positions, the basket camera image is sent to ollama /api/chat, and the returned
pixel point is compared with ground truth obtained by projecting the boxes' world corners
through the camera matrices.

Success = the VLM point lies inside the pixel region (convex hull of the projected corners)
of some object. Also reported: pixel distance to the nearest object centre.
Qwen3-VL answers in 0-1000 normalised coordinates; --coord selects the interpretation
(norm1000 | pixel). Both interpretations are always scored so the choice can be verified.

Output: outputs/vlm_pick/trial_<seed>.png (+ basket image), exit 0 if all trials ran.
"""
import argparse
import base64
import io
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation

import genesis as gs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import robot_sensors  # noqa: E402
import params as P  # noqa: E402

OUT = ROOT / "outputs" / "vlm_pick"
OLLAMA = "http://127.0.0.1:11434/api/chat"
ROBOT_XY, ROBOT_YAW = (1.0, 1.0), 0.0
# (name, colour name, rgb, size xyz [m]); all narrower than the 7 cm gripper opening
ITEMS = [
    ("red", (0.9, 0.15, 0.15), (0.065, 0.045, 0.03)),
    ("green", (0.15, 0.7, 0.2), (0.045, 0.045, 0.03)),
    ("blue", (0.15, 0.3, 0.9), (0.06, 0.035, 0.03)),
    ("yellow", (0.95, 0.85, 0.15), (0.035, 0.035, 0.03)),
]
PROMPT = (
    "This is a top-down camera image of a laundry basket containing {n} small coloured boxes "
    "(substitute laundry). Choose ONE object that a robot should pick up next and point at its centre. "
    'Answer with JSON only, in the form {{"x": <number>, "y": <number>, "label": "<colour of the object>"}}, '
    "where x is the horizontal and y the vertical position of the point in the image."
)


def parse_vlm(text):
    """Extract (x, y, label) from a VLM reply; tolerant of fences, lists, point_2d, bare numbers."""
    t = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"[\{\[].*[\}\]]", t, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, list) and obj:
                obj = obj[0]
            if isinstance(obj, dict):
                label = str(obj.get("label", ""))
                if "x" in obj and "y" in obj:
                    return float(obj["x"]), float(obj["y"]), label
                for k in ("point_2d", "point", "coordinate", "xy"):
                    if k in obj and len(obj[k]) >= 2:
                        return float(obj[k][0]), float(obj[k][1]), label
        except (ValueError, TypeError):
            pass
    nums = re.findall(r"-?\d+(?:\.\d+)?", t)
    mx = re.search(r'"?x"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', t)
    my = re.search(r'"?y"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', t)
    if mx and my:
        return float(mx.group(1)), float(my.group(1)), ""
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1]), ""
    return None


def ask_vlm(img, model, n):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    body = {"model": model, "stream": False, "options": {"temperature": 0},
            "messages": [{"role": "user", "content": PROMPT.format(n=n),
                          "images": [base64.b64encode(buf.getvalue()).decode()]}]}
    req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]["content"]


class World:
    def __init__(self):
        gs.init(backend=gs.gpu, logging_level="warning")
        self.scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=False,
                              vis_options=gs.options.VisOptions(
                                  show_world_frame=False, ambient_light=(0.45, 0.45, 0.45),
                                  lights=[dict(type="directional", dir=(0.2, 0.1, -1.0),
                                               color=(1.0, 1.0, 1.0), intensity=6.0)]))
        sc = self.scene
        sc.add_entity(gs.morphs.Plane())
        self.robot = sc.add_entity(gs.morphs.URDF(file=str(ROOT / "assets/robot/house_robot.urdf"), fixed=True))
        self.boxes = [sc.add_entity(gs.morphs.Box(size=s, pos=(0, 0, 3 + i)),
                                    surface=gs.surfaces.Default(color=(*c, 1)))
                      for i, (_, c, s) in enumerate(ITEMS)]
        self.cam = robot_sensors.add_basket_camera(sc, self.robot)
        sc.build()
        names = ["base_x_joint", "base_y_joint", "base_yaw_joint", "lift_joint"] + [f"joint{i}" for i in range(1, 9)]
        self.dofs = [self.robot.get_joint(n).dofs_idx_local[0] for n in names]
        q = np.zeros(12)
        q[:3] = [*ROBOT_XY, ROBOT_YAW]
        self.robot.set_dofs_position(q, self.dofs)
        kp = np.array([8000, 8000, 3000, 8000] + [300] * 6 + [200, 200], float)
        self.robot.set_dofs_kp(kp, self.dofs)
        self.robot.set_dofs_kv(2 * np.sqrt(kp), self.dofs)
        self.robot.control_dofs_position(q, self.dofs)

    def place(self, seed):
        """Random non-overlapping poses inside the basket; returns nothing, boxes settle in sim."""
        rng = np.random.default_rng(seed)
        ix, iy, _ = P.BASKET_INNER
        kx, ky = P.BASKET_CENTER_XY
        pts = []
        while len(pts) < len(ITEMS):
            p = np.array([rng.uniform(-ix / 2 + 0.05, ix / 2 - 0.05), rng.uniform(-iy / 2 + 0.05, iy / 2 - 0.05)])
            if all(np.linalg.norm(p - q) > 0.095 for q in pts):
                pts.append(p)
        c, s = math.cos(ROBOT_YAW), math.sin(ROBOT_YAW)
        z = P.BASE_Z1 + P.BASKET_WALL + 0.02
        for b, p, (_, _, size) in zip(self.boxes, pts, ITEMS):
            lx, ly = p[0] + kx, p[1] + ky
            b.set_pos((ROBOT_XY[0] + c * lx - s * ly, ROBOT_XY[1] + s * lx + c * ly, z))
            yaw = rng.uniform(0, math.pi) + ROBOT_YAW
            r = Rotation.from_euler("z", yaw).as_quat()
            b.set_quat((r[3], r[0], r[1], r[2]))
            b.zero_all_dofs_velocity() if hasattr(b, "zero_all_dofs_velocity") else None
        for _ in range(150):
            self.scene.step()

    def ground_truth(self):
        """Per item: pixel centre and pixel hull polygon from projected world corners."""
        res = []
        for b, (name, _, size) in zip(self.boxes, ITEMS):
            pos = b.get_pos().cpu().numpy().reshape(3)
            q = b.get_quat().cpu().numpy().reshape(4)
            R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            half = np.array(size) / 2
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
            world = (R @ corners.T).T + pos
            hull = robot_sensors.project_points(self.cam, world)
            ctr = robot_sensors.project_points(self.cam, pos[None] + np.array([[0, 0, size[2] / 2]]))[0]
            res.append(dict(name=name, center=ctr, hull=hull, z=pos[2]))
        return res


def inside(hull, pt):
    try:
        return Delaunay(hull).find_simplex(pt) >= 0
    except Exception:
        return False


def score(gt, pt):
    d = [np.linalg.norm(g["center"] - pt) for g in gt]
    k = int(np.argmin(d))
    return k, float(d[k]), any(inside(g["hull"], pt) for g in gt), inside(gt[k]["hull"], pt)


def annotate(img, gt, pts, path):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    for g in gt:
        h = [tuple(p) for p in g["hull"]]
        c = g["center"]
        xs, ys = [p[0] for p in h], [p[1] for p in h]
        dr.rectangle([min(xs), min(ys), max(xs), max(ys)], outline=(255, 255, 255))
        dr.ellipse([c[0] - 3, c[1] - 3, c[0] + 3, c[1] + 3], outline=(0, 0, 0), width=2)
        dr.text((min(xs), min(ys) - 11), g["name"], fill=(255, 255, 255))
    for pt, col, tag in pts:
        x, y = pt
        dr.line([x - 9, y, x + 9, y], fill=col, width=3)
        dr.line([x, y - 9, x, y + 9], fill=col, width=3)
        dr.text((x + 8, y + 6), tag, fill=col)
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--model", default="qwen3-vl:8b-instruct")
    ap.add_argument("--coord", choices=["norm1000", "pixel"], default="norm1000")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    w = World()
    W, H = P.BASKET_CAM_RES
    stats = {"norm1000": [], "pixel": []}
    for t in range(args.trials):
        seed = args.seed0 + t
        w.place(seed)
        img = robot_sensors.get_basket_image(w.cam)
        gt = w.ground_truth()
        Image.fromarray(img).save(OUT / f"basket_{seed}.png")
        t0 = time.time()
        try:
            reply = ask_vlm(img, args.model, len(ITEMS))
            r = parse_vlm(reply)
        except Exception as e:  # noqa: BLE001
            reply, r = f"ERROR {e}", None
        dt = time.time() - t0
        if r is None:
            print(f"seed {seed}: parse failure, reply={reply!r}")
            for k in stats:
                stats[k].append(None)
            continue
        x, y, label = r
        cand = {"norm1000": np.array([x / 1000 * W, y / 1000 * H]), "pixel": np.array([x, y])}
        line = f"seed {seed}: raw=({x:g},{y:g}) label={label!r} [{dt:.1f}s]"
        marks = []
        for k, pt in cand.items():
            ki, dist, any_in, own_in = score(gt, pt)
            stats[k].append((dist, any_in, own_in, ki, label))
            line += f" | {k}: nearest={gt[ki]['name']} d={dist:.1f}px inside={any_in}"
            marks.append((pt, (255, 0, 255) if k == "norm1000" else (0, 255, 255), k[:4]))
        print(line)
        annotate(img, gt, marks, OUT / f"trial_{seed}.png")
    print("\n=== summary ===")
    for k, v in stats.items():
        ok = [s for s in v if s]
        succ = sum(s[1] for s in ok)
        lab = sum(s[4].lower().strip() == ITEMS[s[3]][0] for s in ok)
        md = np.mean([s[0] for s in ok]) if ok else float("nan")
        print(f"{k}: success(point inside an object) {succ}/{len(v)} = {succ / max(1, len(v)):.0%}, "
              f"mean nearest-centre error {md:.1f}px, label matches nearest object {lab}/{len(v)}")
    print(f"selected coordinate system: {args.coord}")
    sel = stats[args.coord]
    print(f"SUCCESS_RATE {sum(1 for s in sel if s and s[1])}/{len(sel)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
