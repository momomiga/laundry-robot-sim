"""VLM-chosen grasp centre for the towel heap in the basket (CPU-testable; the VLM is injectable).

Pipeline: basket-camera RGB -> VLM (qwen3-vl via ollama) ranks 3 pinch points (0-1000 normalised image
coords) -> pixels -> 3D surface point (point cloud / depth / projected-vertex z-buffer) -> validation
against the basket geometry and the cloth -> first valid candidate, else fallback (centre_xy None; the
caller then uses its heuristic).

Observation npz (written by cloth_ipc_washer_test.py --save-obs; keys, all in the parked-base frame):
  verts   (N,3)  cloth vertices                        [required]
  rgb     (H,W,3) uint8 basket-camera image            [required for the VLM / CLI]
  K       (3,3)  pinhole intrinsics (pixels)           [required unless `points` present]
  E       (4,4)  extrinsics, OpenCV camera (x right, y down, z forward); direction set by
                 `E_mode` ("world2cam" default | "cam2world"), key `E_mode` in the npz (str) or argument
  depth   (H,W)  optional z-depth along the optical axis [m] (nan/0/inf = no surface)
  points  (H,W,3) optional per-pixel 3D point (nan = none)
Missing depth/points -> the vertices are projected through K/E and z-buffered.
"""
import argparse
import base64
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import params as P  # noqa: E402

OLLAMA = "http://127.0.0.1:11434/api/chat"
WALL_BAND = 0.05        # unusable band along the inner basket walls [m]
TOWEL_R = 3             # towel check: half-size of the pixel neighbourhood [px] (7 x 7)
TOWEL_FRAC = 0.6        # towel check: min fraction of towel-coloured pixels in it
VERT_TOL = 0.015        # a cloth vertex must lie within this of the point [m]
FLOOR_Z = P.BASE_Z1 + P.BASKET_WALL

PROMPT = (
    "This is a top-down camera image of a laundry basket with a crumpled towel heap. A robot will pinch "
    "the towel with a two-finger parallel gripper (max opening 7 cm, approaching vertically from above). "
    "Choose the 3 best pinch points, best first: on a raised fold or ridge of cloth near the middle of "
    "the basket, on cloth (never the bare basket floor), at least 5 cm away from the basket walls. "
    'Answer with JSON only, a list of 3 objects: [{{"point_2d": [x, y], "label": "pinch 1"}}, '
    '{{"point_2d": [x, y], "label": "pinch 2"}}, {{"point_2d": [x, y], "label": "pinch 3"}}] '
    "where x is horizontal and y vertical in 0-1000 normalised image coordinates (0,0 = top-left)."
)


def build_prompt():
    return PROMPT.format()


def ask(rgb, model="qwen3-vl:8b-instruct", host=OLLAMA, timeout=60, prompt=None):
    """Send the image to ollama /api/chat (temperature 0); returns the raw reply text. Raises on timeout."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, np.uint8)).save(buf, "PNG")
    body = {"model": model, "stream": False, "options": {"temperature": 0},
            "messages": [{"role": "user", "content": prompt or build_prompt(),
                          "images": [base64.b64encode(buf.getvalue()).decode()]}]}
    req = urllib.request.Request(host, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def parse_ranked(text, size=None):
    """Ranked list of (u, v) from a reply; tolerant of fences, extra text, point_2d/lists.
    Coordinates are 0-1000 normalised; `size`=(W,H) converts to pixels, else the normalised values are returned."""
    t = re.sub(r"```(?:json)?", "", str(text))
    pts = []
    m = re.search(r"[\{\[].*[\}\]]", t, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            obj = None
        items = obj
        if isinstance(obj, dict):
            items = next((obj[k] for k in ("points", "candidates", "pinch_points") if k in obj), [obj])
        for it in items if isinstance(items, list) else []:
            xy = None
            if isinstance(it, dict):
                if "x" in it and "y" in it:
                    xy = (it["x"], it["y"])
                else:
                    for k in ("point_2d", "point", "coordinate", "xy"):
                        if k in it and len(it[k]) >= 2:
                            xy = (it[k][0], it[k][1])
                            break
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                xy = (it[0], it[1])
            try:
                if xy is not None:
                    pts.append((float(xy[0]), float(xy[1])))
            except (TypeError, ValueError):
                pass
    if not pts:   # malformed JSON (seen: {"x": 521, 625, "y": 625}): take the x / y numbers of each {...} block
        for blk in re.findall(r"\{[^{}]*\}", t):
            mx = re.search(r'"x"\s*:\s*(-?\d+(?:\.\d+)?)', blk)
            my = re.search(r'"y"\s*:\s*(-?\d+(?:\.\d+)?)', blk)
            mp = re.search(r'"point_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)', blk)
            if mx and my:
                pts.append((float(mx.group(1)), float(my.group(1))))
            elif mp:
                pts.append((float(mp.group(1)), float(mp.group(2))))
    if size is not None:
        pts = [(x / 1000.0 * size[0], y / 1000.0 * size[1]) for x, y in pts]
    return pts


def load_obs(path, e_mode=None):
    z = np.load(path, allow_pickle=False)
    obs = {k: z[k] for k in z.files}
    # key names of cloth_ipc_washer_test.py --save-obs (Genesis cam.extrinsics is world -> camera, OpenCV axes as
    # used by robot_sensors.project_points)
    for new, old in (("verts", "cloth_verts"), ("K", "intrinsics"), ("E", "extrinsics")):
        if new not in obs and old in obs:
            obs[new] = obs[old]
    if "points" not in obs and "pc_world" in obs:
        pts = np.asarray(obs["pc_world"], float).copy()
        if "pc_mask" in obs:
            pts[~np.asarray(obs["pc_mask"], bool)] = np.nan
        obs["points"] = pts
    mode = e_mode or (str(obs["E_mode"]) if "E_mode" in obs else "world2cam")
    if mode not in ("world2cam", "cam2world"):
        raise ValueError(f"E_mode {mode}")
    obs["E_mode"] = mode
    return obs


def _w2c(obs):
    E = np.asarray(obs["E"], float)
    return np.linalg.inv(E) if obs.get("E_mode", "world2cam") == "cam2world" else E


def project(obs, pts):
    """World/base points (N,3) -> (uv (N,2), z-depth (N,))."""
    E = _w2c(obs)
    pc = pts @ E[:3, :3].T + E[:3, 3]
    uv = (np.asarray(obs["K"], float) @ pc.T).T
    return uv[:, :2] / uv[:, 2:3], pc[:, 2]


def pixel_to_point(u, v, obs, radius_px=6.0, front_tol=0.01):
    """3D point (parked-base frame) under pixel (u, v), or None if no surface there."""
    rgb = obs.get("rgb")
    ui, vi = int(round(u)), int(round(v))
    for key in ("points", "depth"):
        a = obs.get(key)
        if a is None or a.ndim < 2 or not (0 <= vi < a.shape[0] and 0 <= ui < a.shape[1]):
            continue
        if key == "points":
            p = np.asarray(a[vi, ui], float)
            if np.all(np.isfinite(p)):
                return p
            return None
        d = float(a[vi, ui])
        if np.isfinite(d) and d > 0:
            K = np.asarray(obs["K"], float)
            pc = np.array([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d])
            E = _w2c(obs)
            return E[:3, :3].T @ (pc - E[:3, 3])
        return None
    del rgb
    verts = np.asarray(obs["verts"], float)
    uv, z = project(obs, verts)
    ok = z > 0
    d = np.hypot(uv[:, 0] - u, uv[:, 1] - v)
    cand = np.where(ok & (d <= radius_px))[0]
    if not len(cand):
        return None
    cand = cand[z[cand] <= z[cand].min() + front_tol]      # visible (front-most) layer only
    i = cand[np.argmin(d[cand])]
    K = np.asarray(obs["K"], float)
    zc = z[i]
    pc = np.array([(u - K[0, 2]) / K[0, 0] * zc, (v - K[1, 2]) / K[1, 1] * zc, zc])
    E = _w2c(obs)
    return E[:3, :3].T @ (pc - E[:3, 3])


def towel_fraction(rgb, pixel, r=TOWEL_R):
    """Fraction of towel-coloured pixels in the (2r+1)^2 neighbourhood of `pixel` = (u, v).
    The towel renders orange (R > G > B, blue about 0.2-0.3 of R, saturation ~0.8 whatever the shading) and the basket
    blue (B > R), so "red clearly above blue and green in between" separates them; measured on the saved takes the
    towel pixels all have B/R < 0.25 and the basket's B/R > 1, so the thresholds are far from both."""
    im = np.asarray(rgb)
    u, v = int(round(pixel[0])), int(round(pixel[1]))
    h, w = im.shape[:2]
    patch = im[max(v - r, 0):min(v + r + 1, h), max(u - r, 0):min(u + r + 1, w), :3].reshape(-1, 3).astype(float)
    if not len(patch):
        return 0.0
    R, G, B = patch[:, 0], patch[:, 1], patch[:, 2]
    return float(np.mean((R > 60) & (B < 0.5 * R) & (G > B) & (G < 0.9 * R)))


def validate(point, verts, pixel, rgb, band=WALL_BAND, tol=VERT_TOL):
    """(ok, reason). Basket frame = parked-base frame; inner area minus `band`, on towel in the image (the grasp
    height is chosen downstream by select_grasp), on cloth."""
    if point is None:
        return False, "no_surface"
    cx, cy = P.BASKET_CENTER_XY
    ix, iy, _ = P.BASKET_INNER
    dx, dy = point[0] - cx, point[1] - cy
    if abs(dx) > ix / 2 or abs(dy) > iy / 2:
        return False, "outside_basket"
    if abs(dx) > ix / 2 - band or abs(dy) > iy / 2 - band:
        return False, "wall_band"
    if towel_fraction(rgb, pixel) < TOWEL_FRAC:
        return False, "not_towel"
    v = np.asarray(verts, float)
    if not len(v) or np.min(np.linalg.norm(v - point, axis=1)) > tol:
        return False, "no_cloth"
    return True, ""


def choose_center(obs, vlm=None, model="qwen3-vl:8b-instruct", host=OLLAMA, timeout=60):
    """Ask the VLM and pick the first valid candidate.
    vlm: callable(rgb)->text (mock/injection) or None (-> ask()). Never raises on VLM trouble.
    Returns dict(center_xy, center (3D), rank (1-based|None), raw, pixels, points, latency, rejections,
    fallback, error)."""
    res = dict(center_xy=None, center=None, rank=None, raw=None, pixels=[], points=[], latency=None,
               rejections=[], fallback=True, error=None)
    t0 = time.time()
    try:
        rgb = obs["rgb"]
        res["raw"] = vlm(rgb) if vlm is not None else ask(rgb, model, host, timeout)
    except Exception as e:  # timeout, connection refused, bad reply ...
        res["error"] = f"{type(e).__name__}: {e}"
    res["latency"] = time.time() - t0
    if res["raw"] is None:
        return res
    h, w = np.asarray(obs["rgb"]).shape[:2]
    res["pixels"] = parse_ranked(res["raw"], (w, h))
    if not res["pixels"]:
        res["error"] = "parse_failure"
        return res
    for r, (u, v) in enumerate(res["pixels"], 1):
        p = pixel_to_point(u, v, obs)
        res["points"].append(None if p is None else p.tolist())
        ok, why = validate(p, obs["verts"], (u, v), obs["rgb"])
        if ok:
            res.update(center_xy=(float(p[0]), float(p[1])), center=p.tolist(), rank=r, fallback=False)
            break
        res["rejections"].append(dict(rank=r, pixel=(u, v), reason=why))
    if res["fallback"]:
        res["error"] = "no_valid_candidate"
    return res


COLORS = dict(chosen=(0, 255, 0), backup=(255, 215, 0), rejected=(255, 60, 60), other=(200, 200, 200))


def candidate_status(res):
    """[(rank, (u, v), status)] with status chosen | backup | rejected | other.
    With res["valid_candidates"] / res["rejections"] (integrated demo): chosen = the used rank, backup = valid but not
    used (kept for a retry), rejected = failed validation, other = neither (e.g. a duplicate of an earlier point).
    Without those keys the old rule applies: the chosen rank, everything else rejected."""
    new = "valid_candidates" in res or "rejections" in res
    valid = {c["rank"] for c in res.get("valid_candidates") or []}
    rej = {r["rank"] for r in res.get("rejections") or []}
    chosen = None if res.get("fallback") else res.get("rank")
    out = []
    for r, (u, v) in enumerate(res["pixels"], 1):
        if r == chosen:
            st = "chosen"
        elif not new or r in rej:
            st = "rejected"
        elif r in valid:
            st = "backup"
        else:
            st = "other"
        out.append((r, (u, v), st))
    return out


def draw(obs, res, path):
    """Annotate the candidates: chosen green, valid-but-unused (retry backup) yellow, rejected red with " x"."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.asarray(obs["rgb"], np.uint8)).convert("RGB")
    d = ImageDraw.Draw(im)
    for r, (u, v), st in candidate_status(res):
        col = COLORS[st]
        rad = 9 if st == "chosen" else 6
        d.ellipse([u - rad, v - rad, u + rad, v + rad], outline=col, width=3)
        d.text((u + rad + 2, v - rad), f"{r}" + (" x" if st == "rejected" else ""), fill=col)
    im.save(path)


def _selftest():
    rng = np.random.default_rng(0)
    cx, cy = P.BASKET_CENTER_XY
    # synthetic heap: dense gently domed surface over the basket, camera 1 m above looking down
    gx, gy = np.meshgrid(np.arange(cx - 0.2, cx + 0.2, 0.004), np.arange(cy - 0.17, cy + 0.17, 0.004))
    gz = FLOOR_Z + 0.03 + 0.05 * np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / 0.01)
    verts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    W, H, f = 640, 480, 480 / 2 / np.tan(np.radians(30))
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    R = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1.0]])       # camera z = -base z, image up = +x
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = (cx, cy, 1.0)
    w2c = np.linalg.inv(c2w)
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:] = (40, 120, 190)                                      # blue basket
    rgb[H // 2 - 60:H // 2 + 60, W // 2 - 90:W // 2 + 90] = (230, 130, 40)   # orange towel patch around the centre
    base = dict(verts=verts, K=K, rgb=rgb)
    obs_z = dict(base, E=w2c, E_mode="world2cam")
    obs_c = dict(base, E=c2w, E_mode="cam2world")
    # per-pixel point cloud + depth rendered from the analytic surface
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    pts = np.full((H, W, 3), np.nan)
    for tgt in [(cx, cy), (cx + 0.06, cy - 0.04), (cx - 0.08, cy + 0.07)]:
        t = np.array([*tgt, FLOOR_Z + 0.03 + 0.05 * np.exp(-((tgt[0] - cx) ** 2 + (tgt[1] - cy) ** 2) / 0.01)])
        for o in (obs_z, obs_c):
            uv, _ = project(o, t[None])
            p = pixel_to_point(uv[0, 0], uv[0, 1], o)
            assert p is not None and np.linalg.norm(p - t) < 3e-3, ("zbuf", np.linalg.norm(p - t))
        pts[int(round(uv[0, 1])), int(round(uv[0, 0]))] = t
        uvr = uv[0]
        d = float(project(obs_z, t[None])[1][0])
        dep = np.zeros((H, W))
        dep[int(round(uvr[1])), int(round(uvr[0]))] = d
        p = pixel_to_point(round(uvr[0]), round(uvr[1]), dict(obs_z, points=pts))
        assert np.linalg.norm(p - t) < 1e-9
        p = pixel_to_point(uvr[0], uvr[1], dict(obs_z, depth=dep))
        assert np.linalg.norm(p - t) < 3e-3, ("depth", np.linalg.norm(p - t))
    # no surface far outside the heap
    assert pixel_to_point(5, 5, obs_z) is None
    # validation reasons
    def val(xyz):
        uv, _ = project(obs_z, np.array([xyz]))
        return validate(np.array(xyz), verts, uv[0], rgb)
    assert val([cx, cy, FLOOR_Z + 0.08])[0]
    assert val([cx, cy + 0.09, FLOOR_Z + 0.03 + 0.05 * np.exp(-0.0081 / 0.01)])[0]   # on the towel, off-centre
    assert val([cx + P.BASKET_INNER[0] / 2 - 0.02, cy, FLOOR_Z + 0.03])[1] == "wall_band"
    assert val([cx + P.BASKET_INNER[0] / 2 + 0.05, cy, FLOOR_Z + 0.03])[1] == "outside_basket"
    assert val([cx, cy, FLOOR_Z + 0.2])[1] == "no_cloth"
    assert val([cx + 0.12, cy, FLOOR_Z + 0.03])[1] == "not_towel"  # inside the basket, on the blue floor
    assert validate(None, verts, (0, 0), rgb)[1] == "no_surface"
    # parse_ranked
    j = '{"points": [{"x": 500, "y": 250}, {"x": 100, "y": 900}]}'
    assert parse_ranked("```json\n" + j + "\n```") == [(500, 250), (100, 900)]
    assert parse_ranked("Sure! " + j + " done") == [(500, 250), (100, 900)]
    assert parse_ranked("[[1,2],[3,4]]") == [(1, 2), (3, 4)]
    assert parse_ranked("I cannot see anything") == [] and parse_ranked("") == []
    assert parse_ranked(j, (640, 480))[0] == (320, 120)
    # choose_center: rank 1 wall-band pixel, rank 2 good (heap centre) -> rank 2
    uvc, _ = project(obs_z, np.array([[cx, cy, FLOOR_Z + 0.08]]))
    uvw, _ = project(obs_z, np.array([[cx + 0.16, cy, FLOOR_Z + 0.03]]))
    ans = json.dumps({"points": [{"x": uvw[0, 0] / W * 1000, "y": uvw[0, 1] / H * 1000},
                                 {"x": uvc[0, 0] / W * 1000, "y": uvc[0, 1] / H * 1000}]})
    r = choose_center(obs_z, vlm=lambda im: ans)
    assert not r["fallback"] and r["rank"] == 2 and r["rejections"][0]["reason"] == "wall_band", r
    assert abs(r["center_xy"][0] - cx) < 5e-3 and abs(r["center_xy"][1] - cy) < 5e-3
    # fallbacks
    def boom(im):
        raise TimeoutError("timed out")
    r = choose_center(obs_z, vlm=boom)
    assert r["fallback"] and r["center_xy"] is None and "TimeoutError" in r["error"]
    r = choose_center(obs_z, vlm=lambda im: "no idea")
    assert r["fallback"] and r["error"] == "parse_failure"
    r = choose_center(obs_z, vlm=lambda im: '[{"x": 5, "y": 5}]')
    assert r["fallback"] and r["error"] == "no_valid_candidate" and r["rejections"][0]["reason"] == "no_surface"
    # candidate colours: chosen / backup / rejected; old behaviour without the new keys
    px = [(1, 1), (2, 2), (3, 3), (4, 4)]
    st = lambda r: [x[2] for x in candidate_status(r)]
    assert st(dict(pixels=px, rank=1, fallback=False)) == ["chosen", "rejected", "rejected", "rejected"]
    assert st(dict(pixels=px, rank=1, fallback=False, valid_candidates=[dict(rank=1), dict(rank=2)],
                   rejections=[dict(rank=3, reason="floor")])) == ["chosen", "backup", "rejected", "other"]
    assert st(dict(pixels=px[:2], rank=None, fallback=True, valid_candidates=[],
                   rejections=[dict(rank=1), dict(rank=2)])) == ["rejected", "rejected"]
    del rng
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("obs", nargs="?", help="observation npz")
    ap.add_argument("--mock", help="VLM answer text instead of calling ollama")
    ap.add_argument("--out", help="annotated png")
    ap.add_argument("--e-mode", choices=["world2cam", "cam2world"])
    ap.add_argument("--model", default="qwen3-vl:8b-instruct")
    ap.add_argument("--host", default=OLLAMA)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.obs:
        ap.error("OBS.npz required")
    obs = load_obs(a.obs, a.e_mode)
    res = choose_center(obs, vlm=(lambda im: a.mock) if a.mock is not None else None,
                        model=a.model, host=a.host, timeout=a.timeout)
    print(json.dumps(res, indent=1, default=str))
    if a.out:
        draw(obs, res, a.out)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
