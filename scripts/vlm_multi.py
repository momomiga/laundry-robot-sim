"""Multi-garment perception from one basket-camera observation (CPU; the VLM is injectable).

Pipeline (no simulator ground truth is used for detection / centroid / grasp):
  RGB -> VLM (qwen3-vl via ollama) returns one labelled box per garment ([{"bbox_2d": [x1,y1,x2,y2], "label": ..}],
  0-1000 normalised) -> per-detection segmentation from colour + point cloud -> centroid, grasp candidate, pick order.

Segmentation (segment_detections):
  1. basket-interior pixels = valid cloud points whose xy lie inside the basket inner footprint shrunk by INSET (so the
     inner wall faces, which sit exactly on the footprint edge, are excluded) and whose z >= floor - 5 mm.  NOTE: the
     floor itself is *kept* (a cloth lying flat on the floor sits at floor height, so a "above floor + 3 mm" test would
     drop it); garments are separated from the floor by colour instead.
  2. basket colour = median feature of the wall-side strip of the interior, refined once by the median of the pixels
     close to it (a garment would have to cover more than half of the strip to spoil it).  Feature = (r/s, g/s,
     LUMA_W*log luma): chromaticity dominates, so shadows on the blue basket stay "basket" while a white / orange /
     green cloth is far away.  Garments whose chromaticity is close to the basket's blue are NOT detected.
  3. garment pixels = interior pixels with feature distance > GARMENT_THR from the basket colour.
  4. k-means (k = detections that contain garment pixels) on the garment-pixel features inside the union of boxes,
     initialised from each box's median garment feature.  A detection's mask = pixels inside its own box assigned to
     its cluster, then the largest 8-connected component.
  5. Overlapping boxes with different colours: the overlap is split by colour (each garment keeps its own cluster).
     Detections whose colours are similar (centre distance < SIMILAR_THR): they share a cluster group; a pixel in
     several of their boxes goes to the detection whose box centre is nearest (box-normalised), so two similar-looking
     garments are separated only as well as their boxes allow.
  6. A box with (almost) no garment pixels (e.g. over the bare basket floor) gives an empty mask, status "rejected".

Observation npz: see vlm_grasp.py (rgb, K, E, points / pc_world+pc_mask or depth, optional seg_gt (H,W) int, -1 = no
garment, obj_names).  `verts` / `seg_gt` / `obj_names` are used for evaluation only.

Real-VLM run on a saved observation (needs `~/.local/ollama/bin/ollama serve` on 127.0.0.1:11434):
  uv run --no-sync python scripts/vlm_multi.py --obs OBS.npz --out-json out.json --out-png out.png
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlm_grasp as vg  # noqa: E402

P = vg.P
WALL_BAND = vg.WALL_BAND
FLOOR_Z = vg.FLOOR_Z
LABELS = ("towel", "t-shirt", "sock")
INSET = 0.012            # interior footprint is shrunk by this against the wall faces [m]
Z_TOL = 0.005            # interior points must have z >= floor - Z_TOL [m]
LUMA_W = 0.1             # weight of log-luminance in the colour feature
GARMENT_THR = 0.10       # feature distance to the basket colour above which a pixel is "garment"
MAX_CLUSTER_DIST = 0.10  # a pixel farther than this from its nearest cluster centre belongs to no detection
SIMILAR_THR = 0.06       # two cluster centres closer than this are "similar colour"
MIN_AREA = 30            # px; smaller masks are rejected
GRASP_R = 0.05           # grasp search radius around the 3D centroid (xy) [m]
HEIGHT_Q = 80            # percentile of mask heights used for the pick order
HEIGHT_TIE = 0.002       # heights within this are ties (broken by area) [m]
MIN_RESIDUAL_PX = 400    # leftover garment region must have at least this many px to be re-asked. The smallest visible
#                          garment seen (a sock half under a t-shirt) is ~960 px, so 400 keeps every real one while
#                          tight-box slivers / shading remnants (tens to a few hundred px) are not worth a VLM call
RESIDUAL_DILATE = 1      # detection masks are grown by this many px before subtracting (their boundary rim is not a leftover)
RESIDUAL_OPEN = 1        # opening iterations (3x3) that remove thin slivers from the residual
MAX_RECHECK = 3          # at most this many re-asks (largest region first)
RECHECK_PAD = 6          # padding of the red rectangle around a residual region [px]

PROMPT = (
    "This is a top-down camera image of a laundry basket containing several separate garments. "
    "Detect every separate garment (towel, t-shirt, sock) and give one tight bounding box per garment, "
    "covering only its visible part. Label each box with exactly one of: towel, t-shirt, sock. "
    'Answer with JSON only, a list of objects: [{"bbox_2d": [x1, y1, x2, y2], "label": "towel"}, ...] '
    "where x is horizontal and y vertical in 0-1000 normalised image coordinates (0,0 = top-left, "
    "x1,y1 = top-left corner, x2,y2 = bottom-right corner)."
)

RECHECK_PROMPT = (
    "This is a top-down camera image of a laundry basket. A red rectangle marks a region. Is the cloth inside the red "
    "rectangle a separate garment (towel, t-shirt or sock) lying in the basket, and not just a part of a larger "
    "garment or the basket itself? Answer with JSON only: "
    '{"garment": true or false, "label": "towel" or "t-shirt" or "sock"}'
)


def parse_recheck(text):
    """-> (confirmed: bool, label).  Tolerant of fences, prose, true/"yes" strings; a missing `garment` or `label`
    field means NOT confirmed."""
    t = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S)
    t = re.sub(r"```[a-zA-Z]*", "", t)
    obj = None
    i, j = t.find("{"), t.rfind("}")
    if 0 <= i < j:
        try:
            obj = json.loads(t[i:j + 1])
        except ValueError:
            obj = None
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        obj = obj[0]
    g = lab = None
    if isinstance(obj, dict):
        g, lab = obj.get("garment"), obj.get("label")
    else:
        m = re.search(r'"?garment"?\s*:\s*"?(true|false|yes|no)', t, re.I)
        g = m.group(1) if m else None
        m = re.search(r'"?label"?\s*:\s*"([^"]*)"', t)
        lab = m.group(1) if m else None
    if isinstance(g, str):
        g = {"true": True, "yes": True, "false": False, "no": False}.get(g.strip().lower())
    if g is not True or not lab:
        return False, None
    return True, norm_label(lab)


# ------------------------------------------------------------------ parsing
def norm_label(s):
    t = re.sub(r"[^a-z]", "", str(s).lower())
    if "sock" in t or "stocking" in t:
        return "sock"
    if "shirt" in t or "tee" == t or "top" == t:
        return "t-shirt"
    if "towel" in t or "cloth" == t:
        return "towel"
    return "unknown"


def _num_list(v):
    try:
        out = [float(x) for x in v]
    except (TypeError, ValueError):
        return None
    return out if len(out) == 4 and all(np.isfinite(out)) else None


def _items_from_obj(obj):
    """Flatten a parsed JSON value into [(box4 | None, label)]."""
    res = []
    if isinstance(obj, dict):
        for k in ("bbox_2d", "bbox", "box", "bounding_box", "box_2d"):
            if k in obj:
                b = _num_list(obj[k]) if isinstance(obj[k], (list, tuple)) else None
                if b is None and isinstance(obj[k], dict):
                    d = obj[k]
                    b = _num_list([d.get(x) for x in ("x1", "y1", "x2", "y2")]) if all(
                        x in d for x in ("x1", "y1", "x2", "y2")) else None
                return [(b, obj.get("label", obj.get("name", obj.get("class", ""))))]
        for v in obj.values():            # {"objects": [...]} / {"detections": [...]}
            if isinstance(v, (list, dict)):
                res += _items_from_obj(v)
        return res
    if isinstance(obj, list):
        if len(obj) == 4 and _num_list(obj) is not None:
            return [(_num_list(obj), "")]
        for it in obj:
            res += _items_from_obj(it)
    return res


def parse_boxes(text, size=None):
    """Tolerant parse -> list of dict(box=(x1,y1,x2,y2), label, label_raw).
    Coordinates are 0-1000 normalised in the reply; `size`=(W,H) converts to pixels, else normalised values are kept.
    Handles code fences, <think> blocks, prose around the JSON, a single object instead of a list, bbox / bbox_2d /
    box keys, swapped corners (sorted), out-of-range (clipped to 0-1000), degenerate or zero-area boxes (dropped),
    unknown labels (kept as "unknown"), and broken JSON (regex over each {...} block)."""
    t = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S)
    t = re.sub(r"```[a-zA-Z]*", "", t)
    items, obj = [], None
    cands = [t.strip()]
    for a, b in (("[", "]"), ("{", "}")):
        i, j = t.find(a), t.rfind(b)
        if 0 <= i < j:
            cands.append(t[i:j + 1])
    for c in cands:
        try:
            obj = json.loads(c)
            break
        except ValueError:
            obj = None
    if obj is not None:
        items = _items_from_obj(obj)
    if not items:   # broken JSON: pick numbers out of each {...} block
        for blk in re.findall(r"\{[^{}]*\}", t):
            m = re.search(r'"?(?:bbox_2d|bbox|box|box_2d)"?\s*[:=]\s*\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,'
                          r'\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)', blk)
            if m:
                lm = re.search(r'"?label"?\s*[:=]\s*"?([A-Za-z_\- ]+)', blk)
                items.append(([float(g) for g in m.groups()], lm.group(1) if lm else ""))
    out = []
    for b, lab in items:
        if b is None:
            continue
        x1, x2 = sorted((b[0], b[2]))
        y1, y2 = sorted((b[1], b[3]))
        x1, x2, y1, y2 = (min(max(v, 0.0), 1000.0) for v in (x1, x2, y1, y2))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        box = (x1, y1, x2, y2)
        if size is not None:
            box = (x1 / 1000 * size[0], y1 / 1000 * size[1], x2 / 1000 * size[0], y2 / 1000 * size[1])
        out.append(dict(box=tuple(float(v) for v in box), label=norm_label(lab), label_raw=str(lab)))
    return out


# ------------------------------------------------------------------ geometry / colour
def get_points(obs):
    """(H,W,3) cloud with NaN for invalid; from `points`, else from `depth` + K/E."""
    if obs.get("points") is not None:
        return np.asarray(obs["points"], float)
    if obs.get("depth") is not None:
        d = np.asarray(obs["depth"], float)
        h, w = d.shape
        K = np.asarray(obs["K"], float)
        uu, vv = np.meshgrid(np.arange(w), np.arange(h))
        ok = np.isfinite(d) & (d > 0)
        pc = np.stack([(uu - K[0, 2]) / K[0, 0] * d, (vv - K[1, 2]) / K[1, 1] * d, d], -1)
        E = vg._w2c(obs)
        pw = (pc - E[:3, 3]) @ E[:3, :3]
        pw[~ok] = np.nan
        return pw
    raise ValueError("observation has neither `points` nor `depth`")


def color_feat(rgb):
    c = np.asarray(rgb, float)
    s = c.sum(-1, keepdims=True) + 1e-6
    luma = 0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2]
    return np.concatenate([c[..., :2] / s, LUMA_W * np.log(luma[..., None] + 8.0)], -1)


def interior_mask(pts, inset=INSET):
    cx, cy = P.BASKET_CENTER_XY
    ix, iy, _ = P.BASKET_INNER
    ok = np.isfinite(pts).all(-1)
    with np.errstate(invalid="ignore"):
        return (ok & (np.abs(pts[..., 0] - cx) < ix / 2 - inset) & (np.abs(pts[..., 1] - cy) < iy / 2 - inset)
                & (pts[..., 2] >= FLOOR_Z - Z_TOL))


def basket_reference(feat, pts, inter):
    """Median feature of the basket (floor/shadow) estimated from the wall-side strip, refined once."""
    cx, cy = P.BASKET_CENTER_XY
    ix, iy, _ = P.BASKET_INNER
    with np.errstate(invalid="ignore"):
        d = np.minimum(ix / 2 - np.abs(pts[..., 0] - cx), iy / 2 - np.abs(pts[..., 1] - cy))
    strip = inter & (d < 0.04)
    if strip.sum() < 50:
        strip = inter
    ref = np.median(feat[strip], 0)
    near = inter & (np.linalg.norm(feat - ref, axis=-1) < GARMENT_THR)
    if near.sum() >= 50:
        ref = np.median(feat[near], 0)
    return ref


def _largest_cc(mask):
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    cnt = np.bincount(lab.ravel())
    cnt[0] = 0
    return lab == int(np.argmax(cnt))


def segment_detections(obs, dets):
    """Add dets[i]["mask"] (H,W bool), ["status"] ("ok" | "rejected"), ["reason"].  Returns (dets, info)."""
    rgb = np.asarray(obs["rgb"], np.uint8)
    h, w = rgb.shape[:2]
    pts = get_points(obs)
    inter = interior_mask(pts)
    feat = color_feat(rgb)
    ref = basket_reference(feat, pts, inter)
    garment = inter & (np.linalg.norm(feat - ref, axis=-1) > GARMENT_THR)
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    boxm = []
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        boxm.append((uu >= x1) & (uu <= x2) & (vv >= y1) & (vv <= y2))
    n = len(dets)
    for d in dets:
        d.update(mask=np.zeros((h, w), bool), status="rejected", reason="no_garment_pixels")
    info = dict(basket_feat=ref.tolist(), garment_px=int(garment.sum()), interior_px=int(inter.sum()), _garment=garment)
    # initial centres: median garment feature in each box
    idx, centres = [], []
    for i in range(n):
        m = boxm[i] & garment
        if m.sum() >= MIN_AREA:
            idx.append(i)
            centres.append(np.median(feat[m], 0))
    if not idx:
        return dets, info
    centres = np.array(centres)
    union = garment & np.any([boxm[i] for i in idx], 0)
    X = feat[union]
    for _ in range(15):                     # Lloyd iterations
        lab = np.argmin(((X[:, None] - centres[None]) ** 2).sum(-1), 1)
        new = np.array([np.median(X[lab == c], 0) if np.any(lab == c) else centres[c] for c in range(len(idx))])
        if np.allclose(new, centres, atol=1e-5):
            break
        centres = new
    lab_img = np.full((h, w), -1, int)
    dd = np.linalg.norm(X[:, None] - centres[None], axis=-1)
    lab_u = np.argmin(dd, 1)
    lab_u[dd.min(1) > MAX_CLUSTER_DIST] = -1     # garment pixel far from every box's colour: left for the recheck
    lab_img[union] = lab_u
    sim = np.linalg.norm(centres[:, None] - centres[None], axis=-1) < SIMILAR_THR   # similar-colour groups
    claim = np.full((len(idx), h, w), False)
    for a, i in enumerate(idx):
        allowed = np.isin(lab_img, np.where(sim[a])[0])
        claim[a] = boxm[i] & allowed
    # a pixel claimed by several detections (overlapping boxes AND similar colour) -> nearest box centre
    multi = claim.sum(0) > 1
    if multi.any():
        score = np.full((len(idx), h, w), np.inf)
        for a, i in enumerate(idx):
            x1, y1, x2, y2 = dets[i]["box"]
            score[a] = np.hypot((uu - (x1 + x2) / 2) / max(x2 - x1, 1), (vv - (y1 + y2) / 2) / max(y2 - y1, 1))
        score[~claim] = np.inf
        best = np.argmin(score, 0)
        for a in range(len(idx)):
            claim[a] &= ~multi | (best == a)
    for a, i in enumerate(idx):
        m = _largest_cc(claim[a]) if claim[a].any() else claim[a]
        dets[i]["mask"] = m
        if m.sum() >= MIN_AREA:
            dets[i].update(status="ok", reason="")
        else:
            dets[i].update(status="rejected", reason="mask_too_small")
    return dets, info


def residual_components(garment, dets, min_px=None):
    """Garment pixels not claimed by any ok detection (masks grown by RESIDUAL_DILATE), opened, connected components
    with >= min_px pixels, largest first -> list of (mask, area).  Components are not split by colour: two different
    garments touching in the leftover would be one region and be asked about once (a sock next to a leftover patch of
    cloth is still one region of "unexplained cloth")."""
    from scipy import ndimage
    min_px = MIN_RESIDUAL_PX if min_px is None else min_px
    claimed = np.zeros_like(garment)
    for d in dets:
        if d["status"] == "ok":
            claimed |= d["mask"]
    if claimed.any() and RESIDUAL_DILATE:
        claimed = ndimage.binary_dilation(claimed, structure=np.ones((3, 3)), iterations=RESIDUAL_DILATE)
    res = garment & ~claimed
    if RESIDUAL_OPEN:
        res = ndimage.binary_opening(res, structure=np.ones((3, 3)), iterations=RESIDUAL_OPEN)
    lab, n = ndimage.label(ndimage.binary_dilation(res, structure=np.ones((3, 3)), iterations=5),
                           structure=np.ones((3, 3)))     # pieces <= 10 px apart (a thin occluding fold) are one region
    out = []
    for k in range(1, n + 1):
        m = (lab == k) & res
        if m.sum() >= min_px:
            out.append((m, int(m.sum())))
    return sorted(out, key=lambda t: -t[1])


def recheck_residuals(obs, dets, garment, recheck, max_n=MAX_RECHECK, min_px=None):
    """Show each sizeable leftover region (red rectangle on the full image) to the VLM; confirmed ones are appended
    to `dets` as source="recheck".  Returns the log (one entry per re-ask)."""
    from PIL import Image, ImageDraw
    log = []
    for m, area in residual_components(garment, dets, min_px)[:max_n]:
        vs, us = np.nonzero(m)
        h, w = m.shape
        box = (float(max(us.min() - RECHECK_PAD, 0)), float(max(vs.min() - RECHECK_PAD, 0)),
               float(min(us.max() + 1 + RECHECK_PAD, w)), float(min(vs.max() + 1 + RECHECK_PAD, h)))
        im = Image.fromarray(np.asarray(obs["rgb"], np.uint8)).convert("RGB")
        ImageDraw.Draw(im).rectangle(box, outline=(255, 0, 0), width=3)
        ent = dict(box=box, area_px=area, raw=None, latency=None, error=None, confirmed=False, label=None)
        t0 = time.time()
        try:
            ent["raw"] = recheck(np.asarray(im))
        except Exception as e:
            ent["error"] = f"{type(e).__name__}: {e}"
        ent["latency"] = time.time() - t0
        if ent["raw"] is not None:
            ent["confirmed"], ent["label"] = parse_recheck(ent["raw"])
        if ent["confirmed"]:
            vs, us = np.nonzero(m)
            dets.append(dict(box=(float(us.min()), float(vs.min()), float(us.max() + 1), float(vs.max() + 1)),
                             label=ent["label"], label_raw=str(ent["label"]), mask=m, status="ok", reason="",
                             source="recheck"))
        log.append(ent)
    return log


def describe_detection(obs, d, pts):
    """centroid pixel / 3D point, grasp candidate and height statistic for an ok detection (in place)."""
    m = d["mask"]
    vs, us = np.nonzero(m)
    cu, cv = us.mean(), vs.mean()
    k = int(np.argmin((us - cu) ** 2 + (vs - cv) ** 2))
    d["centroid_px"] = (float(us[k]), float(vs[k]))
    d["centroid_mean_px"] = (float(cu), float(cv))
    mp = pts[vs, us]
    d["centroid_3d"] = pts[vs[k], us[k]].tolist()
    d["centroid_3d_mean"] = np.nanmean(mp, 0).tolist()
    d["height_q"] = float(np.nanpercentile(mp[:, 2], HEIGHT_Q))
    c3 = np.array(d["centroid_3d"])
    near = np.hypot(mp[:, 0] - c3[0], mp[:, 1] - c3[1]) <= GRASP_R
    cx, cy = P.BASKET_CENTER_XY
    ix, iy, _ = P.BASKET_INNER
    inside = (np.abs(mp[:, 0] - cx) <= ix / 2 - WALL_BAND) & (np.abs(mp[:, 1] - cy) <= iy / 2 - WALL_BAND)
    sel = near & inside
    d["grasp_in_wall_band"] = False
    if not sel.any():
        sel, d["grasp_in_wall_band"] = near, True
    j = int(np.argmax(np.where(sel, mp[:, 2], -np.inf)))
    d["grasp_px"] = (float(us[j]), float(vs[j]))
    d["grasp_3d"] = mp[j].tolist()


def pick_order(dets):
    """Indices (into dets) of ok detections, highest first (80th-percentile height; ties by larger area)."""
    ok = [i for i, d in enumerate(dets) if d["status"] == "ok"]
    return sorted(ok, key=lambda i: (-round(dets[i]["height_q"] / HEIGHT_TIE), -int(dets[i]["mask"].sum())))


# ------------------------------------------------------------------ evaluation
def _iou_box(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(iw, 0) * max(ih, 0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / u) if u > 0 else 0.0


def evaluate(obs, dets, pts):
    """Score against obs["seg_gt"] (never used by the perception).  None if absent."""
    if obs.get("seg_gt") is None:
        return None
    from scipy.optimize import linear_sum_assignment
    seg = np.asarray(obs["seg_gt"])
    names = [str(x) for x in obs["obj_names"]] if obs.get("obj_names") is not None else None
    gts = [int(k) for k in np.unique(seg) if k >= 0 and (seg == k).sum() > 0]
    okd = [i for i, d in enumerate(dets) if d["status"] == "ok"]
    iou = np.zeros((len(okd), len(gts)))
    for a, i in enumerate(okd):
        for b, k in enumerate(gts):
            g = seg == k
            iou[a, b] = (dets[i]["mask"] & g).sum() / max((dets[i]["mask"] | g).sum(), 1)
    pairs = []
    if iou.size:
        r, c = linear_sum_assignment(-iou)
        pairs = [(okd[a], gts[b], float(iou[a, b])) for a, b in zip(r, c) if iou[a, b] >= 0.1]
    objs = []
    for i, k, mi in pairs:
        g = seg == k
        vs, us = np.nonzero(g)
        gbox = (us.min(), vs.min(), us.max() + 1, vs.max() + 1)
        gp = pts[g]
        gp = gp[np.isfinite(gp).all(-1)]
        cu, cv = (int(round(x)) for x in dets[i]["centroid_px"])
        e3 = float(np.linalg.norm(np.array(dets[i]["centroid_3d"]) - gp.mean(0))) if len(gp) else None
        em = float(np.linalg.norm(np.array(dets[i]["centroid_3d_mean"]) - gp.mean(0))) if len(gp) else None
        objs.append(dict(
            gt_index=k, gt_name=names[k] if names and k < len(names) else None, det_index=i, label=dets[i]["label"],
            label_correct=(dets[i]["label"] == names[k]) if names and k < len(names) else None,
            box_iou=_iou_box(dets[i]["box"], gbox), mask_iou=mi, centroid_in_gt=bool(g[cv, cu]),
            centroid_err_m=e3, centroid_mean_err_m=em))
    matched_d = {p[0] for p in pairs}
    matched_g = {p[1] for p in pairs}
    return dict(objects=objs, missed=[k for k in gts if k not in matched_g],
                spurious=[i for i in range(len(dets)) if i not in matched_d])


# ------------------------------------------------------------------ top level
def run(obs, vlm=None, model="qwen3-vl:8b-instruct", host=vg.OLLAMA, timeout=120, recheck=None, do_recheck=True):
    """vlm: callable(rgb)->text (mock) or None (ollama).  Never raises on VLM trouble.
    recheck: callable(annotated_rgb)->text for the leftover-region question; None -> ollama with RECHECK_PROMPT when
    `vlm` is None, and no recheck at all when `vlm` is a mock (so a mock run never touches the network).
    do_recheck=False disables it."""
    res = dict(raw=None, latency=None, error=None, detections=[], order=[], evaluation=None, recheck=[])
    t0 = time.time()
    try:
        rgb = obs["rgb"]
        res["raw"] = vlm(rgb) if vlm is not None else vg.ask(rgb, model, host, timeout, prompt=PROMPT)
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    res["latency"] = time.time() - t0
    if res["raw"] is None:
        return res, []
    h, w = np.asarray(obs["rgb"]).shape[:2]
    dets = parse_boxes(res["raw"], (w, h))
    if not dets:
        res["error"] = "parse_failure"
        return res, []
    pts = get_points(obs)
    dets, info = segment_detections(obs, dets)
    garment = info.pop("_garment")
    res["segmentation"] = info
    for d in dets:
        d["source"] = "vlm"
    if do_recheck and (recheck is not None or vlm is None):
        rk = recheck or (lambda im: vg.ask(im, model, host, timeout, prompt=RECHECK_PROMPT))
        res["recheck"] = recheck_residuals(obs, dets, garment, rk)
    for d in dets:
        if d["status"] == "ok":
            describe_detection(obs, d, pts)
    res["order"] = pick_order(dets)
    for rank, i in enumerate(res["order"], 1):
        dets[i]["order"] = rank
    res["evaluation"] = evaluate(obs, dets, pts)
    for d in dets:
        d["area_px"] = int(d["mask"].sum())
    res["detections"] = [{k: v for k, v in d.items() if k != "mask"} for d in dets]
    return res, dets


PALETTE = [(255, 64, 64), (64, 220, 255), (255, 220, 40), (190, 90, 255), (70, 255, 110), (255, 140, 0)]


def _font(px):
    from PIL import ImageFont
    for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, px)
    return ImageFont.load_default()


def draw(obs, res, dets, path):
    from PIL import Image, ImageDraw
    rgb = np.asarray(obs["rgb"], np.uint8).copy()
    for i, d in enumerate(dets):                      # tinted masks
        if d["status"] == "ok":
            c = np.array(PALETTE[i % len(PALETTE)], float)
            rgb[d["mask"]] = (0.6 * rgb[d["mask"]] + 0.4 * c).astype(np.uint8)
    im = Image.fromarray(rgb).convert("RGB")
    dr = ImageDraw.Draw(im)
    f = _font(max(14, im.height // 28))
    for i, d in enumerate(dets):
        col = PALETTE[i % len(PALETTE)]
        if d.get("source") == "recheck":
            x1, y1, x2, y2 = d["box"]
            for a in np.arange(x1, x2, 12):
                dr.line([a, y1, min(a + 6, x2), y1], fill=col, width=3)
                dr.line([a, y2, min(a + 6, x2), y2], fill=col, width=3)
            for b in np.arange(y1, y2, 12):
                dr.line([x1, b, x1, min(b + 6, y2)], fill=col, width=3)
                dr.line([x2, b, x2, min(b + 6, y2)], fill=col, width=3)
        else:
            dr.rectangle(d["box"], outline=col, width=3)
        txt = d["label"] + (" (recheck)" if d.get("source") == "recheck" else "") + ("" if d["status"] == "ok" else f" (rejected: {d['reason']})")
        if d["status"] == "ok":
            txt = f"#{d['order']} {txt}"
        x, y = d["box"][0], max(d["box"][1] - f.size - 4, 0)
        tb = dr.textbbox((x + 2, y), txt, font=f)
        dr.rectangle(tb, fill=(0, 0, 0))
        dr.text((x + 2, y), txt, font=f, fill=col)
        if d["status"] == "ok":
            u, v = d["centroid_px"]
            dr.ellipse([u - 5, v - 5, u + 5, v + 5], fill=col, outline=(0, 0, 0), width=2)
            gu, gv = d["grasp_px"]
            for wd, c in ((6, (0, 0, 0)), (3, col)):
                dr.line([gu - 9, gv - 9, gu + 9, gv + 9], fill=c, width=wd)
                dr.line([gu - 9, gv + 9, gu + 9, gv - 9], fill=c, width=wd)
    im.save(path)


# ------------------------------------------------------------------ selftest
def _synthetic():
    """Basket (blue floor + walls) at the camera footprint, three garments at different heights; analytic cloud."""
    cx, cy = P.BASKET_CENTER_XY
    ix, iy, dep = P.BASKET_INNER
    W, H = 640, 480
    f = H / 2 / np.tan(np.radians(30))
    R = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1.0]])      # as in vlm_grasp selftest: image up = +x, right = -y
    cam = np.array([cx, cy, 1.0])
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = R, cam
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    dirw = np.stack([(uu - W / 2) / f, (vv - H / 2) / f, np.ones((H, W))], -1) @ R.T
    # garments: centre, semi-axes, base height above floor, colour (selftest only)
    blobs = [((cx + 0.02, cy + 0.06), (0.075, 0.055), 0.03, (238, 140, 40)),    # A (partly covered by B)
             ((cx + 0.02, cy - 0.01), (0.06, 0.05), 0.07, (50, 160, 80)),       # B, highest, covers A's edge
             ((cx - 0.08, cy - 0.08), (0.05, 0.04), 0.02, (235, 235, 230))]     # C, lowest
    rim = FLOOR_Z + dep

    def height_colour(x, y):
        h = np.zeros_like(x)
        col = np.zeros(x.shape + (3,))
        dx, dy = np.abs(x - cx), np.abs(y - cy)
        inb = (dx < ix / 2) & (dy < iy / 2)
        wall = ((dx < ix / 2 + 0.01) & (dy < iy / 2 + 0.01)) & ~inb
        h[:] = 0.0
        col[:] = (110, 90, 70)
        h[wall], col[wall] = rim, (40, 95, 130)
        h[inb], col[inb] = FLOOR_Z, (46, 110, 150)
        for k, ((bx, by), (ax, ay), base, c) in enumerate(blobs):
            r2 = ((x - bx) / ax) ** 2 + ((y - by) / ay) ** 2
            m = r2 < 1
            hb = FLOOR_Z + base + 0.01 * (1 - r2)
            top = m & (hb > h)
            h = np.where(top, hb, h)
            col[top] = np.array(c) * (0.85 + 3.0 * (hb[top, None] - FLOOR_Z - base))
        return h, col

    pts = np.zeros((H, W, 3))
    rgb = np.zeros((H, W, 3))
    done = np.zeros((H, W), bool)
    for z in np.arange(rim + 0.01, FLOOR_Z - 0.2, -0.002):
        t = (cam[2] - z)
        x, y = cam[0] + dirw[..., 0] * t / 1.0, cam[1] + dirw[..., 1] * t
        h, col = height_colour(x, y)
        hit = ~done & (z <= h)
        pts[hit] = np.stack([x[hit], y[hit], h[hit]], -1)
        rgb[hit] = col[hit]
        done |= hit
        if done.all():
            break
    rng = np.random.default_rng(0)
    rgb = np.clip(rgb + rng.normal(0, 2.5, rgb.shape), 0, 255).astype(np.uint8)
    # GT: visible garment pixels = pixels whose colour source was a blob (recompute with ids)
    seg = np.full((H, W), -1, np.int16)
    x, y = pts[..., 0], pts[..., 1]
    best = np.full((H, W), -np.inf)
    for k, ((bx, by), (ax, ay), base, _) in enumerate(blobs):
        r2 = ((x - bx) / ax) ** 2 + ((y - by) / ay) ** 2
        hb = FLOOR_Z + base + 0.01 * (1 - r2)
        vis = (r2 < 1) & (np.abs(pts[..., 2] - hb) < 1e-9)
        seg[vis] = k
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    names = np.array(["towel", "t-shirt", "sock"])
    obs = dict(rgb=rgb, K=K, E=np.linalg.inv(c2w), E_mode="world2cam", points=pts, seg_gt=seg, obj_names=names,
               verts=np.zeros((1, 3)))
    del best
    return obs


def _gt_boxes(obs, jitter=0, seed=0):
    seg = obs["seg_gt"]
    h, w = seg.shape
    rng = np.random.default_rng(seed)
    out = []
    for k in sorted(int(v) for v in np.unique(seg) if v >= 0):
        vs, us = np.nonzero(seg == k)
        b = np.array([us.min(), vs.min(), us.max() + 1, vs.max() + 1], float) + rng.integers(-jitter, jitter + 1, 4)
        out.append(dict(bbox_2d=[round(b[0] / w * 1000), round(b[1] / h * 1000), round(b[2] / w * 1000),
                                 round(b[3] / h * 1000)], label=str(obs["obj_names"][k])))
    return out


def _selftest():
    # --- parser
    W, H = 640, 480
    good = '[{"bbox_2d": [100, 200, 300, 400], "label": "towel"}]'
    assert parse_boxes(good, (W, H))[0]["box"] == (64.0, 96.0, 192.0, 192.0)
    assert len(parse_boxes("```json\n" + good + "\n```")) == 1
    assert len(parse_boxes("Sure, here you go: " + good + " Hope this helps.")) == 1
    s = parse_boxes('{"bbox_2d": [100, 200, 300, 400], "label": "T-Shirt"}')    # single object, label case
    assert len(s) == 1 and s[0]["label"] == "t-shirt"
    assert parse_boxes('[{"bbox": [300, 400, 100, 200], "label": "sock"}]')[0]["box"] == (100, 200, 300, 400)  # swapped
    assert parse_boxes('[{"bbox_2d": [-50, 10, 1200, 500], "label": "towel"}]')[0]["box"] == (0, 10, 1000, 500)  # clip
    assert parse_boxes('[{"bbox_2d": [100, 100, 100, 300], "label": "towel"}]') == []          # zero width
    assert parse_boxes('[{"bbox_2d": [1100, 100, 1300, 300], "label": "towel"}]') == []        # clipped to zero area
    assert parse_boxes('[{"bbox_2d": [10, 10, 50, 50], "label": "blanket"}]')[0]["label"] == "unknown"
    assert parse_boxes('[{"bbox_2d": [10, 10, 50, 50], "label": "socks"}, {"bbox_2d": [60, 60, 90, 90], '
                       '"label": "towel"}]')[0]["label"] == "sock"
    br = parse_boxes('[{"bbox_2d": [10, 10, 50, 50], "label": "towel"}, {"bbox_2d": [60, 60, 90, 90] "label": "sock"}')
    assert len(br) == 2 and br[1]["label"] == "sock"                                           # broken JSON
    assert parse_boxes('{"objects": [{"bbox_2d": [1, 2, 30, 40], "label": "sock"}]}')[0]["label"] == "sock"
    assert parse_boxes("I see no garments.") == [] and parse_boxes("") == []
    assert parse_boxes('[{"bbox_2d": [1, 2, "x", 40], "label": "sock"}]') == []
    # --- synthetic three-garment scene
    obs = _synthetic()
    ans = json.dumps(_gt_boxes(obs, jitter=6))
    res, dets = run(obs, vlm=lambda im: ans)
    ev = res["evaluation"]
    assert res["error"] is None and len(dets) == 3 and not ev["missed"] and not ev["spurious"], res.get("error")
    for o in ev["objects"]:
        assert o["mask_iou"] >= 0.8, o
        assert o["centroid_in_gt"] and o["label_correct"], o
        assert o["centroid_err_m"] < 0.03, o
        print("  gt %d %-8s mask IoU %.3f box IoU %.3f centroid err %.1f mm" % (
            o["gt_index"], o["gt_name"], o["mask_iou"], o["box_iou"], o["centroid_err_m"] * 1e3))
    assert [dets[i]["label"] for i in res["order"]] == ["t-shirt", "towel", "sock"], res["order"]   # B, A, C
    for d in dets:
        assert abs(d["grasp_3d"][2] - d["height_q"]) < 0.02 and not d["grasp_in_wall_band"]
    # --- one detection only
    one = json.dumps(_gt_boxes(obs)[1:2])
    res1, d1 = run(obs, vlm=lambda im: one)
    assert len(d1) == 1 and res1["order"] == [0] and res1["evaluation"]["objects"][0]["mask_iou"] >= 0.8
    assert len(res1["evaluation"]["missed"]) == 2
    # --- overlapping boxes of different colour: loose boxes that overlap each other still separate by colour
    allb = json.dumps([dict(b, bbox_2d=[b["bbox_2d"][0] - 40, b["bbox_2d"][1] - 40, b["bbox_2d"][2] + 40,
                                        b["bbox_2d"][3] + 40]) for b in _gt_boxes(obs)])
    resa, _ = run(obs, vlm=lambda im: allb)
    assert all(o["mask_iou"] >= 0.8 for o in resa["evaluation"]["objects"]) and len(resa["evaluation"]["objects"]) == 3
    # --- box over bare floor -> rejected, no crash; a good box next to it is unaffected
    bare = json.dumps([dict(bbox_2d=[640, 700, 760, 840], label="towel")] + _gt_boxes(obs)[:1])
    resb, db = run(obs, vlm=lambda im: bare)
    assert db[0]["status"] == "rejected" and db[0]["mask"].sum() == 0 and db[1]["status"] == "ok"
    assert resb["order"] == [1] and resb["evaluation"]["spurious"] == [0]
    only_bare = json.dumps([dict(bbox_2d=[640, 700, 760, 840], label="towel")])
    rb, dbb = run(obs, vlm=lambda im: only_bare)
    assert dbb[0]["status"] == "rejected" and rb["order"] == []
    # --- recheck: mock first answer omits one blob (the sock)
    gb = _gt_boxes(obs)
    two = json.dumps(gb[:2])
    asked = []

    def yes(im):
        asked.append(np.asarray(im))
        return '```json\n{"garment": true, "label": "sock"}\n```'
    rr, dr_ = run(obs, vlm=lambda im: two, recheck=yes)
    assert len(asked) == 1 and len(dr_) == 3 and dr_[2]["source"] == "recheck" and dr_[0]["source"] == "vlm", rr["recheck"]
    assert (asked[0][..., 0] > 200).any() and rr["recheck"][0]["confirmed"]
    assert len(rr["evaluation"]["objects"]) == 3 and not rr["evaluation"]["missed"], rr["evaluation"]
    assert all(o["mask_iou"] >= 0.8 for o in rr["evaluation"]["objects"]), rr["evaluation"]
    assert [dr_[i]["label"] for i in rr["order"]] == ["t-shirt", "towel", "sock"]
    assert dr_[2]["grasp_3d"] and dr_[2]["centroid_px"]
    # re-ask answers false / garbled / missing fields -> nothing added
    for txt in ('{"garment": false, "label": "sock"}', "no idea", '{"label": "sock"}', '{"garment": true}'):
        rf, df = run(obs, vlm=lambda im: two, recheck=lambda im, t=txt: t)
        assert len(df) == 2 and len(rf["recheck"]) == 1 and not rf["recheck"][0]["confirmed"], txt
    # no residual (all three found) -> no re-ask; mock vlm without recheck -> no re-ask; do_recheck=False -> none
    n0 = len(asked)
    run(obs, vlm=lambda im: ans, recheck=yes)
    run(obs, vlm=lambda im: two)
    run(obs, vlm=lambda im: two, recheck=yes, do_recheck=False)
    assert len(asked) == n0
    # residual below the threshold -> no re-ask
    old = MIN_RESIDUAL_PX
    globals()["MIN_RESIDUAL_PX"] = 10 ** 6
    try:
        rs, _ = run(obs, vlm=lambda im: two, recheck=yes)
    finally:
        globals()["MIN_RESIDUAL_PX"] = old
    assert rs["recheck"] == [] and len(asked) == n0, rs["recheck"]
    # --- failures
    assert run(obs, vlm=lambda im: "nothing")[0]["error"] == "parse_failure"

    def boom(im):
        raise TimeoutError("timed out")
    assert "TimeoutError" in run(obs, vlm=boom)[0]["error"]
    # --- observation without seg_gt / names
    o2 = {k: v for k, v in obs.items() if k not in ("seg_gt", "obj_names")}
    r2, _ = run(o2, vlm=lambda im: ans)
    assert r2["evaluation"] is None and len(r2["order"]) == 3
    print("selftest OK")


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--obs", help="observation npz")
    ap.add_argument("--mock-answer", help="VLM answer text instead of calling ollama (no network)")
    ap.add_argument("--model", default="qwen3-vl:8b-instruct")
    ap.add_argument("--host", default=vg.OLLAMA)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--out-json")
    ap.add_argument("--out-png")
    ap.add_argument("--no-recheck", action="store_true", help="skip the leftover-region question")
    ap.add_argument("--mock-recheck", help="answer text for the leftover-region question (no network)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.obs:
        ap.error("--obs required")
    obs = vg.load_obs(a.obs)
    res, dets = run(obs, vlm=(lambda im: a.mock_answer) if a.mock_answer is not None else None,
                    model=a.model, host=a.host, timeout=a.timeout,
                    recheck=(lambda im: a.mock_recheck) if a.mock_recheck is not None else None,
                    do_recheck=not a.no_recheck)
    res["prompt"], res["recheck_prompt"] = PROMPT, RECHECK_PROMPT
    txt = json.dumps(res, indent=1, default=_jsonable)
    if a.out_json:
        Path(a.out_json).write_text(txt)
        print("wrote", a.out_json)
    else:
        print(txt)
    if a.out_png and dets:
        draw(obs, res, dets, a.out_png)
        print("wrote", a.out_png)
    for i in res["order"]:
        d = res["detections"][i]
        print(f"#{d['order']} {d['label']} [{d['source']}]: area {d['area_px']} px, centroid {d['centroid_px']}, grasp {d['grasp_3d']}")
    if res["error"]:
        print("error:", res["error"])


if __name__ == "__main__":
    main()
