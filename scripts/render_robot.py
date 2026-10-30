"""Render the robot alone (no house, no cloth) for the "ロボット構成" card of the submission video.

Scene: the robot at the origin on a plain ground plane, neutral background, one 3/4 camera (front-left-above),
arm at the planner standby pose, lift at a mid height so arm, lift column and basket are all visible.

Outputs (in --out-dir, default outputs/):
  robot_view.png           plain render (1280x960 by default)
  robot_view_anchors.json  per part: 3D anchor (world, from link poses) and projected pixel (u, v)
  robot_view_labeled.png   render with short Japanese labels outside the silhouette, thin leader lines to the
                           anchors; the virtual basket camera (no mesh) is drawn as a camera marker with a dashed
                           line straight down to the basket centre (its view direction)

Run (GPU, serialised with the other GPU jobs):
  flock /tmp/house_robot-gpu.lock uv run --no-sync python scripts/render_robot.py [--lift 0.25] [--res 1280 960]
      [--pos X Y Z] [--lookat X Y Z] [--fov DEG] [--out-dir outputs]
CPU check of the label layout and the projection (no Genesis import):
  uv run --no-sync python scripts/render_robot.py --selftest
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import params as P  # noqa: E402

FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
JP_INDEX = 0

# 3/4 view from the front-left-above (robot frame: +x forward, +y left, z up). Robot spans about 0.5 x 0.5 x 1.0 m
# (+ the virtual camera marker at z = 1.0 m); looking at z = 0.5 m from ~2.7 m with a 32 deg vertical fov, ~37 deg
# elevation (picked from test renders: the basket interior shows over its near wall, labels fit beside the robot).
CAM_POS = (1.6, 1.4, 2.1)
CAM_LOOKAT = (0.0, 0.0, 0.5)
CAM_FOV = 32.0
DEFAULT_LIFT = 0.25
# planner standby arm joints joint1..joint8 (scripts/drift_probe.py:23, STANDBY[4:]; the lift there is 0.5 m)
STANDBY_ARM = [-0.0055, 0.2178, 0.0, -0.002, 0.0091, -0.0612, 0.0, -0.0]
JOINT_NAMES = ["base_x_joint", "base_y_joint", "base_yaw_joint", "lift_joint"] + [f"joint{i}" for i in range(1, 9)]

INK = (28, 32, 40)
LEADER = (60, 66, 76)
PAD = 16          # min distance of a label box from the image edge [px]
GAP = 24          # silhouette -> label column gap [px]
LINE_GAP = 18     # default min vertical gap between label boxes in a column [px] (draw_labeled uses 0.4 x font)
CROP_MARGIN = 24  # margin around the union of silhouette and label boxes in the crop output [px]

# part key -> (label, link, local point in that link's frame, or None for the special cases handled in anchors_world)
LABELS = {
    "base": "台車",
    "lift": "昇降軸",
    "arm": "6自由度アーム\nPiPER",
    "gripper": "グリッパ",
    "basket": "籠",
    "basket_camera": "籠カメラ(仮想)",
}


# ------------------------------------------------------------------ pure geometry (CPU)
def lookat_camera(pos, lookat, res, fov_deg, up=(0.0, 0.0, 1.0)):
    """World->camera extrinsics E (4x4, OpenCV: x right, y down, z forward) and intrinsics K (3x3), vertical fov."""
    pos = np.asarray(pos, float)
    f = np.asarray(lookat, float) - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, float))
    r /= np.linalg.norm(r)
    d = np.cross(f, r)           # down
    R = np.stack([r, d, f])      # rows: camera axes in world
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = -R @ pos
    w, h = res
    fy = 0.5 * h / np.tan(np.radians(fov_deg) / 2)
    K = np.array([[fy, 0, w / 2], [0, fy, h / 2], [0, 0, 1.0]])
    return E, K


def project(E, K, pts_world):
    """World points (N,3) -> (uv (N,2) with u right / v down, z-depth (N,)). Same convention as
    scripts/robot_sensors.py project_points (Genesis camera.extrinsics = world->camera, camera.intrinsics)."""
    pts = np.asarray(pts_world, float).reshape(-1, 3)
    E = np.asarray(E, float)
    pc = pts @ E[:3, :3].T + E[:3, 3]
    uv = pc @ np.asarray(K, float).T
    return uv[:, :2] / uv[:, 2:3], pc[:, 2]


def layout_labels(anchors_uv, sizes, img_wh, bbox, pad=PAD, gap=GAP, line_gap=LINE_GAP):
    """Place one label per anchor in a left or right column outside bbox = (x0, y0, x1, y1) (robot silhouette).
    A label goes to the side of its anchor (left of the bbox centre -> left column) unless it does not fit there
    but fits on the other side. Within a column labels are ordered by anchor v and pushed apart to keep `line_gap`.
    Returns rects (x0, y0, x1, y1), all inside the image (>= pad from the edge)."""
    W, H = img_wh
    cx = 0.5 * (bbox[0] + bbox[2])
    room = {"L": bbox[0] - gap - pad, "R": W - pad - (bbox[2] + gap)}
    cols = {"L": [], "R": []}
    for i, a in enumerate(anchors_uv):
        side = "L" if a[0] < cx else "R"
        other = "R" if side == "L" else "L"
        if sizes[i][0] > room[side] and sizes[i][0] <= room[other]:
            side = other
        cols[side].append(i)
    rects = [None] * len(sizes)
    for side, idx in cols.items():
        idx.sort(key=lambda i: anchors_uv[i][1])
        ys = [anchors_uv[i][1] - sizes[i][1] / 2 for i in idx]
        # push down, then pull back up from the bottom if the column overflows
        for k in range(len(idx)):
            lo = pad if k == 0 else ys[k - 1] + sizes[idx[k - 1]][1] + line_gap
            ys[k] = max(ys[k], lo)
        for k in range(len(idx) - 1, -1, -1):
            hi = H - pad - sizes[idx[k]][1] if k == len(idx) - 1 else ys[k + 1] - line_gap - sizes[idx[k]][1]
            ys[k] = min(ys[k], hi)
        for k, i in enumerate(idx):
            w, h = sizes[i]
            if side == "L":
                x1 = bbox[0] - gap
                x0 = max(x1 - w, pad)
                x1 = x0 + w
            else:
                x0 = bbox[2] + gap
                x1 = min(x0 + w, W - pad)
                x0 = x1 - w
            rects[i] = (x0, ys[k], x1, ys[k] + h)
    return rects


def crop_box(boxes, margin, img_wh):
    """Union of boxes (x0, y0, x1, y1) plus margin, clamped to the image, as integers."""
    b = np.asarray(boxes, float)
    W, H = img_wh
    return (int(max(0, np.floor(b[:, 0].min() - margin))), int(max(0, np.floor(b[:, 1].min() - margin))),
            int(min(W, np.ceil(b[:, 2].max() + margin))), int(min(H, np.ceil(b[:, 3].max() + margin))))


def rects_overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def pose_apply(pos, quat_wxyz, p_local):
    """Point in a link frame -> world, given the link position and quaternion (w, x, y, z)."""
    w, x, y, z = np.asarray(quat_wxyz, float)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return np.asarray(pos, float) + R @ np.asarray(p_local, float)


# ------------------------------------------------------------------ drawing
def _font(px):
    try:
        return ImageFont.truetype(FONT_REG, px, index=JP_INDEX)
    except OSError:
        return ImageFont.load_default()


def dashed_line(draw, p0, p1, fill, width=2, dash=9, gap=6):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    if L < 1e-6:
        return
    d = (p1 - p0) / L
    t = 0.0
    while t < L:
        a, b = p0 + d * t, p0 + d * min(t + dash, L)
        draw.line([tuple(a), tuple(b)], fill=fill, width=width)
        t += dash + gap


def draw_camera_marker(draw, c, s=11, fill=INK):
    """Small camera icon (body + lens) centred at c, with a white outline so it reads on any background."""
    x, y = c
    draw.rounded_rectangle([x - s, y - 0.7 * s, x + s, y + 0.7 * s], radius=3, fill=fill, outline=(255, 255, 255), width=2)
    draw.ellipse([x - 0.45 * s, y - 0.45 * s, x + 0.45 * s, y + 0.45 * s], fill=(235, 238, 242), outline=(255, 255, 255))
    draw.rectangle([x - 0.7 * s, y - 0.7 * s - 4, x - 0.1 * s, y - 0.7 * s], fill=fill)


def text_box(draw, text, font, font_px):
    """Box size (w, h), text offset (dx, dy) so that the ink bbox of `text` lies fully inside the padded box."""
    spacing = int(0.15 * font_px)
    bb = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    padx, pady = int(round(0.3 * font_px)), int(round(0.2 * font_px))
    return (bb[2] - bb[0] + 2 * padx, bb[3] - bb[1] + 2 * pady), (padx - bb[0], pady - bb[1]), spacing


def draw_labeled(rgb, anchors, bbox, font_px=30, margin=CROP_MARGIN):
    """anchors: dict key -> dict(uv=(u, v), ...); bbox = robot silhouette. Returns (labelled PIL image, crop PIL image,
    crop box, label rects)."""
    img = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font(font_px)
    keys = list(LABELS)
    uvs = [anchors[k]["uv"] for k in keys]
    cam_uv, ctr_uv = anchors["basket_camera"]["uv"], anchors["basket_camera"]["target_uv"]
    mk = 14   # camera marker half size incl. outline
    # keep boxes clear of the silhouette and of the camera marker / its dashed line
    avoid = (min(bbox[0], cam_uv[0] - mk, ctr_uv[0] - 6), min(bbox[1], cam_uv[1] - mk), max(bbox[2], cam_uv[0] + mk, ctr_uv[0] + 6),
             max(bbox[3], cam_uv[1] + mk, ctr_uv[1] + 6))
    tb = [text_box(draw, LABELS[k], font, font_px) for k in keys]
    rects = layout_labels(uvs, [t[0] for t in tb], img.size, avoid, line_gap=0.4 * font_px)
    dashed_line(draw, cam_uv, ctr_uv, INK)
    draw.ellipse([ctr_uv[0] - 5, ctr_uv[1] - 5, ctr_uv[0] + 5, ctr_uv[1] + 5], outline=INK, width=2)
    draw_camera_marker(draw, cam_uv)
    for k, uv, r, (_, off, spacing) in zip(keys, uvs, rects, tb):
        left = r[2] <= uv[0]
        edge = (r[2] if left else r[0], 0.5 * (r[1] + r[3]))
        draw.line([edge, tuple(uv)], fill=(255, 255, 255), width=5)   # halo so the leader reads on dark parts
        draw.line([edge, tuple(uv)], fill=LEADER, width=2)
        draw.ellipse([uv[0] - 5, uv[1] - 5, uv[0] + 5, uv[1] + 5], fill=INK, outline=(255, 255, 255), width=2)
        draw.rounded_rectangle(r, radius=6, fill=(250, 250, 248), outline=LEADER, width=1)
        draw.multiline_text((r[0] + off[0], r[1] + off[1]), LABELS[k], font=font, fill=INK, spacing=spacing)
    cb = crop_box([avoid] + list(rects), margin, img.size)
    return img, img.crop(cb), cb, rects


# ------------------------------------------------------------------ Genesis (GPU)
def _np(x):
    return np.asarray(x.cpu().numpy() if hasattr(x, "cpu") else x, float).reshape(-1)


def anchors_world(robot):
    """Per part: world anchor from link poses after the pose is set (base_link frame for the platform parts)."""
    bl = robot.get_link("base_link")
    bp, bq = _np(bl.get_pos()), _np(bl.get_quat())

    def in_base(p):
        return pose_apply(bp, bq, p)

    bx, by = P.BASKET_CENTER_XY
    top = P.BASE_Z1 + P.BASKET_INNER[2]
    return {
        "base": dict(p=in_base((0.10, P.BASE_SIZE[1] / 2, 0.15)), note="base_link, left face of the platform"),
        "lift": dict(p=in_base((-0.19, 0.17, 0.40)), note="base_link, lift column (0.05 m square mast) at z 0.40, below the carriage"),
        "arm": dict(p=_np(robot.get_link("link3").get_pos()), note="link3 position"),
        "gripper": dict(p=_np(robot.get_link("link6").get_pos()), note="link6 position (gripper_base is merged into link6)"),
        "basket": dict(p=in_base((bx, by + P.BASKET_INNER[1] / 2 + P.BASKET_WALL / 2, top + 0.0)),
                       note="base_link, top edge of the left basket wall"),
        "basket_camera": dict(p=in_base(P.BASKET_CAM_POS), virtual=True,
                              target=in_base((bx, by, P.BASE_Z1 + 0.01)),
                              note="BASKET_CAM_POS in the robot frame -> world; virtual, no mesh; looks straight down"),
    }


def silhouette_bbox(rgb):
    """Bounding box (x0, y0, x1, y1) of the robot in a render with a light grey gradient ground and a soft shadow:
    pixels that are dark (< 150 mean; ground and shadow stay above ~170) or coloured (max-min channel > 14)."""
    a = np.asarray(rgb).astype(int)[..., :3]
    mask = (a.mean(2) < 150) | ((a.max(2) - a.min(2)) > 14)
    ys, xs = np.where(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def save_labeled(out, rgb, meta, font_px):
    out = Path(out)
    bbox = meta.get("bbox") or silhouette_bbox(rgb)
    img, crop, cb, rects = draw_labeled(rgb, meta["parts"], bbox, font_px=font_px)
    img.save(out / "robot_view_labeled.png")
    crop.save(out / "robot_view_labeled_crop.png")
    print("silhouette bbox", bbox, "| saved", out / "robot_view_labeled.png", "and", out / "robot_view_labeled_crop.png",
          "crop box", cb, "size", crop.size)


def relabel(args):
    """CPU only: redo the labelled images from DIR/robot_view.png and DIR/robot_view_anchors.json."""
    d = Path(args.relabel)
    rgb = np.asarray(Image.open(d / "robot_view.png").convert("RGB"))
    meta = json.loads((d / "robot_view_anchors.json").read_text())
    save_labeled(d, rgb, meta, args.font_px)


def render(args):
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
        vis_options=gs.options.VisOptions(
            show_world_frame=False, ambient_light=(0.55, 0.55, 0.55), background_color=(0.93, 0.93, 0.92),
            lights=[dict(type="point", pos=(2.0, 2.0, 3.5), color=(1.0, 1.0, 1.0), intensity=14.0),
                    dict(type="point", pos=(-2.0, 1.5, 2.5), color=(1.0, 1.0, 1.0), intensity=7.0)],
        ),
    )
    scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.88, 0.88, 0.86)))
    robot = scene.add_entity(gs.morphs.URDF(file=str(ROOT / "assets/robot/house_robot.urdf"), fixed=True))
    cam = scene.add_camera(res=tuple(args.res), pos=tuple(args.pos), lookat=tuple(args.lookat), fov=args.fov, GUI=False)
    scene.build()
    dofs = [robot.get_joint(n).dofs_idx_local[0] for n in JOINT_NAMES]
    q = np.zeros(len(JOINT_NAMES))
    q[3] = args.lift
    q[4:] = STANDBY_ARM
    robot.set_dofs_position(q, dofs)
    robot.control_dofs_position(q, dofs)
    for _ in range(5):
        scene.step()

    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(cam.render()[0]).astype(np.uint8)
    Image.fromarray(rgb).save(out / "robot_view.png")
    print("saved", out / "robot_view.png")

    E, K = np.asarray(cam.extrinsics, float), np.asarray(cam.intrinsics, float)
    E2, K2 = lookat_camera(args.pos, args.lookat, args.res, args.fov)
    print("camera extrinsics/intrinsics vs look-at model: max diff", float(np.abs(E - E2).max()), float(np.abs(K - K2).max()))
    aw = anchors_world(robot)
    keys = list(LABELS)
    pts = np.array([aw[k]["p"] for k in keys] + [aw["basket_camera"]["target"]])
    uv, depth = project(E, K, pts)
    anchors = {}
    for i, k in enumerate(keys):
        anchors[k] = dict(label=LABELS[k], world=[round(float(v), 4) for v in aw[k]["p"]],
                          uv=[round(float(v), 1) for v in uv[i]], depth=round(float(depth[i]), 3),
                          virtual=bool(aw[k].get("virtual", False)), note=aw[k]["note"])
    anchors["basket_camera"]["target_world"] = [round(float(v), 4) for v in aw["basket_camera"]["target"]]
    anchors["basket_camera"]["target_uv"] = [round(float(v), 1) for v in uv[-1]]
    meta = dict(res=list(args.res), cam_pos=list(args.pos), cam_lookat=list(args.lookat), fov_deg=args.fov, lift=args.lift,
                standby_arm=STANDBY_ARM, parts=anchors, bbox=list(silhouette_bbox(rgb)), image_size=[rgb.shape[1], rgb.shape[0]])
    (out / "robot_view_anchors.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print("saved", out / "robot_view_anchors.json")

    save_labeled(out, rgb, meta, args.font_px)


# ------------------------------------------------------------------ selftest (CPU)
def selftest():
    res = (1280, 960)
    E, K = lookat_camera(CAM_POS, CAM_LOOKAT, res, CAM_FOV)
    axis_pt = np.asarray(CAM_POS) + 1.7 * (np.asarray(CAM_LOOKAT) - np.asarray(CAM_POS))
    uv, z = project(E, K, [axis_pt, CAM_LOOKAT])
    assert np.allclose(uv, [[res[0] / 2, res[1] / 2]] * 2, atol=1e-6) and z[0] > 0, uv
    # a point above the look-at point must land above the centre (v smaller), one to the robot's left (+y) ... on screen
    up_uv, _ = project(E, K, [np.asarray(CAM_LOOKAT) + [0, 0, 0.3]])
    assert up_uv[0, 1] < res[1] / 2
    # synthetic anchors: crowded on both sides of a central silhouette, one near the image border
    bbox = (480, 150, 800, 800)
    anchors = [(520, 700), (540, 690), (560, 680), (700, 200), (720, 210), (740, 220), (790, 940), (500, 20)]
    sizes = [(150, 44), (120, 44), (260, 44), (150, 44), (230, 44), (190, 44), (150, 44), (150, 44)]
    rects = layout_labels(anchors, sizes, res, bbox)
    for r in rects:
        assert r[0] >= 0 and r[1] >= 0 and r[2] <= res[0] and r[3] <= res[1], r
        assert not rects_overlap(r, (bbox[0], bbox[1], bbox[2], bbox[3])), r
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not rects_overlap(rects[i], rects[j]), (rects[i], rects[j])
    # drawing path runs on a blank image
    rgb = np.full((res[1], res[0], 3), 236, np.uint8)
    fake = {k: dict(uv=(300.0 + 120 * i, 300.0 + 60 * (i % 3))) for i, k in enumerate(LABELS)}
    fake["basket_camera"]["target_uv"] = (fake["basket_camera"]["uv"][0], fake["basket_camera"]["uv"][1] + 150)
    img, crop, cb, rr = draw_labeled(rgb, fake, bbox, font_px=46)
    assert crop.size == (cb[2] - cb[0], cb[3] - cb[1]) and crop.size[0] <= res[0]
    # text ink bbox inside its box for a 46 px font (single and two-line labels)
    font = _font(46)
    assert isinstance(font, ImageFont.FreeTypeFont), "CJK font missing"
    dr = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    for k, r in zip(LABELS, rr):
        (w, h), off, sp = text_box(dr, LABELS[k], font, 46)
        ib = dr.multiline_textbbox((r[0] + off[0], r[1] + off[1]), LABELS[k], font=font, spacing=sp)
        assert ib[0] >= r[0] + 10 and ib[1] >= r[1] + 8 and ib[2] <= r[2] - 10 and ib[3] <= r[3] - 8, (k, ib, r)
        assert r[0] >= PAD and r[1] >= PAD and r[2] <= res[0] - PAD and r[3] <= res[1] - PAD, (k, r)
    # real-looking case: widest label must switch to the side that has room, 46 px gaps >= 0.4 x font
    bb2 = (447, 237, 899, 823)
    uv2 = [(704, 737), (794, 517), (727, 392), (727, 392), (672, 543), (595, 195)]
    sz = [text_box(dr, LABELS[k], font, 46)[0] for k in LABELS]
    r2 = layout_labels(uv2, sz, res, bb2, line_gap=0.4 * 46)
    for i, r in enumerate(r2):
        assert r[0] >= PAD and r[2] <= res[0] - PAD and not rects_overlap(r, bb2), (i, r)
        for j in range(i + 1, len(r2)):
            assert not rects_overlap(r, (r2[j][0], r2[j][1] - 18, r2[j][2], r2[j][3] + 18)), (i, j)
    print("selftest ok: projection on the optical axis -> image centre; %d labels inside the image, no overlap; "
          "46 px text inside its box" % len(rects))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relabel", metavar="DIR", help="CPU only: redo robot_view_labeled(.png/_crop.png) in DIR from "
                    "DIR/robot_view.png and DIR/robot_view_anchors.json")
    ap.add_argument("--selftest", action="store_true", help="CPU-only check of projection and label layout")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--res", type=int, nargs=2, default=[1280, 960], metavar=("W", "H"))
    ap.add_argument("--lift", type=float, default=DEFAULT_LIFT, help="lift joint [m], 0..0.5")
    ap.add_argument("--pos", type=float, nargs=3, default=list(CAM_POS), metavar=("X", "Y", "Z"))
    ap.add_argument("--lookat", type=float, nargs=3, default=list(CAM_LOOKAT), metavar=("X", "Y", "Z"))
    ap.add_argument("--fov", type=float, default=CAM_FOV, help="vertical fov [deg]")
    ap.add_argument("--font-px", type=int, default=46, help="label font size [px] (46 stays readable when the crop is shrunk onto the video card)")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.relabel:
        relabel(args)
        return
    render(args)


if __name__ == "__main__":
    main()
