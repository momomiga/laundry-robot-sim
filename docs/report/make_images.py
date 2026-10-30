"""Image figures for the report (fig01-06, fig08, fig10). Run from the repo root:
    uv run --no-sync python docs/report/make_images.py
Inputs are existing simulator renders / JSON under outputs/ and tools/params.py only; no scene is re-rendered.
"""
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "tools"))

FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
INK = "#0b0b0b"
BLUE, GREY, ORANGE = "#2a78d6", "#898781", "#eb6834"
missing = []


def src(rel):
    p = ROOT / rel
    if not p.exists():
        missing.append(rel)
        return None
    return p


def tint(hex_, a=0.2):
    r, g, b = (int(hex_[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(round(255 - (255 - c) * a) for c in (r, g, b))


# ------------------------------------------------------------------ fig01
def fig01():
    from matplotlib import font_manager as fm
    fm.fontManager.addfont(FONT_REG)
    fm.fontManager.addfont(FONT_BOLD)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    # 2600 x 900 px; at 170 mm print width 1 pt of figure text = 0.77 pt printed
    W, H = 8.667, 3.0
    fig = plt.figure(figsize=(W, H), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    TS, BS = 9.7, 9.7

    def box(x, y, w, h, title, body, color, fill=None, dashed=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=fill or tint(color), ec=color, lw=1.4,
                                    ls=(0, (3, 2)) if dashed else "-"))
        ax.text(x + w / 2, y + h - 0.07, title, ha="center", va="top", fontsize=TS,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.32, body, ha="center", va="top", fontsize=BS, color=INK,
                linespacing=1.25)

    def arrow(p0, p1, color=INK, ls="-"):
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, ls=ls, shrinkA=0, shrinkB=0))

    def line(pts, color=INK, ls="-"):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=1.3, ls=ls, solid_capstyle="butt")

    by, bh = 1.23, 1.25          # main row
    gy0, gy1 = 1.15, 2.82        # group box
    GF = "#f4f4f2"
    # start box
    sx, sw = 0.06, 0.8
    box(sx, by, sw, bh, "走行・静置", "洗濯物を落\nとして静止\nさせ、洗濯\n機の前まで\n走る", GREY, fill="#ffffff")
    # group box
    gx0 = 1.08
    ax.add_patch(FancyBboxPatch((gx0, gy0), 6.51, gy1 - gy0, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=GF, ec=INK, lw=1.1))
    ax.text(gx0 + 0.15, gy1, "1 サイクル（籠が空になるまで繰り返す）", fontsize=11, fontweight="bold",
            va="center", color=INK, bbox=dict(fc="#ffffff", ec="none", pad=2.5))
    arrow((sx + sw, by + bh / 2), (gx0, by + bh / 2))
    steps = [
        ("検出", "VLM が\n洗濯物ごと\nの bbox と\nラベルを\n返す", BLUE, 0.89),
        ("切り分け", "点群と色で\n布の画素を\n選び、bbox\nごとに\nマスク", BLUE, 0.89),
        ("聞き直し", "検出に入ら\nない布色の\n領域を VLM\nに再質問", BLUE, 0.89),
        ("順番", "見える表面\nが一番高い\nものから\n（種類は\n使わない）", BLUE, 0.89),
        ("把持点・向き", "マスク内の\n点群で最も\n高い点と\nグリッパの\nヨー", BLUE, 0.89),
        ("把持・運搬・解放", "空振り・保持\nが弱い・指に\n残るの判定に\n模擬の触覚\n・力覚", ORANGE, 1.2),
    ]
    gap = 0.14
    x = gx0 + 0.08
    cxs = []
    for i, (t, b, c, w) in enumerate(steps):
        box(x, by, w, bh, t, b, c)
        if i:
            arrow((x - gap, by + bh / 2), (x, by + bh / 2))
        cxs.append((x, w))
        x += w + gap
    # loop: end of group back to the start (over the top of the boxes)
    yl = by + bh + 0.18
    xe = cxs[-1][0] + cxs[-1][1] / 2
    xs = cxs[0][0] + cxs[0][1] / 2
    line([(xe, by + bh), (xe, yl), (xs, yl)])
    arrow((xs, yl), (xs, by + bh))
    ax.text((xe + xs) / 2, yl, "観測し直す", ha="center", va="center", fontsize=BS, color=INK,
            bbox=dict(fc=GF, ec="none", pad=2.5))
    # end box
    ex, ew = 7.86, 0.75
    arrow((gx0 + 6.51, by + bh / 2), (ex, by + bh / 2))
    box(ex, by, ew, bh, "終了", "観測して\nVLM が\n何も返さ\nなければ", BLUE)

    # ground truth (scoring only), below the last step
    gtx, gtw = 4.9, 3.71
    gty, gth = 0.42, 0.62
    box(gtx, gty, gtw, gth, "真値（採点のみ）", "槽に入った割合と動画の「シミュレーション判定」", GREY, dashed=True)
    arrow((cxs[-1][0] + cxs[-1][1] / 2 + 0.3, gy0), (cxs[-1][0] + cxs[-1][1] / 2 + 0.3, gty + gth),
          color=GREY, ls=(0, (3, 2)))

    # legend, one line
    items = [
        (BLUE, tint(BLUE), "知覚のみ（画像・深度・VLM 出力）", 0.06),
        (ORANGE, tint(ORANGE), "模擬した触覚・力覚（接触・力）", 3.05),
        (GREY, tint(GREY), "真値（採点と字幕のみ）", 5.8),
        (GREY, "#ffffff", "動作のみ", 7.75),
    ]
    ly = 0.17
    for i, (ec, fc, txt, lx) in enumerate(items):
        ax.add_patch(FancyBboxPatch((lx, ly - 0.07), 0.26, 0.14, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=fc, ec=ec, lw=1.2, ls=(0, (3, 2)) if i == 2 else "-"))
        ax.text(lx + 0.34, ly, txt, fontsize=BS, va="center", color=INK)
    fig.savefig(OUT / "fig01_pipeline.png", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ fig02
def fig02():
    import params as P
    img_p, js_p = src("outputs/scene_top.png"), src("outputs/multi/r3_s0.json")
    if not (img_p and js_p):
        return
    wps = json.loads(js_p.read_text())["drive"]["waypoints_room"]
    # camera of scripts/view_scene.py: cam_top pos (1.35, 4.2, 14) lookat (1.35, 4.2, 0), up +y, fov 45 (vertical), res 800x1000.
    # Looking straight down with up = +y gives image right = +x, image up = +y.
    cx, cy, cz, fov, w, h = 1.35, 4.2, 14.0, 45.0, 800, 1000
    f = (h / 2) / math.tan(math.radians(fov / 2))

    def proj(x, y, z=0.0):
        s = f / (cz - z)
        return (w / 2 + (x - cx) * s, h / 2 - (y - cy) * s)

    im = Image.open(img_p).convert("RGB")
    d = ImageDraw.Draw(im)
    pts = [proj(*p) for p in wps]
    d.line(pts, fill=BLUE, width=2)
    font = ImageFont.truetype(FONT_BOLD, 16)

    def label(xy, text, anchor="lm"):
        bb = d.textbbox(xy, text, font=font, anchor=anchor)
        d.rectangle((bb[0] - 4, bb[1] - 3, bb[2] + 4, bb[3] + 3), fill="white", outline=GREY)
        d.text(xy, text, font=font, fill=INK, anchor=anchor)

    sx, sy = pts[0]
    ex, ey = pts[-1]
    d.ellipse((sx - 6, sy - 6, sx + 6, sy + 6), fill=BLUE, outline="white", width=2)
    d.rectangle((ex - 6, ey - 6, ex + 6, ey + 6), fill=BLUE, outline="white", width=2)
    label((sx + 56, sy), "出発")
    label((ex + 40, ey + 40), "洗濯機の前")
    im = im.crop((225, 30, 575, 970))
    im.save(OUT / "fig02_scene_top.png")
    # landmark check numbers (washer opening at the pan height, park pose on the floor)
    wp = proj(P.WASHER_POS[0], P.WASHER_POS[1], P.WASHER_H)
    print("fig02 projections (full-image px): start", [round(v) for v in pts[0]], "end/park",
          [round(v) for v in pts[-1]], "washer top", [round(v) for v in wp])


# ------------------------------------------------------------------ copies
def copy_scaled(rel, name, maxw=1600):
    p = src(rel)
    if not p:
        return
    im = Image.open(p).convert("RGB")
    if im.width > maxw:
        im = im.resize((maxw, round(im.height * maxw / im.width)), Image.LANCZOS)
    im.save(OUT / name)


def fig05():
    a, b = src("outputs/multi/r3_s0_obs_c1_rgb.png"), src("outputs/multi/r3_s0_obs_c1_seg.png")
    if not (a and b):
        return
    # the _seg.png is the simulator ground-truth seg (seg_gt) overlaid on the rgb (scripts/multi_cloth_scene.py _write_pngs),
    # not the VLM-box masks, so it is labelled as ground truth.
    ia, ib = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
    gap, bar = 12, 44
    canvas = Image.new("RGB", (ia.width + ib.width + gap, ia.height + bar), "white")
    canvas.paste(ia, (0, bar))
    canvas.paste(ib, (ia.width + gap, bar))
    d = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT_REG, 20)
    d.text((ia.width // 2, bar // 2), "(a) カメラ画像", font=font, fill=INK, anchor="mm")
    d.text((ia.width + gap + ib.width // 2, bar // 2), "(b) シミュレータの正解マスク（採点用）", font=font, fill=INK, anchor="mm")
    canvas.save(OUT / "fig06_segmentation.png")


def fig09():
    p = src("outputs/remote/integrated_semi/int_s1.mp4")
    if not p:
        return
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "f.png"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "54", "-i", str(p), "-frames:v", "1", str(f)], check=True)
        im = Image.open(f).convert("RGB").crop((600, 60, 1000, 360))
        im = im.resize((im.width * 3, im.height * 3), Image.LANCZOS)
        im.save(OUT / "fig08_release_stick.png")


if __name__ == "__main__":
    fig01()
    fig02()
    copy_scaled("outputs/corridor_view_robot.png", "fig03_corridor_robot.png")
    copy_scaled("outputs/multi/r3_s0_card_c1.png", "fig05_multi_card.png")
    copy_scaled("outputs/robot_view_labeled_crop.png", "fig10_robot_setup.png")  # from scripts/render_robot.py
    fig05()
    fig09()
    if missing:
        print("MISSING SOURCES:", missing)
        sys.exit(1)
