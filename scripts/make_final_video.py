"""Compose the submission video from one integrated-demo take (CPU only: ffmpeg + PIL, no Genesis / GPU / ollama).

  title card (~4 s) -> setup card (~4 s) -> the take (1x sim time, small corner label) -> results card (~6 s)

The take (scripts/integrated_demo.py) is recorded at 1x simulation time. While the VLM runs, the simulation is paused
and one card is held for `video_extra_card_s` seconds; the corner label switches to "VLM inference (clock stopped)"
for exactly that span. Where the span is: the JSON phase `video_t` values are the sim-frame clock (nframes / FPS, no card
time), so the card starts at video_t of the "vlm" phase and lasts video_extra_card_s; the video itself is then scanned
for the longest run of identical frames (the card) and, if it agrees within 1 s, that exact span is used.

Every number on a card comes from the take JSON or from --stats (a JSON list of {"label": ..., "value": ...}).

  uv run python scripts/make_final_video.py --take T.mp4 --json T.json [--stats S.json] [--out outputs/final/final.mp4]
  uv run python scripts/make_final_video.py --selftest        # CPU unit checks, no take video needed
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import vlm_grasp  # noqa: E402  (draw / candidate_status / load_obs / project; CPU only)

FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
JP_INDEX = 0   # "Noto Sans CJK JP" is the first face of the collection
TITLE_S, SETUP_S, RESULT_S = 4.0, 4.0, 6.0
SETUP_PARTS_S = 8.0   # the multi-garment ja setup card (robot render + price table) has more to read
MIN_S, MAX_S = 60.0, 180.0
MODEL = "qwen3-vl:8b-instruct"
DEJAVU = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]
COVER_MARGIN = 6           # px added around the old (burned-in) caption boxes before they are covered
HEUR_COLOR = (80, 200, 255)
# every English bottom caption integrated_demo.py can burn in (set_phase captions; "in tub" has a per-frame percentage)
OLD_CAPTIONS = ["towel settles in the basket", "driving through the corridor to the washer",
                "towel settles after the drive", "arm lowered into the basket", "VLM picks the grasp point",
                "pinch missed / lift failed: retry at another point", "friction grasp (IPC contact + friction only)",
                "lift the towel", "carry over the washer", "release", "retract the arm", "in tub"]
PHASES_JA = {
    "drop": "籠の中でタオルが落ち着く", "settle": "籠の中でタオルが落ち着く",
    "drive": "廊下を通って洗濯機へ走行", "settle2": "走行後にタオルが落ち着くのを待つ",
    "lower": "アームを籠へ下ろす", "grasp": "摩擦で把持（IPC の接触と摩擦のみ）",
    "lift": "タオルを持ち上げる", "carry": "洗濯機の上へ運ぶ", "release": "放す", "retract": "アームを戻す",
    "retry": "つかみ損ね・持ち上げ失敗：別の点で再試行",
    "in tub": "洗濯槽の中（最終 {pct:.0f} %）",
}
# multi-garment takes: the label already names the garment, so the action text does not
PHASES_JA_MULTI = dict(PHASES_JA, drop="籠の中で衣類が落ち着く", settle="籠の中で衣類が落ち着く",
                       settle2="走行後に衣類が落ち着くのを待つ", lift="持ち上げる", retry="つかみ損ね・持ち上げ失敗：別の点で再試行")
# English actions integrated_demo.py / multi_integrated_demo.py can burn after "[k/N] <label>: "
MULTI_LABELS_EN = ["towel", "t-shirt", "sock", "unknown"]

T = {
    "ja": dict(
        title="籠のタオルを洗濯機へ — 家の中を移動するロボット",
        sub=["Genesis 1.4.4 シミュレーション（1K の部屋を概算寸法で再現）",
             "把持点は自前ホストの VLM（Qwen3-VL 8B, ollama）が画像から選ぶ"],
        setup_title="ロボット構成",
        setup=["ホロノミック台車 + 昇降軸（0–0.5 m）+ 6 自由度アーム（PiPER）+ 籠",
               "IPC によるタオル（布）シミュレーションと摩擦把持",
               "干渉チェック付きのアーム動作計画"],
        label_1x="再生速度 1x（シミュレーション時刻）",
        label_vlm="VLM 推論中（時計停止）",
        res_title="結果",
        m_res_title="12 シードの結果",
        m_note_1x="動画はシミュレーション時刻の等倍（VLM の推論中はシミュレーションを止めてカードを表示）",
        success="成功", failure="失敗",
        l_success="タスク結果", l_attempts="把持の試行回数", l_tub="タオルが洗濯槽内にある割合",
        l_vlm="把持点の選択元", l_rank="採用した VLM 候補の順位",
        v_attempts="{n} 回", none="なし",
        src_vlm="VLM（候補 {rank}）", src_heur="重心法（代替）", top="t = {t:.1f} 秒（シミュレーション時刻）",
        card_title="VLM が把持点を選ぶ（" + MODEL + "・自前ホスト）",
        card_sub="VLM 推論中はシミュレーションを停止（{lat:.1f} 秒）",
        lg_chosen="緑 = 採用", lg_backup="黄 = 予備（再試行用）", lg_rej="赤 = 除外（理由）",
        lg_heur="水色の十字 = 重心法の点", heur_lab="重心法",
        c_chosen="候補 {rank}（全 {n} 点）を採用", c_use="把持点の探索中心として使う",
        c_backup="候補 {rank}: 予備", c_rej="候補 {rank}: 除外（{why}）", c_other="候補 {rank}: 重複のため未使用",
        c_fb="有効な点なし → 重心法で代替", c_fb2="VLM の候補はすべて除外された",
        c_fb3="VLM から候補が得られなかった（{err}）",
        why=dict(not_towel="タオルでない", wall_band="籠の壁際", no_cloth="布がない", floor="床",
                 outside_basket="籠の外", no_surface="面なし"),
        speed="計算はシミュレーション時刻の約 {x:.0f} 倍の時間（動画は等倍）",
        # --- multi-garment takes
        m_title="籠の洗濯物を 1 着ずつ洗濯機へ — 家の中を移動するロボット",
        m_sub=["① Genesis × IPC による衣類（布）の物理シミュレーション",
               "② 自前ホストの VLM（Qwen3-VL 8B）で衣類を検出し、その結果から把持を計画"],
        m_note="1K の住居モデル・すべてシミュレーション",
        m_setup=["ホロノミック台車 + 昇降軸（0–0.5 m）+ 6 自由度アーム（PiPER）+ 籠",
                 "IPC による衣類（タオル・Tシャツ・靴下）の布シミュレーションと摩擦把持",
                 "干渉チェック付きのアーム動作計画"],
        m_card_title="VLM が籠の衣類を検出する（" + MODEL + "・自前ホスト）",
        m_card_sub="VLM 推論中はシミュレーションを停止（{lat:.1f} 秒）",
        m_card_cycle="{k} 回目", m_card_left="籠の中：{n} 着（シミュレーション判定）",
        m_lg_solid="実線 = VLM の回答", m_lg_dash="破線 = 聞き直しで確認",
        m_lg_order="番号 = 掴む順（表面の高い順）", m_lg_cen="● 重心", m_lg_grasp="× 把持点",
        m_det="{o}. {name}{src}", m_det_rej="（除外：{why}）", m_src_re="（聞き直し）",
        m_cap="[{k}/{n}] VLM判定「{label}」：{action}", m_cap_plain="{action}", m_cap_2="（{n}着同時）",
        m_tub_cap="洗濯機の中：{n}/{N} 着（シミュレーション判定）",
        m_l_loaded="投入できた衣類", m_v_loaded="{n}/{N} 着",
        m_v_where=dict(tub="洗濯槽の中", basket="籠の中", floor="床の上", robot="ロボットが保持"),
        m_v_cycle="（{k} 回目）", m_l_attempts="把持の試行回数（合計）", m_v_attempts="{n} 回",
        m_l_recheck="VLM が見落として聞き直しで見つけた物", m_v_recheck="{n} 着",
        m_l_double="同時に掴んだ回", m_v_double="{k} 回目（{names}）", m_sep="・",
        m_l_vlmlab="VLM の種類ラベル", m_v_vlmlab="{a}/{b} 正解（掴む順は高さで決定）",
        m_labels=dict(towel="タオル", **{"t-shirt": "Tシャツ"}, sock="靴下", unknown="衣類"),
    ),
    "en": dict(
        title="Towel from the basket into the washer — a robot moving through the home",
        sub=["Genesis 1.4.4 simulation (a 1K apartment modelled from approximate dimensions)",
             "Grasp points are chosen from images by a self-hosted VLM (Qwen3-VL 8B, ollama)"],
        setup_title="Robot setup",
        setup=["Holonomic base + lift 0-0.5 m + 6-DoF arm (PiPER) + basket",
               "IPC cloth simulation of the towel with friction grasp",
               "Collision-checked arm motion planning"],
        label_1x="Playback 1x (simulation time)",
        label_vlm="VLM inference (clock stopped)",
        res_title="Results",
        m_res_title="Results over 12 seeds",
        m_note_1x="The video plays at 1x simulation time (the simulation is paused and a card is shown during VLM inference)",
        success="success", failure="failure",
        l_success="Task result", l_attempts="Grasp attempts", l_tub="Towel inside the tub",
        l_vlm="Grasp point source", l_rank="VLM candidate rank used",
        v_attempts="{n}", none="none",
        src_vlm="VLM (candidate {rank})", src_heur="heuristic centroid (fallback)",
        top="1x real time   t = {t:5.1f} s",
        card_title="VLM picks the grasp point (" + MODEL + ", self-hosted)",
        card_sub="Simulation paused during VLM inference ({lat:.1f} s)",
        lg_chosen="green = chosen", lg_backup="yellow = backup (for a retry)", lg_rej="red = rejected (reason)",
        lg_heur="cyan cross = heuristic point", heur_lab="heuristic",
        c_chosen="candidate {rank} of {n} chosen", c_use="used as the grasp search centre",
        c_backup="candidate {rank}: backup", c_rej="candidate {rank}: rejected ({why})",
        c_other="candidate {rank}: duplicate, unused",
        c_fb="no valid point -> heuristic centroid grasp", c_fb2="all VLM candidates were rejected",
        c_fb3="the VLM returned no candidate ({err})",
        why=dict(not_towel="not towel", wall_band="wall band", no_cloth="no cloth", floor="floor",
                 outside_basket="outside basket", no_surface="no surface"),
        speed="Compute took about {x:.0f}x the simulation time (the video is 1x)",
        m_title="Garments from the basket into the washer, one by one — a robot moving through the home",
        m_sub=["Genesis 1.4.4 simulation (a 1K apartment modelled from approximate dimensions)",
               "A self-hosted VLM (Qwen3-VL 8B, ollama) detects the garments; the grasp is planned from the detections"],
        m_setup=["Holonomic base + lift 0-0.5 m + 6-DoF arm (PiPER) + basket",
                 "IPC cloth simulation of the garments (towel, t-shirt, sock) with friction grasp",
                 "Collision-checked arm motion planning"],
        m_card_title="The VLM detects the garments in the basket (" + MODEL + ", self-hosted)",
        m_card_sub="Simulation paused during VLM inference ({lat:.1f} s)",
        m_card_cycle="Cycle {k}", m_card_left="In the basket: {n} (simulator count)",
        m_lg_solid="solid = VLM answer", m_lg_dash="dashed = found by re-asking",
        m_lg_order="number = picking order (highest surface first)", m_lg_cen="dot = centroid", m_lg_grasp="x = grasp point",
        m_det="{o}. {name}{src}", m_det_rej=" (rejected: {why})", m_src_re=" (re-asked)",
        m_cap="[{k}/{n}] VLM: \"{label}\" — {action}", m_cap_plain="{action}", m_cap_2=" ({n} at once)",
        m_tub_cap="in washer {n}/{N} (sim)",
        m_l_loaded="Garments loaded", m_v_loaded="{n}/{N}",
        m_v_where=dict(tub="in the tub", basket="in the basket", floor="on the floor", robot="held by the robot"),
        m_v_cycle=" (cycle {k})", m_l_attempts="Grasp attempts (total)", m_v_attempts="{n}",
        m_l_recheck="Found only by re-asking the VLM", m_v_recheck="{n}",
        m_l_double="Grasped two at once", m_v_double="cycle {k} ({names})", m_sep=" + ",
        m_l_vlmlab="VLM type labels", m_v_vlmlab="{a}/{b} correct (picking order set by height)",
        m_labels=dict(towel="towel", **{"t-shirt": "t-shirt"}, sock="sock", unknown="garment"),
    ),
}


# ------------------------------------------------------------------ fonts / drawing
def check_fonts():
    for f in (FONT_REG, FONT_BOLD):
        if not os.path.exists(f):
            raise SystemExit(f"ERROR: font not found: {f} (install fonts-noto-cjk)")


def _font(px, bold=False):
    from PIL import ImageFont
    check_fonts()
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, px, index=JP_INDEX)


def _wrap(d, text, font, maxw):
    """Greedy character wrap (works for CJK without spaces)."""
    lines, cur = [], ""
    for ch in text:
        if cur and d.textlength(cur + ch, font=font) > maxw:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    return lines + [cur]


def _canvas(size):
    from PIL import Image
    w, h = size
    arr = np.zeros((h, w, 3), np.uint8)
    arr[:] = (22, 24, 30)
    arr[:, :, 2] = np.linspace(30, 46, h)[:, None]
    return Image.fromarray(arr)


def card_text(size, title, lines, title_px=None, line_px=None, left=False, note=None):
    """Dark card: a bold title on top, `lines` below (wrapped, vertically centred as a block)."""
    from PIL import ImageDraw
    w, h = size
    im = _canvas(size)
    d = ImageDraw.Draw(im)
    ft = _font(title_px or h // 14, True)
    fl = _font(line_px or h // 22)
    maxw = int(w * 0.84)
    parts = title.split(" — ")   # break at the dash first (the dash stays at the end of line 1)
    parts = [p + " —" for p in parts[:-1]] + parts[-1:]
    tl = [t for part in parts for t in _wrap(d, part, ft, maxw)]
    body = []
    for ln in lines:
        body.append(_wrap(d, ln, fl, maxw))
    th = int(ft.size * 1.35)
    lh = int(fl.size * 1.5)
    total = len(tl) * th + int(ft.size * 0.8) + sum(len(b) for b in body) * lh + (len(body) - 1) * lh // 3
    fn = _font(int(fl.size * 0.75)) if note else None
    nh = int(fn.size * 1.5) if note else 0
    if note:
        total += lh // 2 + nh
    y = (h - total) // 2
    for t in tl:
        d.text(((w - d.textlength(t, font=ft)) / 2, y), t, font=ft, fill=(255, 255, 255))
        y += th
    y += int(ft.size * 0.8)
    for b in body:
        for t in b:
            x = int(w * 0.08) if left else (w - d.textlength(t, font=fl)) / 2
            d.text((x, y), t, font=fl, fill=(205, 215, 230))
            y += lh
        y += lh // 3
    if note:
        y += lh // 2 - lh // 3
        d.text(((w - d.textlength(note, font=fn)) / 2, y), note, font=fn, fill=(140, 150, 165))
    return im


def round_half_up(x):
    return int(math.floor(x + 0.5))


def parts_summary(data):
    """-> ([(label, '$1,999'), ...], total line). Total = rounded sum in USD and in 10,000-yen units."""
    rows = [(p["label"], f"${round_half_up(p['price_usd']):,}") for p in data["parts"]]
    tot = sum(p["price_usd"] for p in data["parts"])
    man = round_half_up(tot * data["usd_jpy"] / 10000)
    return rows, f"合計 約 ${round_half_up(tot):,}（約 {man} 万円）"


def _wrap_words(d, text, font, maxw):
    """Like _wrap, but keeps ASCII runs with their unit ('15 kg', '648 mm') and closing brackets unbroken."""
    import re
    toks = re.findall(r"[\x21-\x7e]+(?: [\x21-\x7e]+)*[）」、。]*|[^\x21-\x7e][）」、。]*", text)
    lines, cur = [], ""
    for t in toks:
        if cur and d.textlength(cur + t, font=font) > maxw:
            lines.append(cur)
            cur = t.lstrip()
        else:
            cur += t
    return lines + [cur]


def card_setup_parts(size, data, robot_img):
    """Setup card of the multi ja video: robot render on the left, parts + reference prices on the right."""
    from PIL import Image, ImageDraw
    w, h = size
    im = _canvas(size)
    d = ImageDraw.Draw(im)
    ft = _font(h // 14, True)
    d.text(((w - d.textlength(data["title"], font=ft)) / 2, int(h * 0.04)), data["title"], font=ft, fill=(255, 255, 255))
    top = int(h * 0.04) + int(ft.size * 1.45)
    fnt = _font(max(22, h // 31))
    foot = _wrap(d, data["footnote"], fnt, int(w * 0.94))
    flh = int(fnt.size * 1.4)
    foot_y = h - int(h * 0.03) - len(foot) * flh
    # robot image (left), light rounded panel as a border
    mx = int(w * 0.03)
    iw = int(w * 0.46)
    ri = Image.open(robot_img).convert("RGB")
    ih = int(round(iw * ri.height / ri.width))
    ri = ri.resize((iw, ih), Image.LANCZOS)
    pad = 6
    iy = top + max(0, (foot_y - top - ih) // 2 - pad)
    d.rounded_rectangle((mx - pad, iy - pad, mx + iw + pad, iy + ih + pad), radius=14, fill=(225, 230, 238))
    im.paste(ri, (mx, iy))
    # price rows (right)
    xl = mx + iw + int(w * 0.035)
    xr = w - mx
    fl = _font(max(22, h // 29))
    fb = _font(max(22, h // 29), True)
    rows, total_line = parts_summary(data)
    pw = int(max(fb.getlength(v) for _, v in rows)) + 12
    wrapped = [_wrap_words(d, lab, fl, xr - xl - pw - 8) for lab, _ in rows]
    lh = int(fl.size * 1.35)
    gap = int(fl.size * 0.6)
    block = sum(len(x) * lh + gap for x in wrapped) + int(fl.size * 0.4) + int(lh * 1.3)
    y = top + max(0, (foot_y - top - block) // 2)
    for lines, (_, val) in zip(wrapped, rows):
        for i, t in enumerate(lines):
            d.text((xl, y + i * lh), t, font=fl, fill=(205, 215, 230))
        d.text((xr - fb.getlength(val), y), val, font=fb, fill=(255, 255, 255))
        y += len(lines) * lh + gap
    y += int(fl.size * 0.1)
    d.line((xl, y, xr, y), fill=(150, 160, 175), width=2)
    y += int(fl.size * 0.3)
    d.text((xr - fb.getlength(total_line), y), total_line, font=fb, fill=(255, 255, 255))
    for i, t in enumerate(foot):
        d.text((int(w * 0.03), foot_y + i * flh), t, font=fnt, fill=(150, 160, 175))
    return im


def results_layout(size, rows, speed_line, tx=None):
    """Geometry of the results card: fonts + row pitch shrink so that up to 10 rows and the speed line always fit.
    -> dict(y0, lh, px, xl, xv, speed_lines, speed_y, fs_px, bottom)."""
    from PIL import Image, ImageDraw
    w, h = size
    d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    xl, xv = int(w * 0.07), int(w * 0.07) + int(w * 0.52)
    fs = _font(h // 29)
    sl = _wrap(d, speed_line, fs, int(w * 0.86))
    speed_h = int(len(sl) * fs.size * 1.5)
    speed_y = h - int(h * 0.05) - speed_h
    y0 = int(h * 0.20) if len(rows) > 6 else int(h * 0.24)
    n = max(len(rows), 1)
    lh = min(int(h // 24 * 1.9), (speed_y - int(h * 0.02) - y0) // n)
    px = max(12, min(h // 24, int(lh * 0.62)))
    # shrink until every label / value fits its column
    while px > 12 and any(_font(px).getlength(a) > xv - xl - 10 or _font(px, True).getlength(b) > w - xv - int(w * 0.03)
                          for a, b in rows):
        px -= 1
    return dict(y0=y0, lh=lh, px=px, xl=xl, xv=xv, speed_lines=sl, speed_y=speed_y, fs_px=fs.size,
                bottom=max(y0 + n * lh, speed_y + speed_h))


def card_results(size, lang, rows, speed_line, title=None):
    """Results card: label / value rows in two columns, the compute-speed line at the bottom."""
    from PIL import ImageDraw
    w, h = size
    tx = T[lang]
    im = _canvas(size)
    d = ImageDraw.Draw(im)
    ft = _font(h // 14, True)
    ty = int(h * (0.05 if len(rows) > 6 else 0.08))
    title = title or tx["res_title"]
    d.text(((w - d.textlength(title, font=ft)) / 2, ty), title, font=ft, fill=(255, 255, 255))
    g = results_layout(size, rows, speed_line)
    fl, fv = _font(g["px"]), _font(g["px"], True)
    for i, (lab, val) in enumerate(rows):
        y = g["y0"] + i * g["lh"]
        d.text((g["xl"], y), lab, font=fl, fill=(190, 200, 215))
        d.text((g["xv"], y), val, font=fv, fill=(255, 255, 255))
    fs = _font(g["fs_px"])
    for j, t in enumerate(g["speed_lines"]):
        d.text((g["xl"], g["speed_y"] + j * int(fs.size * 1.5)), t, font=fs, fill=(255, 215, 120))
    return im


def label_png(size, text, path):
    """Transparent PNG (full frame) with a small top-right label for the overlay filter."""
    from PIL import Image, ImageDraw
    w, h = size
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    px = max(14, h // 30)
    f = _font(px)
    bb = d.textbbox((0, 0), text, font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = px // 2
    x0, y0 = w - tw - 2 * pad, pad
    d.rectangle([x0 - pad, y0 - pad // 2, x0 + tw + pad, y0 + th + pad + pad // 2], fill=(0, 0, 0, 150))
    d.text((x0, y0 - bb[1]), text, font=f, fill=(255, 255, 255, 255))
    im.save(path)


# ------------------------------------------------------------------ data
def results_rows(res, stats, lang):
    """(rows, speed_line) from the take JSON (+ --stats); raises KeyError if a needed field is missing."""
    tx = T[lang]
    va = (res.get("vlm_attempts") or [{}])
    last = va[-1]
    src = last.get("source", "?")
    rank = last.get("rank")
    rows = [
        (tx["l_success"], tx["success"] if res["success"] else tx["failure"]),
        (tx["l_attempts"], tx["v_attempts"].format(n=res["attempts"])),
        (tx["l_tub"], f"{100.0 * res['in_tub_frac_end']:.0f} %"),
        (tx["l_vlm"], tx["src_vlm"].format(rank=rank) if src == "vlm" and rank is not None
         else tx["src_heur"] if src == "heuristic" else str(src)),
    ]
    for s in stats or []:
        rows.append((str(s["label"]), str(s["value"])))
    return rows, tx["speed"].format(x=slowdown(res))


def is_multi(res):
    return bool(res.get("cards"))


def norm_label(name):
    """towel / t-shirt / sock / unknown from a free VLM label or object name."""
    n = str(name or "unknown").lower().replace("_", "-").replace(" ", "-")
    n = {"tshirt": "t-shirt", "shirt": "t-shirt", "t-shirts": "t-shirt", "socks": "sock", "towels": "towel"}.get(n, n)
    return n if n in ("towel", "t-shirt", "sock") else "unknown"


def label_ja(name, lang="ja"):
    return T[lang]["m_labels"][norm_label(name)]


def n_garments(res):
    if res.get("objects_final"):
        return len(res["objects_final"])
    return max(len(res.get("cycles") or []), 1)


def get_cycle(res, k):
    cs = res.get("cycles") or []
    if isinstance(cs, dict):
        return cs.get(str(k)) or cs.get(k)
    for c in cs:
        if isinstance(c, dict) and c.get("cycle") == k:
            return c
    return cs[k - 1] if 1 <= k <= len(cs) and isinstance(cs[k - 1], dict) else None


def n_in_tub_at(res, k=None):
    """Latest n_in_tub_after known for cycle <= k (k None: the latest of all cycles); 0 before the first."""
    n = 0
    top = max((len(res.get("cycles") or []), *(c["cycle"] for c in res.get("cards", []))))
    for j in range(1, (top if k is None else k) + 1):
        c = get_cycle(res, j)
        if c and c.get("n_in_tub_after") is not None:
            n = int(c["n_in_tub_after"])
    return n


def _attempts(c):
    a = c.get("attempts") if c else None
    return len(a) if isinstance(a, (list, tuple)) else int(a or 0)


def resolve_path(p, base_dirs):
    """A path from the take JSON: absolute, or relative to the JSON's directory / the repo root / the cwd, or (a take
    collected from the remote keeps the remote's repo-relative paths) its file name next to the JSON."""
    q = Path(p)
    if q.is_absolute():
        return q
    for b in [*base_dirs, ROOT, Path.cwd()]:
        if b and (Path(b) / q).exists():
            return Path(b) / q
    for b in base_dirs:
        if b and (Path(b) / q.name).exists():
            return Path(b) / q.name
    return Path(base_dirs[0] if base_dirs else ".") / q


def load_vlm_json(card, base_dirs):
    path = resolve_path(card["vlm_json"], base_dirs)
    if not path.exists():
        raise SystemExit(f"ERROR: card {card.get('cycle')}: vlm_json not found: {path}")
    return json.load(open(path))


def recheck_labels(vlms):
    """Distinct garment labels the VLM missed at first and that a re-ask confirmed (status ok, source recheck)."""
    return sorted({norm_label(d.get("label")) for v in vlms for d in v.get("detections", [])
                   if d.get("source") == "recheck" and d.get("status") == "ok"})


def recheck_count(res, vlms):
    """Distinct ground-truth garments (evaluation gt_name) matched to a recheck-ok detection over all cycles; without
    any evaluation, the recheck-ok detections of the first cycle."""
    names, have_eval = set(), False
    for v in vlms:
        ev = v.get("evaluation")
        if not ev:
            continue
        have_eval = True
        dets = v.get("detections") or []
        for o in ev.get("objects") or []:
            i = o.get("det_index")
            if isinstance(i, int) and 0 <= i < len(dets) and dets[i].get("source") == "recheck" \
                    and dets[i].get("status", "ok") == "ok":
                names.add(o.get("gt_name"))
    if have_eval:
        return len(names)
    return sum(1 for d in (vlms[0].get("detections") or []) if d.get("source") == "recheck"
               and d.get("status", "ok") == "ok") if vlms else 0


def double_grasp_cycles(res):
    """[(cycle, [sim names lifted together])] for cycles whose last grasp attempt lifted two garments at once."""
    out = []
    for i, c in enumerate(res.get("cycles") or []):
        at = c.get("attempts")
        last = at[-1] if isinstance(at, list) and at and isinstance(at[-1], dict) else None
        if last and last.get("double_grasp"):
            out.append((c.get("cycle", i + 1), list(last.get("lifted_objects") or c.get("lifted_objects") or [])))
    return out


def is_double_cycle(res, k):
    return any(c == k for c, _ in double_grasp_cycles(res))


def n_carried(res, k):
    """Garments of cycle k that were lifted and at least half over the tub opening at the release (sim truth): a
    second garment lifted with the target can fall back into the basket before the carry ends."""
    c = next((c for i, c in enumerate(res.get("cycles") or []) if c.get("cycle", i + 1) == k), None)
    per = ((c or {}).get("objects") or {}).get("per_object") or []
    return sum(1 for o in per if o.get("lifted") and (o.get("over_opening_frac") or 0.0) >= 0.5)


MAX_ROWS = 10


def results_rows_multi(res, stats, lang, vlms):
    """Multi-garment results card: only the --stats rows (multi-seed statistics) + the fixed 1x note."""
    tx = T[lang]
    rows = [(str(x["label"]), str(x["value"])) for x in stats or []]
    if len(rows) > MAX_ROWS:
        print(f"WARN: results card holds {MAX_ROWS} rows: dropping {len(rows) - MAX_ROWS} --stats row(s)")
        rows = rows[:MAX_ROWS]
    return rows, tx["m_note_1x"]


def sim_dt(res):
    """Sim step from the drive block (duration_s / steps); no value is assumed."""
    dr = res["drive"]
    return dr["duration_s"] / dr["steps"]


def slowdown(res):
    """Wall seconds per sim second = 1 / (wall steps per s * dt); incl-render rate when present."""
    sps = res.get("wall_steps_per_s_incl_render") or res["sim_steps_per_s"]
    return 1.0 / (sps * sim_dt(res))


def pause_span(res):
    """(start, end) of the paused VLM card in take-video seconds, from the JSON only."""
    t0 = next(p["video_t"] for p in res["phases"] if p["phase"] == "vlm")
    return t0, t0 + res["video_extra_card_s"]


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    a, b = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(a) / float(b), float(j["format"]["duration"])


def still_runs(take, fps, w=64, h=36, min_s=2.0):
    """Every run of identical frames (held cards) of at least `min_s` -> [(start_s, end_s)] in time order."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(take), "-vf", f"scale={w}:{h},format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, h * w).astype(np.int16)
    same = np.abs(np.diff(fr, axis=0)).max(axis=1) <= 2
    runs, cur, start = [], 0, 0
    for i, s_ in enumerate(list(same) + [False]):
        if s_:
            if cur == 0:
                start = i
            cur += 1
        else:
            if cur >= int(min_s * fps):
                runs.append((start / fps, (start + cur + 1) / fps, cur))
            cur = 0
    return [(a, b) for a, b, _ in runs], [c for _, _, c in runs]


def detect_pause(take, fps, w=64, h=36):
    """Longest run of identical frames in the take (the held card) -> (start_s, end_s) or None."""
    runs, lens = still_runs(take, fps, w, h, min_s=2.0)
    if not runs:
        return None
    return runs[int(np.argmax(lens))]   # argmax = the earliest of equally long runs


def card_spans(res):
    """Multi takes: [(frame0, frame1)] from res["cards"] (authoritative, end exclusive), sorted, validated."""
    cs = sorted(res["cards"], key=lambda c: c["frame0"])
    prev = 0
    for c in cs:
        if not 0 <= c["frame0"] < c["frame1"] or c["frame0"] < prev:
            raise SystemExit(f"ERROR: bad card frame range in the JSON: {c}")
        prev = c["frame1"]
    return [(int(c["frame0"]), int(c["frame1"])) for c in cs], cs


def check_cards_against_scan(take, spans, fps, tol=0.5):
    """Warn (never fail) where a JSON card range disagrees with the still-frame scan by more than `tol` seconds."""
    runs, _ = still_runs(take, fps, min_s=1.5)
    nwarn = 0
    for k, (f0, f1) in enumerate(spans, 1):
        a, b = f0 / fps, f1 / fps
        best = min(runs, key=lambda r: abs(r[0] - a) + abs(r[1] - b), default=None)
        if best is None or abs(best[0] - a) > tol or abs(best[1] - b) > tol:
            print(f"WARN: card {k}: JSON frames {f0}-{f1} ({a:.2f}-{b:.2f} s) vs frame scan {best}: differ by > {tol} s;"
                  " using the JSON range")
            nwarn += 1
    print(f"card ranges: {len(spans)} from the JSON, {len(runs)} still runs found, {nwarn} disagreement warning(s)")
    return nwarn


def locate_pause(take, res, fps):
    j0, j1 = pause_span(res)
    det = detect_pause(take, fps)
    if det and abs(det[0] - j0) < 1.0 and abs(det[1] - j1) < 1.0:
        return det, "frame scan (agrees with JSON within 1 s)"
    print(f"WARN: frame scan {det} does not match the JSON estimate {(j0, j1)}; using the JSON estimate")
    return (j0, j1), "JSON estimate (phase video_t + video_extra_card_s)"


# ------------------------------------------------------------------ VLM card (regenerated)
def vlm_card(size, lang, obs, vlm):
    """Card frame for the paused simulation, built from the observation + the take JSON `vlm` block."""
    from PIL import Image, ImageDraw
    w, h = size
    tx = T[lang]
    tmp = Path(tempfile.mkdtemp(prefix="vlm_card_")) / "ann.png"
    vlm_grasp.draw(obs, vlm, tmp)
    ann = Image.open(tmp).convert("RGB")
    cands = vlm_grasp.candidate_status(vlm)
    valid = vlm.get("valid_candidates") or []
    fb = bool(vlm.get("fallback")) or not valid
    heur = vlm.get("heuristic_point")
    if fb and heur is not None:
        uv, _ = vlm_grasp.project(obs, np.array([heur], float))
        u, v = float(uv[0, 0]), float(uv[0, 1])
        dd = ImageDraw.Draw(ann)
        dd.line([u - 12, v, u + 12, v], fill=HEUR_COLOR, width=3)
        dd.line([u, v - 12, u, v + 12], fill=HEUR_COLOR, width=3)
        dd.text((u + 14, v - 22), tx["heur_lab"], font=_font(18, True), fill=HEUR_COLOR)
    k = 0.62 * h / ann.height
    ann = ann.resize((int(ann.width * k), int(ann.height * k)))
    im = _canvas(size)
    d = ImageDraw.Draw(im)
    ft, fs, fl, fb_ = _font(h // 22, True), _font(h // 26), _font(h // 28), _font(h // 28, True)
    title = tx["card_title"]
    d.text(((w - d.textlength(title, font=ft)) / 2, int(h * 0.09)), title, font=ft, fill=(255, 255, 255))
    ix, iy = int(w * 0.04), int(h * 0.20)
    im.paste(ann, (ix, iy))
    d.rectangle([ix - 1, iy - 1, ix + ann.width, iy + ann.height], outline=(90, 100, 120))
    x, y = ix + ann.width + int(w * 0.035), iy
    maxw = w - x - int(w * 0.03)
    lh = int(fl.size * 1.5)
    items = [(vlm_grasp.COLORS["chosen"], tx["lg_chosen"]), (vlm_grasp.COLORS["backup"], tx["lg_backup"]),
             (vlm_grasp.COLORS["rejected"], tx["lg_rej"])]
    if fb and heur is not None:
        items.append((HEUR_COLOR, tx["lg_heur"]))
    for col, t in items:
        d.ellipse([x, y + lh // 2 - 8, x + 16, y + lh // 2 + 8], outline=col, width=3)
        d.text((x + 28, y), t, font=fl, fill=(230, 235, 245))
        y += lh
    y += lh // 2
    lines = []   # (text, colour, bold)
    if fb:
        lines.append((tx["c_fb"], (255, 215, 120), True))
        if vlm.get("pixels"):
            lines.append((tx["c_fb2"], (205, 215, 230), False))
        else:
            lines.append((tx["c_fb3"].format(err=vlm.get("error") or "?"), (205, 215, 230), False))
    else:
        lines.append((tx["c_chosen"].format(rank=vlm["rank"], n=len(vlm["pixels"])), vlm_grasp.COLORS["chosen"], True))
        lines.append((tx["c_use"], (205, 215, 230), False))
    why = {r["rank"]: r.get("reason", "") for r in vlm.get("rejections") or []}
    for r, _, st in cands:
        if st == "backup":
            lines.append((tx["c_backup"].format(rank=r), vlm_grasp.COLORS["backup"], False))
        elif st == "rejected":
            lines.append((tx["c_rej"].format(rank=r, why=tx["why"].get(why.get(r, ""), why.get(r) or "?")),
                          vlm_grasp.COLORS["rejected"], False))
        elif st == "other":
            lines.append((tx["c_other"].format(rank=r), vlm_grasp.COLORS["other"], False))
    for t, col, bold in lines:
        for seg in _wrap(d, t, fb_ if bold else fl, maxw):
            d.text((x, y), seg, font=fb_ if bold else fl, fill=col)
            y += lh
    sub = tx["card_sub"].format(lat=float(vlm.get("latency") or 0.0))
    d.text(((w - d.textlength(sub, font=fs)) / 2, h - int(h * 0.1)), sub, font=fs, fill=(205, 215, 230))
    return im


def _fit_font(d, text, px, maxw, bold=False, lo=12):
    while px > lo and d.textlength(text, font=_font(px, bold)) > maxw:
        px -= 1
    return _font(px, bold)


def _dashed_rect(dr, box, col, width=3, on=6, off=6):
    x1, y1, x2, y2 = box
    for a in np.arange(x1, x2, on + off):
        for yy in (y1, y2):
            dr.line([a, yy, min(a + on, x2), yy], fill=col, width=width)
    for b in np.arange(y1, y2, on + off):
        for xx in (x1, x2):
            dr.line([xx, b, xx, min(b + on, y2)], fill=col, width=width)


def _place_label(dr, font, txt, box, bounds, placed):
    """Label position (x, y, rect) for a box: above its top-left, inside its top-left, below its bottom-left, then
    shifted down by the label height until it intersects no earlier label and stays inside the image."""
    W, H = bounds
    bb = dr.textbbox((0, 0), txt, font=font)
    lw, lh = bb[2] - bb[0] + 4, bb[3] - bb[1] + 4
    x = min(max(box[0], 0), max(W - lw - 2, 0))

    def rect_at(y):
        r = dr.textbbox((x + 2, y), txt, font=font)
        return r

    def free(r):
        return r[0] >= 0 and r[1] >= 0 and r[2] <= W and r[3] <= H and not any(
            r[0] < q[2] and q[0] < r[2] and r[1] < q[3] and q[1] < r[3] for q in placed)

    for y in (box[1] - lh - 2, box[1] + 2, box[3] + 2):
        if free(rect_at(y)):
            return x, y, rect_at(y)
    y = box[3] + 2
    while y + lh <= H:
        y += lh
        if free(rect_at(y)):
            return x, y, rect_at(y)
    y = max(min(box[1], H - lh), 0)
    return x, y, rect_at(y)


def vlm_card_multi(size, lang, obs, vlm, k, n_left, pause_s):
    """Card for one cycle of a multi take: the observation with the VLM boxes (solid = answer, dashed = re-ask),
    pick-order numbers, centroid dots and grasp crosses, a legend, and the "k 回目" / basket-count heading."""
    from PIL import Image, ImageDraw
    w, h = size
    tx = T[lang]
    try:
        from vlm_multi import PALETTE
    except Exception:   # noqa: BLE001
        PALETTE = [(255, 64, 64), (64, 220, 255), (255, 220, 40), (190, 90, 255), (70, 255, 110), (255, 140, 0)]
    src = Image.fromarray(np.asarray(obs["rgb"], np.uint8)).convert("RGB")
    sc = 0.62 * h / src.height
    ann = src.resize((max(1, int(src.width * sc)), max(1, int(src.height * sc))))
    dr = ImageDraw.Draw(ann)
    f = _font(max(14, ann.height // 18), True)
    dets = vlm.get("detections") or []
    placed = []
    for i, d in enumerate(dets):
        ok = d.get("status", "ok") == "ok"
        col = PALETTE[i % len(PALETTE)] if ok else (150, 150, 150)
        box = [float(v) * sc for v in d["box"]]
        if d.get("source") == "recheck":
            _dashed_rect(dr, box, col)
        else:
            dr.rectangle(box, outline=col, width=3)
        txt = (f"{d['order']} " if ok and d.get("order") else "") + label_ja(d.get("label"), lang)
        x, y, rect = _place_label(dr, f, txt, box, ann.size, placed)
        placed.append(rect)
        dr.rectangle(rect, fill=(0, 0, 0))
        dr.text((x + 2, y), txt, font=f, fill=col)
        if ok and d.get("centroid_px"):
            u, v = (float(c) * sc for c in d["centroid_px"])
            dr.ellipse([u - 5, v - 5, u + 5, v + 5], fill=col, outline=(0, 0, 0), width=2)
        if ok and d.get("grasp_px"):
            gu, gv = (float(c) * sc for c in d["grasp_px"])
            for wd, c in ((6, (0, 0, 0)), (3, col)):
                dr.line([gu - 9, gv - 9, gu + 9, gv + 9], fill=c, width=wd)
                dr.line([gu - 9, gv + 9, gu + 9, gv - 9], fill=c, width=wd)
    im = _canvas(size)
    d = ImageDraw.Draw(im)
    title = tx["m_card_title"]
    ft = _fit_font(d, title, h // 22, int(w * 0.94), True)
    d.text(((w - d.textlength(title, font=ft)) / 2, int(h * 0.07)), title, font=ft, fill=(255, 255, 255))
    ix, iy = int(w * 0.04), int(h * 0.20)
    im.paste(ann, (ix, iy))
    d.rectangle([ix - 1, iy - 1, ix + ann.width, iy + ann.height], outline=(90, 100, 120))
    x, y = ix + ann.width + int(w * 0.035), iy
    maxw = w - x - int(w * 0.02)
    fl, fb_ = _font(h // 30), _font(h // 28, True)
    lh = int(fl.size * 1.5)
    head = tx["m_card_cycle"].format(k=k)
    d.text((x, y), head, font=_fit_font(d, head, h // 28, maxw, True), fill=(255, 215, 120))
    y += int(lh * 1.1)
    head2 = tx["m_card_left"].format(n=n_left)
    d.text((x, y), head2, font=_fit_font(d, head2, int(h / 36), maxw), fill=(255, 215, 120))
    y += int(lh * 1.3)
    for t in (tx["m_lg_solid"], tx["m_lg_dash"], tx["m_lg_order"], tx["m_lg_cen"], tx["m_lg_grasp"]):
        d.text((x, y), t, font=_fit_font(d, t, fl.size, maxw), fill=(230, 235, 245))
        y += lh
    y += lh // 2
    order = sorted(range(len(dets)), key=lambda i: (dets[i].get("order") or 99, i))
    for i in order:
        dd = dets[i]
        ok = dd.get("status", "ok") == "ok"
        t = tx["m_det"].format(o=dd.get("order") or "-", name=label_ja(dd.get("label"), lang),
                               src=tx["m_src_re"] if dd.get("source") == "recheck" else "")
        if not ok:
            t += tx["m_det_rej"].format(why=dd.get("reason") or "?")
        col = PALETTE[i % len(PALETTE)] if ok else (150, 150, 150)
        for seg in _wrap(d, t, fl, maxw):
            d.text((x, y), seg, font=fl, fill=col)
            y += lh
    sub = tx["m_card_sub"].format(lat=pause_s)
    fs = _font(h // 26)
    d.text(((w - d.textlength(sub, font=fs)) / 2, h - int(h * 0.1)), sub, font=fs, fill=(205, 215, 230))
    return im


# ------------------------------------------------------------------ burned-in caption cover (ja)
def _dejavu(px):
    from PIL import ImageFont
    for f in DEJAVU:
        if os.path.exists(f):
            return ImageFont.truetype(f, px)
    raise SystemExit("ERROR: DejaVuSans not found (needed to size the old caption boxes)")


def old_box(size, text, anchor, font=None):
    """Rectangle (x0, y0, x1, y1) integrated_demo.overlay() draws for `text` (re-derived from its code)."""
    from PIL import Image, ImageDraw
    w, h = size
    px = max(14, h // 26)
    f = font or _dejavu(px)
    d = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    bb = d.textbbox((0, 0), text, font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = px // 2
    x0 = pad if anchor == "top" else (w - tw) // 2
    y0 = pad if anchor == "top" else h - th - 3 * pad
    return x0 - pad, y0 - pad // 2, x0 + tw + pad, y0 + th + pad + pad // 2


def old_texts(max_sim=400.0):
    top = [T["en"]["top"].format(t=i / 10.0) for i in range(int(max_sim * 10) + 1)]
    bottom = OLD_CAPTIONS + [f"in tub: {p:.0f} %" for p in range(0, 101)]
    return top, bottom


def old_texts_multi():
    """Bottom captions a multi take can burn: "[k/N] <label>: <action>" (digits are equally wide in DejaVuSans, so
    k = N = 1..9 covers every width) and the in-tub view "in washer n/N (sim)" (with and without the prefix)."""
    acts = [a for a in OLD_CAPTIONS if a != "in tub"]
    out = [f"[{k}/{k}] {lab}: {a}" for k in range(1, 10) for lab in MULTI_LABELS_EN for a in acts]
    tubs = [f"in washer {n}/{N} (sim)" for N in range(1, 10) for n in range(0, N + 1)]
    out += tubs + [f"[{k}/{k}] {lab}: {t}" for k in range(1, 10) for lab in MULTI_LABELS_EN for t in tubs[:12]]
    return out


def cover_rects(size, max_sim=400.0, multi=False):
    """Union of every old box (+ margin), per anchor -> {"top": rect, "bottom": rect}; a text change never leaks."""
    w, h = size
    top, bottom = old_texts(max_sim)
    if multi:
        bottom = bottom + old_texts_multi()
    out = {}
    for anchor, texts in (("top", top), ("bottom", bottom)):
        rs = [old_box(size, t, anchor) for t in texts]
        m = COVER_MARGIN
        out[anchor] = (max(0, min(r[0] for r in rs) - m), max(0, min(r[1] for r in rs) - m),
                       min(w, max(r[2] for r in rs) + m), min(h, max(r[3] for r in rs) + m))
    return out


def caption_patch(size, text, anchor, cover):
    """Opaque RGB patch (np array) + its (x, y) for a Japanese caption: at least the cover, grown to fit the text."""
    from PIL import Image, ImageDraw
    w, h = size
    px = max(14, h // 26)
    f = _font(px)
    d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    bb = d.textbbox((0, 0), text, font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = px // 2
    while tw + 2 * pad > w and px > 10:   # a caption wider than the frame is shrunk to fit
        px -= 1
        f = _font(px)
        bb = d.textbbox((0, 0), text, font=f)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = px // 2
    x0, y0, x1, y1 = cover
    bw, bh = max(x1 - x0, tw + 2 * pad), max(y1 - y0, th + pad)
    if anchor == "top":
        bx, by = x0, y0
    else:
        bx, by = (w - bw) // 2, y1 - bh
    bx, by = max(0, bx), max(0, by)
    bw, bh = min(bw, w - bx), min(bh, h - by)
    im = Image.new("RGB", (bw, bh), (12, 12, 16))
    ImageDraw.Draw(im).text(((bw - tw) // 2 if anchor == "bottom" else pad + COVER_MARGIN, (bh - th) // 2 - bb[1]),
                            text, font=f, fill=(255, 255, 255))
    return np.asarray(im), bx, by


# ------------------------------------------------------------------ timeline
def sim_time(idx, fps, i0, i1):
    """Simulation time of take frame `idx`: the card frames [i0, i1) do not advance the clock."""
    return (idx if idx < i0 else idx - (i1 - i0)) / fps


def phase_at(phases, s):
    """Phase active at sim-frame clock `s` (last phase whose video_t <= s; list order breaks ties)."""
    cur = None
    for p in phases:
        if p["video_t"] <= s + 1e-6:
            cur = p["phase"]
    return cur


def caption_ja(phase, res):
    if phase == "vlm":
        return PHASES_JA["lower"]
    if phase not in PHASES_JA:
        raise ValueError(f"unknown phase {phase!r}: no Japanese caption")
    return PHASES_JA[phase].format(pct=100.0 * res.get("in_tub_frac_end", 0.0))


def phase_obj_at(phases, s):
    """Like phase_at but returns the whole phase dict (cycle / target included)."""
    cur = None
    for p in phases:
        if p["video_t"] <= s + 1e-6:
            cur = p
    return cur


# Sim seconds of each release_check step, mirrored from multi_integrated_demo.py (not imported: it pulls in genesis):
# wiggle_wrist() = three 0.4 s plan_exec moves + hold(1.0) = 2.2 s; the second round's p_grip("release_open_max", ..., 0.6).
WIGGLE_STEP_S = {"wiggle1": 2.2, "wiggle2": 2.2, "open_max": 0.6}
WIGGLE_ACTION_JA = "指に残った布を、手首を振って落とす"


def wiggle_window(res, ph):
    """(t0, t1) video_t window of the release-check wiggle after release phase `ph`, or None. t1 = next phase's video_t,
    t0 = t1 - sum(step durations), clamped to the release's own video_t."""
    phases = res["phases"]
    idx = next((i for i, p in enumerate(phases) if p is ph), None)
    if idx is None or idx + 1 >= len(phases):
        return None
    k = ph.get("cycle")
    c = next((c for i, c in enumerate(res.get("cycles") or []) if c.get("cycle", i + 1) == k), None)
    rc = (c or {}).get("release_check") or {}
    if not rc.get("wiggled"):
        return None
    t1 = phases[idx + 1]["video_t"]
    t0 = max(ph["video_t"], t1 - sum(WIGGLE_STEP_S.get(st.get("step"), 0.0) for st in rc.get("steps") or []))
    return t0, t1


def caption_ja_multi(res, s, lang="ja"):
    """Bottom caption of a multi take at sim-frame clock `s`: "[k/N] Tシャツ：持ち上げる" or the in-tub count."""
    tx = T[lang]
    ph = phase_obj_at(res["phases"], s)
    name = "lower" if ph["phase"] == "vlm" else ph["phase"]
    N, k = n_garments(res), ph.get("cycle")
    if name == "in tub":
        return tx["m_tub_cap"].format(n=n_in_tub_at(res, k), N=N)
    if name not in PHASES_JA_MULTI:
        raise ValueError(f"unknown phase {name!r}: no Japanese caption")
    action = PHASES_JA_MULTI[name].format(pct=100.0 * res.get("in_tub_frac_end", 0.0))
    if name == "release" and lang == "ja":
        win = wiggle_window(res, ph)
        if win and win[0] - 1e-6 <= s:
            action = WIGGLE_ACTION_JA
    if k is None:
        return tx["m_cap_plain"].format(action=action)
    cap = tx["m_cap"].format(k=k, n=N, label=label_ja(ph.get("target"), lang), action=action)
    if is_double_cycle(res, k):
        n_lift = len(next(names for c, names in double_grasp_cycles(res) if c == k))
        n_with = n_lift if name == "lift" else n_carried(res, k) if name in ("carry", "release") else 0
        if n_with >= 2:
            cap += tx["m_cap_2"].format(n=n_with)
    return cap


def check_phases(res):
    for p in res["phases"]:
        if p["phase"] != "vlm":
            if is_multi(res):
                if p["phase"] not in PHASES_JA_MULTI:
                    raise ValueError(f"unknown phase {p['phase']!r}: no Japanese caption")
            else:
                caption_ja(p["phase"], res)


def process_take(take, res, size, fps, spans, cards, lang, out):
    """Decode the take, replace each card span (frame indices [i0, i1), in order) by its card (PIL image or None to keep
    the burned frames) and, for ja, cover the burned English boxes and draw Japanese ones. Writes a high-quality
    intermediate mp4 (no audio). The simulation clock excludes all card frames before a frame."""
    w, h = size
    ja = lang == "ja"
    multi = is_multi(res)
    if ja:
        check_phases(res)
        cov = cover_rects(size, multi=multi)
    cache = {}

    def patch(text, anchor):
        key = (text, anchor)
        if key not in cache:
            cache[key] = caption_patch(size, text, anchor, cov[anchor])
        return cache[key]

    arrs = [None if c is None else np.asarray(c.convert("RGB").resize(size), np.uint8) for c in cards]
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(take), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                            "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-crf", "12", "-preset", "fast",
                            "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    n, nb = 0, w * h * 3
    while True:
        buf = dec.stdout.read(nb)
        if len(buf) < nb:
            break
        fr = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        ci = next((j for j, (i0, i1) in enumerate(spans) if i0 <= n < i1), None)
        if ci is not None:
            if arrs[ci] is not None:
                fr = arrs[ci]
        elif ja:
            fr = fr.copy()
            done = sum(min(n, i1) - i0 for i0, i1 in spans if n >= i0)   # card frames before this one
            s = (n - done) / fps
            cap = caption_ja_multi(res, s) if multi else caption_ja(phase_at(res["phases"], s), res)
            for text, anchor in ((T["ja"]["top"].format(t=s), "top"), (cap, "bottom")):
                pt, bx, by = patch(text, anchor)
                fr[by:by + pt.shape[0], bx:bx + pt.shape[1]] = pt
        enc.stdin.write(np.ascontiguousarray(fr).tobytes())
        n += 1
    enc.stdin.close()
    dec.stdout.close()
    if enc.wait() != 0 or dec.wait() != 0:
        raise SystemExit("ERROR: ffmpeg failed while processing the take")
    print(f"processed {n} take frames (card frames {spans})")
    return n


# ------------------------------------------------------------------ build
def total_duration(take_s, setup_s=SETUP_S):
    return TITLE_S + setup_s + take_s + RESULT_S


def check_duration(total, multi=False, setup_s=SETUP_S):
    print(f"total duration: {total:.1f} s (allowed {MIN_S:.0f}-{MAX_S:.0f} s)")
    if not MIN_S <= total <= MAX_S:
        what = (f"the multi-garment take is {total - TITLE_S - setup_s - RESULT_S:.1f} s (cards included) and the video "
                f"would be {total:.1f} s; shorten the take (fewer cycles / shorter cards)" if multi and total > MAX_S
                else "")
        raise SystemExit(f"ERROR: total duration {total:.1f} s is outside {MIN_S:.0f}-{MAX_S:.0f} s. {what}".rstrip())


def build(take, res, stats, out, lang, obs_path=None, json_dir=None, parts_path=None, robot_image=None):
    check_fonts()
    w, h, fps, take_s = probe(take)
    size = (w, h)
    multi = is_multi(res)
    parts_card = multi and lang == "ja"
    setup_s = SETUP_PARTS_S if parts_card else SETUP_S
    if parts_card:
        for what, pth in (("--parts", parts_path), ("--robot-image", robot_image)):
            if not pth or not Path(pth).is_file():
                raise SystemExit(f"ERROR: {what} file not found: {pth} (needed for the multi-garment ja setup card)")
        parts_data = json.load(open(parts_path))
    total = total_duration(take_s, setup_s)
    check_duration(total, multi, setup_s)
    tx = T[lang]
    tmp = Path(tempfile.mkdtemp(prefix="final_video_"))
    take = Path(take)
    if multi:
        spans, cs = card_spans(res)
        check_cards_against_scan(take, spans, fps)
        bases = [json_dir] if json_dir else []
        N = n_garments(res)
        vlms = [load_vlm_json(c, bases) for c in cs]
        cards = []
        if lang == "ja":
            for c, v, (f0, f1) in zip(cs, vlms, spans):
                op = resolve_path(c["obs"], bases)
                if not op.exists():
                    raise SystemExit(f"ERROR: card {c['cycle']}: observation not found: {op}")
                n_left = N - n_in_tub_at(res, c["cycle"] - 1)
                card = vlm_card_multi(size, lang, vlm_grasp.load_obs(op), v, c["cycle"], n_left, (f1 - f0) / fps)
                card.save(tmp / f"vlm_card_{c['cycle']}.png")
                cards.append(card)
            print(f"{len(cards)} VLM cards regenerated")
        else:
            cards = [None] * len(spans)
        rows, speed_line = results_rows_multi(res, stats, lang, vlms)
        pauses = [(a / fps, b / fps) for a, b in spans]
        title, sub, setup = tx["m_title"], tx["m_sub"], tx["m_setup"]
    else:
        (p0, p1), how = locate_pause(take, res, fps)
        print(f"VLM pause in the take: {p0:.2f}-{p1:.2f} s [{how}]")
        rows, speed_line = results_rows(res, stats, lang)
        obs_path = Path(obs_path) if obs_path else take.with_name(take.stem + "_vlm_obs.npz")
        card = None
        if obs_path.exists() and res.get("vlm"):
            card = vlm_card(size, lang, vlm_grasp.load_obs(obs_path), res["vlm"])
            card.save(tmp / "vlm_card.png")
            print(f"VLM card regenerated from {obs_path}")
        elif lang == "ja":
            raise SystemExit(f"ERROR: --lang ja needs the VLM observation ({obs_path}) and the JSON 'vlm' block")
        else:
            print(f"WARN: no VLM observation at {obs_path}; the burned card is kept")
        spans, cards, pauses = [(int(round(p0 * fps)), int(round(p1 * fps)))], [card], [(p0, p1)]
        title, sub, setup = tx["title"], tx["sub"], tx["setup"]
    take_p = tmp / "take_processed.mp4"
    process_take(take, res, size, fps, spans, cards, lang, take_p)
    if parts_card:
        card_text(size, title, sub, line_px=size[1] // 25, note=tx["m_note"]).save(tmp / "title.png")
        card_setup_parts(size, parts_data, robot_image).save(tmp / "setup.png")
    else:
        card_text(size, title, sub).save(tmp / "title.png")
        card_text(size, tx["setup_title"], setup, left=True).save(tmp / "setup.png")
    card_results(size, lang, rows, speed_line, tx["m_res_title"] if multi else None).save(tmp / "results.png")
    label_png(size, tx["label_1x"], tmp / "l1x.png")
    label_png(size, tx["label_vlm"], tmp / "lvlm.png")
    fr = f"{fps:g}"
    # the labels are separate looped inputs, switched by time (take-video seconds)
    inside = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in pauses)
    fc = (
        f"[0:v]fps={fr},format=yuv420p,setsar=1[a];[1:v]fps={fr},format=yuv420p,setsar=1[b];"
        f"[2:v]fps={fr},scale={w}:{h},format=yuv420p,setsar=1[t];"
        f"[t][3:v]overlay=enable='not({inside})':format=auto[c0];"
        f"[c0][4:v]overlay=enable='{inside}':format=auto,format=yuv420p[c];"
        f"[5:v]fps={fr},format=yuv420p,setsar=1[d];"
        f"[a][b][c][d]concat=n=4:v=1:a=0[v]"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-loop", "1", "-t", str(TITLE_S), "-i", str(tmp / "title.png"),
           "-loop", "1", "-t", str(setup_s), "-i", str(tmp / "setup.png"),
           "-i", str(take_p),
           "-loop", "1", "-t", f"{take_s:.3f}", "-i", str(tmp / "l1x.png"),
           "-loop", "1", "-t", f"{take_s:.3f}", "-i", str(tmp / "lvlm.png"),
           "-loop", "1", "-t", str(RESULT_S), "-i", str(tmp / "results.png"),
           "-filter_complex", fc, "-map", "[v]", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    w2, h2, fps2, dur2 = probe(out)
    print(f"wrote {out}: {w2}x{h2} @ {fps2:g} fps, {dur2:.1f} s")
    return 0


# ------------------------------------------------------------------ synthetic multi take (selftest / --make-synthetic)
def make_synthetic_multi(tmp, size=(640, 360), fps=10, n_cyc=3, cyc_s=9.0, pre_s=14.0, card_s=3.0):
    """A short stand-in for a multi-garment take: noisy sim frames with English captions burned by
    integrated_demo.overlay, n_cyc identical-frame card segments, and the matching JSON + tiny obs / vlm_json files.
    -> (take_mp4, json_path, res)"""
    from PIL import Image, ImageDraw
    sys.path.insert(0, str(ROOT / "scripts"))
    from integrated_demo import overlay
    w, h = size
    tmp = Path(tmp)
    labels = ["towel", "t-shirt", "sock"][:n_cyc]
    rng = np.random.default_rng(0)
    phases = [dict(phase="drop", video_t=0.0), dict(phase="settle", video_t=1.0), dict(phase="drive", video_t=3.0),
              dict(phase="settle2", video_t=10.0)]
    cycles, cards, objs = [], [], []
    caps = []   # (sim_t0, English bottom caption)
    for k in range(1, n_cyc + 1):
        t0 = pre_s + (k - 1) * cyc_s
        lab = labels[k - 1]
        for dt, ph, act in ((0.0, "vlm", None), (0.0, "lower", "arm lowered into the basket"),
                            (1.5, "grasp", "friction grasp (IPC contact + friction only)"), (3.0, "lift", "lift the towel"),
                            (4.5, "carry", "carry over the washer"), (6.0, "release", "release"),
                            (7.0, "in tub", f"in washer {k}/{n_cyc} (sim)")):
            phases.append(dict(phase=ph, video_t=t0 + dt, cycle=k, target=lab))
            if act:
                caps.append((t0 + dt, act if ph == "in tub" else f"[{k}/{n_cyc}] {lab}: {act}"))
        cycles.append(dict(cycle=k, target=lab, attempts=1 + (k == 2), n_in_tub_after=k))
        objs.append(dict(name=lab, label_by_vlm=lab, where="tub", cycle_loaded=k))
    objs[-1]["where"], objs[-1]["cycle_loaded"] = "floor", None
    cycles[-1]["n_in_tub_after"] = n_cyc - 1
    caps = [(0.0, "towel settles in the basket"), (3.0, "driving through the corridor to the washer"),
            (10.0, "towel settles after the drive")] + caps
    sim_n = int((pre_s + n_cyc * cyc_s) * fps)
    card_n = int(card_s * fps)
    frames, n_card_before = [], 0
    for i in range(sim_n):
        t = i / fps
        for k in range(1, n_cyc + 1):   # the card is held right before the cycle's first sim frame
            if i == int(round((pre_s + (k - 1) * cyc_s) * fps)):
                f0 = len(frames)
                cards.append(dict(cycle=k, frame0=f0, frame1=f0 + card_n, sim_t=t,
                                  obs=f"obs_c{k}.npz", vlm_json=f"vlm_c{k}.json"))
                eng = np.zeros((h, w, 3), np.uint8)
                eng[:] = (200, 60, 60)
                frames += [overlay(eng, "1x real time   t = %5.1f s" % t, f"VLM card {k} (English, burned)")] * card_n
        base = np.zeros((h, w, 3), np.uint8)
        base[:] = (40 + int(100 * (i % 50) / 50), 90, 140)
        base += rng.integers(0, 12, base.shape, dtype=np.uint8)
        cap = [c for t0_, c in caps if t0_ <= t + 1e-6][-1]
        frames.append(overlay(base, "1x real time   t = %5.1f s" % t, cap))
    take = tmp / "synth_multi.mp4"
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                            "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(take)],
                           stdin=subprocess.PIPE)
    for fr_ in frames:
        enc.stdin.write(np.ascontiguousarray(fr_).tobytes())
    enc.stdin.close()
    enc.wait()
    # tiny obs + vlm json per cycle: all garments boxed, order = remaining ones, one found by a re-ask
    for k in range(1, n_cyc + 1):
        rgb = np.zeros((240, 320, 3), np.uint8)
        rgb[:] = (70, 100, 150)
        np.savez(tmp / f"obs_c{k}.npz", rgb=rgb, K=np.eye(3), E=np.eye(4), E_mode="world2cam", verts=np.zeros((1, 3)))
        dets = []
        for j, lab in enumerate(labels[k - 1:]):
            x0 = 20 + 90 * j
            dets.append(dict(label=lab, box=[x0, 60, x0 + 80, 150], status="ok", order=j + 1,
                             source="recheck" if (lab == "sock") else "vlm",
                             centroid_px=[x0 + 40, 105], grasp_px=[x0 + 45, 95], area_px=500))
        if k == 1:
            dets.append(dict(label="sock", box=[250, 170, 300, 215], status="rejected", reason="floor", source="vlm"))
        json.dump(dict(detections=dets, order=list(range(len(dets))), recheck=[], latency=6.5),
                  open(tmp / f"vlm_c{k}.json", "w"))
    res = dict(phases=phases, cards=cards, cycles=cycles, objects_final=objs, success=False,
               video_extra_card_s=card_s, phase_wall_s=dict(a=100.0), phase_sim_s=dict(a=2.0))
    jp = tmp / "synth_multi.json"
    json.dump(res, open(jp, "w"))
    return take, jp, res


# ------------------------------------------------------------------ selftest
def selftest():
    ok = True

    def chk(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name, info)

    check_fonts()
    f = _font(40)
    chk("font loads (CJK glyph wider than 0)", f.getlength("籠") > 20)
    size = (1280, 720)
    fake = dict(success=True, attempts=1, in_tub_frac_end=1.0, vlm_attempts=[dict(source="heuristic", rank=None)],
                drive=dict(duration_s=18.06, steps=1807), sim_steps_per_s=2.1, wall_steps_per_s_incl_render=1.9,
                phases=[dict(phase="vlm", video_t=23.1)], video_extra_card_s=6.47)
    tmp = Path(tempfile.mkdtemp(prefix="final_video_selftest_"))
    for lang in ("ja", "en"):
        tx = T[lang]
        rows, sp = results_rows(fake, [dict(label="x", value="y")], lang)
        chk(f"[{lang}] results rows", len(rows) == 5 and rows[2][1] == "100 %" and rows[3][1] == tx["src_heur"], rows[3])
        paths = []
        for name, im in (("title", card_text(size, tx["title"], tx["sub"])),
                         ("setup", card_text(size, tx["setup_title"], tx["setup"], left=True)),
                         ("results", card_results(size, lang, rows, sp))):
            p = tmp / f"{lang}_{name}.png"
            im.save(p)
            paths.append(p)
        label_png(size, tx["label_1x"], tmp / f"{lang}_l1x.png")
        label_png(size, tx["label_vlm"], tmp / f"{lang}_lvlm.png")
        chk(f"[{lang}] PNGs written", all(p.exists() and p.stat().st_size > 1000 for p in paths), str(tmp))
    # --- multi ja title / setup-with-prices cards
    from PIL import Image
    chk("rounding 2968.94 -> $2,969 and 47 man-yen",
        parts_summary(dict(parts=[dict(label="a", price_usd=2968.94)], usd_jpy=157.54))[1] == "合計 約 $2,969（約 47 万円）",
        parts_summary(dict(parts=[dict(label="a", price_usd=2968.94)], usd_jpy=157.54))[1])
    chk("price row has thousands separator, half up",
        parts_summary(dict(parts=[dict(label="a", price_usd=1999.0), dict(label="b", price_usd=0.5)], usd_jpy=100))[0]
        == [("a", "$1,999"), ("b", "$1")])
    chk("multi ja title has 3 sub lines (2 + note)", len(T["ja"]["m_sub"]) == 2 and bool(T["ja"]["m_note"]))
    tc = card_text(size, T["ja"]["m_title"], T["ja"]["m_sub"], line_px=size[1] // 25, note=T["ja"]["m_note"])
    tc.save(tmp / "ja_m_title.png")
    Image.new("RGB", (1208, 709), (120, 120, 130)).save(tmp / "dummy_robot.png")
    syn = dict(title="ロボット構成と参考価格", usd_jpy=157.54, footnote="注記" * 40,
               parts=[dict(label="部品" + str(i) + "（長い名前）" * 3, price_usd=100.5 * (i + 1)) for i in range(4)])
    sc = card_setup_parts(size, syn, tmp / "dummy_robot.png")
    sc.save(tmp / "ja_m_setup.png")
    chk("setup card renders at card size", sc.size == size and tc.size == size)
    chk("multi ja total with the 8 s setup card", abs(total_duration(60.1, SETUP_PARTS_S) - 78.1) < 1e-9)
    # --- phase -> caption, shift after the pause
    ph = [dict(phase=n, video_t=t) for n, t in [("drop", 0.0), ("settle", 1.03), ("drive", 3.53), ("settle2", 21.6),
          ("vlm", 23.1), ("lower", 23.1), ("grasp", 24.6), ("in tub", 49.57)]]
    fps, i0, i1 = 30.0, 693, 693 + 201          # 23.1 s ... +6.7 s
    chk("phase drive", phase_at(ph, 10.0) == "drive" and caption_ja("drive", fake) == "廊下を通って洗濯機へ走行")
    chk("phase just before the pause = settle2", phase_at(ph, sim_time(i0 - 1, fps, i0, i1)) == "settle2")
    chk("first frame after the pause = lower (clock shifted)",
        abs(sim_time(i1, fps, i0, i1) - 23.1) < 1e-9 and phase_at(ph, sim_time(i1, fps, i0, i1)) == "lower")
    chk("frame 6.7 s later in the mp4 = grasp at sim 24.6", phase_at(ph, sim_time(i1 + 45, fps, i0, i1)) == "grasp")
    chk("in tub caption uses in_tub_frac_end", "100 %" in caption_ja("in tub", fake), caption_ja("in tub", fake))
    try:
        caption_ja("nonsense", fake)
        chk("unknown phase raises", False)
    except ValueError:
        chk("unknown phase raises", True)
    chk("every JSON-style phase maps", all(caption_ja(p, fake) for p in PHASES_JA))
    # --- cover boxes are at least the old boxes (+ margin); the Japanese patch is opaque and inside the frame
    cov = cover_rects(size, max_sim=120.0)
    top, bottom = old_texts(120.0)
    for anchor, texts in (("top", top), ("bottom", bottom)):
        c = cov[anchor]
        ok_ = all(c[0] <= max(0, r[0] - COVER_MARGIN) and c[1] <= max(0, r[1] - COVER_MARGIN) and c[2] >= r[2] and c[3] >= r[3]
                  for r in (old_box(size, t, anchor) for t in texts))
        chk(f"cover {anchor} >= every old box", ok_, c)
        pt, bx, by = caption_patch(size, T["ja"]["top"].format(t=123.4) if anchor == "top" else PHASES_JA["retry"],
                                   anchor, c)
        chk(f"patch {anchor} covers the cover and is in frame",
            bx <= c[0] and by <= c[1] and bx + pt.shape[1] >= c[2] and by + pt.shape[0] >= c[3]
            and bx + pt.shape[1] <= size[0] and by + pt.shape[0] <= size[1], (bx, by, pt.shape))
    # --- results layout for 0..10 rows
    lay_ok = True
    for n in range(1, 11):
        rws = [(f"ラベル{i}" * 2, "放した後もタオルが指に残った") for i in range(n)]
        g = results_layout(size, rws, T["ja"]["speed"].format(x=52.6))
        lay_ok &= g["bottom"] <= size[1] and g["y0"] + n * g["lh"] <= g["speed_y"] and g["px"] >= 12
    chk("results layout fits 1..10 rows", lay_ok)
    # --- VLM card: normal (3 valid, rank 1 chosen) and fallback
    rgb = np.zeros((480, 640, 3), np.uint8)
    rgb[:] = (60, 90, 140)
    obs = dict(rgb=rgb, K=np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]]), E=np.eye(4), E_mode="world2cam",
               verts=np.zeros((1, 3)))
    px3 = [[334, 229], [320, 282], [347, 203]]
    vn = dict(pixels=px3, rank=1, fallback=False, latency=6.7, error=None, rejections=[], heuristic_point=[0, 0, 1],
              valid_candidates=[dict(rank=i, xy=[0, 0], point=[0, 0, 0]) for i in (1, 2, 3)])
    vf = dict(pixels=px3, rank=None, fallback=True, latency=6.5, error="no_valid_candidate", valid_candidates=[],
              rejections=[dict(rank=i, pixel=px3[i - 1], reason="floor") for i in (1, 2, 3)], heuristic_point=[0.1, 0, 1])
    for lang in ("ja", "en"):
        for nm, v in (("normal", vn), ("fallback", vf)):
            im = vlm_card(size, lang, obs, v)
            arr = np.asarray(im)
            im.save(tmp / f"{lang}_card_{nm}.png")
            chk(f"[{lang}] card {nm}: 1280x720", arr.shape == (720, 1280, 3))
    sa = np.asarray(vlm_card(size, "ja", obs, vn)).reshape(-1, 3)
    chk("normal card has green and yellow", (sa == (0, 255, 0)).all(1).any() and (sa == (255, 215, 0)).all(1).any())
    sb = np.asarray(vlm_card(size, "ja", obs, vf)).reshape(-1, 3)
    chk("fallback card has red + heuristic cross", (sb == (255, 60, 60)).all(1).any() and (sb == HEUR_COLOR).all(1).any())
    chk("normal card has no heuristic cross", not (sa == HEUR_COLOR).all(1).any())
    chk("slowdown = 1/(1.9*0.01) ~ 52.6", abs(slowdown(fake) - 52.63) < 0.05, f"{slowdown(fake):.2f}")
    chk("dt from drive = 0.01", abs(sim_dt(fake) - 0.01) < 1e-4, sim_dt(fake))
    chk("pause span", pause_span(fake) == (23.1, 23.1 + 6.47), pause_span(fake))
    chk("total = 4 + 4 + take + 6", abs(total_duration(60.1) - 74.1) < 1e-9, total_duration(60.1))
    # a synthetic held-card run must be found by the frame scan
    h, w = 36, 64
    fr = np.zeros((90, h * w), np.int16)
    fr[:, :] = np.arange(90)[:, None] * 5
    fr[30:70] = 7
    same = np.abs(np.diff(fr, axis=0)).max(axis=1) <= 2
    chk("identical-run logic", same[30:69].all() and not same[29], int(same.sum()))
    # --- multi-garment take (synthetic): cards, captions per cycle, results card
    mt = Path(tempfile.mkdtemp(prefix="final_video_multi_"))
    take_m, json_m, rm = make_synthetic_multi(mt)
    sz = probe(take_m)[:2]
    fps_m = probe(take_m)[2]
    chk("synthetic take has 3 JSON cards", is_multi(rm) and len(rm["cards"]) == 3, [c["frame0"] for c in rm["cards"]])
    runs, _ = still_runs(take_m, fps_m, min_s=1.5)
    chk("frame scan sees the 3 held cards", len(runs) == 3, runs)
    spans_m, _ = card_spans(rm)
    chk("no card-range warning for the synthetic take", check_cards_against_scan(take_m, spans_m, fps_m) == 0)
    chk("a 1 s shifted card range warns",
        check_cards_against_scan(take_m, [(spans_m[0][0] + 10, spans_m[0][1] + 10)] + spans_m[1:], fps_m) == 1)
    chk("single-card detect_pause still returns the longest run", detect_pause(take_m, fps_m) is not None)
    cap = lambda s_: caption_ja_multi(rm, s_)   # noqa: E731
    sim_c2 = 14.0 + 9.0
    chk("caption before cycle 1 (no prefix)", cap(11.0) == "走行後に衣類が落ち着くのを待つ", cap(11.0))
    chk("caption cycle 2 lift", cap(sim_c2 + 3.2) == "[2/3] VLM判定「Tシャツ」：持ち上げる", cap(sim_c2 + 3.2))
    chk("caption cycle 3 grasp = sock", cap(14 + 18 + 2.0) == "[3/3] VLM判定「靴下」：摩擦で把持（IPC の接触と摩擦のみ）")
    chk("in-tub caption cycle 1 = 1/3", cap(14 + 7.5) == "洗濯機の中：1/3 着（シミュレーション判定）", cap(14 + 7.5))
    chk("in-tub caption cycle 3 = 2/3 (sock stayed out)", cap(14 + 18 + 7.5) == "洗濯機の中：2/3 着（シミュレーション判定）")
    wr = dict(phases=[dict(phase="release", video_t=10.0, cycle=1, target="towel"),
                      dict(phase="retract", video_t=15.0, cycle=1, target="towel"),
                      dict(phase="release", video_t=20.0, cycle=2, target="towel"),
                      dict(phase="retract", video_t=22.0, cycle=2, target="towel")],
              cycles=[dict(cycle=1, release_check=dict(wiggled=True, steps=[dict(step="wiggle1")])),
                      dict(cycle=2, release_check=dict(wiggled=True, steps=[dict(step="wiggle1"), dict(step="open_max"),
                                                                           dict(step="wiggle2")]))])
    c1 = lambda s_: caption_ja_multi(wr, s_)   # noqa: E731
    chk("wiggled release: 放す before the wiggle window", c1(12.7).endswith("：放す"), c1(12.7))
    chk("wiggled release: wiggle caption inside [t1-2.2, next phase)", c1(13.0).endswith("：" + WIGGLE_ACTION_JA)
        and c1(14.9).endswith(WIGGLE_ACTION_JA) and not c1(15.0).endswith(WIGGLE_ACTION_JA), c1(13.0))
    chk("wiggle window clamped to the release start", c1(20.0).endswith(WIGGLE_ACTION_JA), c1(20.0))
    wr["cycles"][0]["release_check"]["wiggled"] = False
    chk("not wiggled: caption stays 放す", c1(14.9).endswith("：放す"), c1(14.9))
    chk("unknown target -> 衣類", label_ja("towel-ish") == "衣類" and label_ja("tshirt") == "Tシャツ")
    mc = cover_rects(size, multi=True)["bottom"]
    sc_ = cover_rects(size)["bottom"]
    chk("multi bottom cover >= single cover", mc[0] <= sc_[0] and mc[2] >= sc_[2], (mc, sc_))
    longest = max(old_texts_multi(), key=lambda t: old_box(size, t, "bottom")[2] - old_box(size, t, "bottom")[0])
    lb = old_box(size, longest, "bottom")
    chk("multi cover holds the longest English caption", mc[0] <= lb[0] and mc[2] >= lb[2] and mc[1] <= lb[1] and mc[3] >= lb[3],
        (longest, lb, mc))
    vm = [json.load(open(mt / f"vlm_c{k}.json")) for k in (1, 2, 3)]
    chk("recheck count = distinct labels found by re-ask", recheck_labels(vm) == ["sock"], recheck_labels(vm))
    rows_m, sp_m = results_rows_multi(rm, [dict(label="a", value="b"), dict(label="c", value="d")], "ja", vm)
    st5 = json.load(open(ROOT / "docs/final_video/stats_multi_ja.json"))
    chk("stats_multi_ja.json has 5 rows", len(st5) == 5, len(st5))
    rows5, sp5 = results_rows_multi(rm, st5, "ja", vm)
    chk("multi results rows == the stats rows", rows5 == [(x["label"], x["value"]) for x in st5], rows5)
    chk("multi note line is the fixed text", sp5 == T["ja"]["m_note_1x"] and "等倍" in sp5, sp5)
    g5 = results_layout(size, rows5, sp5)
    card_results(size, "ja", rows5, sp5, T["ja"]["m_res_title"]).save(mt / "results_5rows.png")
    chk("results card fits with 5 rows", g5["bottom"] <= size[1] and g5["y0"] + 5 * g5["lh"] <= g5["speed_y"], g5["bottom"])
    rows10, _ = results_rows_multi(rm, [dict(label=f"s{i}", value="v") for i in range(12)], "ja", vm)
    chk("more than MAX_ROWS --stats rows are capped", len(rows10) == MAX_ROWS, len(rows10))
    # single-towel results path unchanged
    rs, sps = results_rows(fake, [dict(label="x", value="y")], "ja")
    chk("single results path unchanged", len(rs) == 5 and "倍" in sps and rs[0][0] == "タスク結果", (rs, sps))
    for k in (1, 2, 3):
        obs_k = vlm_grasp.load_obs(mt / f"obs_c{k}.npz")
        im = vlm_card_multi(size, "ja", obs_k, vm[k - 1], k, 4 - k, 3.0)
        im.save(mt / f"card_{k}.png")
        chk(f"multi card {k}: 1280x720", np.asarray(im).shape == (720, 1280, 3))
    try:
        check_duration(190.0, True)
        chk("multi take over 180 s is an error", False)
    except SystemExit as e:
        chk("multi take over 180 s is an error", "multi-garment take" in str(e) and "180" in str(e), str(e))
    outm = mt / "final_multi.mp4"
    json.dump(syn, open(tmp / "syn_parts.json", "w"), ensure_ascii=False)
    build(take_m, rm, [dict(label="a", value="b")], outm, "ja", None, mt, tmp / "syn_parts.json", tmp / "dummy_robot.png")
    w2, h2, fps2, dur2 = probe(outm)
    chk("multi final duration = 4 + 8 + take + 6", abs(dur2 - (4 + SETUP_PARTS_S + 6 + probe(take_m)[3])) < 0.3, dur2)
    try:
        build(take_m, rm, None, outm, "ja", None, mt, tmp / "nope.json", tmp / "dummy_robot.png")
        chk("missing --parts is an error", False)
    except SystemExit as e:
        chk("missing --parts is an error", "--parts" in str(e), str(e))

    def still(t):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(outm), "-frames:v", "1", "-f", "rawvideo",
                              "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.uint8).reshape(h2, w2, 3)
    for k, (f0, f1) in enumerate(spans_m, 1):
        fr_ = still(4.0 + SETUP_PARTS_S + (f0 + f1) / 2 / fps_m).astype(float)
        chk(f"final: card {k} replaced (no burned red card left)", fr_[:, :, 0].mean() < 120,
            f"R mean {fr_[:, :, 0].mean():.0f}")
    n_c2 = int((sim_c2 + 3.2) * fps_m) + 2 * (spans_m[0][1] - spans_m[0][0])
    fr2 = still(4.0 + SETUP_PARTS_S + n_c2 / fps_m).astype(float)
    pt, bx, by = caption_patch(sz, "[2/3] VLM判定「Tシャツ」：持ち上げる", "bottom", cover_rects(sz, multi=True)["bottom"])
    d_ = np.abs(fr2[by:by + pt.shape[0], bx:bx + pt.shape[1]] - pt.astype(float)).mean()
    chk("final: cycle 2 caption patch drawn at the cover position", d_ < 12, f"mean abs diff {d_:.1f}")
    print("synthetic stills for inspection:", mt)
    for nm, t in (("card2", 4.0 + SETUP_PARTS_S + (spans_m[1][0] + 5) / fps_m), ("cap_c2", 4.0 + SETUP_PARTS_S + n_c2 / fps_m),
                  ("tub_c1", 4.0 + SETUP_PARTS_S + (int((14 + 7.5) * fps_m) + spans_m[0][1] - spans_m[0][0]) / fps_m),
                  ("results", dur2 - 2.0)):
        from PIL import Image
        Image.fromarray(still(t)).save(mt / f"still_{nm}.png")
    print("selftest:", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--take")
    ap.add_argument("--json")
    ap.add_argument("--stats", help='JSON list of {"label": str, "value": str}')
    ap.add_argument("--out", default=str(ROOT / "outputs/final/final.mp4"))
    ap.add_argument("--obs", help="VLM observation npz (default: <take stem>_vlm_obs.npz)")
    ap.add_argument("--lang", choices=["ja", "en"], default="ja")
    ap.add_argument("--parts", default=str(ROOT / "docs/final_video/parts_ja.json"),
                    help="robot parts + prices JSON for the multi ja setup card")
    ap.add_argument("--robot-image", default=str(ROOT / "outputs/robot_view_labeled_crop.png"),
                    help="robot render for the multi ja setup card")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--make-synthetic", metavar="DIR", help="write a synthetic multi-garment take + JSON to DIR and exit")
    a = ap.parse_args()
    if a.make_synthetic:
        Path(a.make_synthetic).mkdir(parents=True, exist_ok=True)
        print(*make_synthetic_multi(a.make_synthetic)[:2])
        return 0
    if a.selftest:
        return selftest()
    if not a.take or not a.json:
        ap.error("--take and --json are required (or --selftest)")
    res = json.load(open(a.json))
    stats = json.load(open(a.stats)) if a.stats else None
    return build(a.take, res, stats, a.out, a.lang, a.obs, Path(a.json).resolve().parent, a.parts, a.robot_image)


if __name__ == "__main__":
    sys.exit(main() or 0)
