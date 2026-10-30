"""Static charts for the report: fig06 (3-garment cycles), fig07 (phase wall time), fig08 (drift probe).

Run from the repo root:  uv run --no-sync python docs/report/make_charts.py
CPU only. Reads outputs/**.json / .npz, writes docs/report/figures/fig0{6,7,8}_*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "docs" / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- style -------------------------------------------------------------
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
NEUTRAL = "#898781"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # same as scripts/make_final_video.py
fm.fontManager.addfont(FONT)
FP = fm.FontProperties(fname=FONT)
plt.rcParams.update({
    "font.family": FP.get_name(), "font.size": 8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASE, "axes.labelcolor": INK3, "xtick.color": INK3, "ytick.color": INK3,
    "axes.spines.top": False, "axes.spines.right": False,
    "hatch.linewidth": 0.8,
})
W = 16 / 2.54
DPI = 200


def label_color(c):
    return INK if c == "#eda100" else "white"


def style_axes(ax, grid="x"):
    ax.grid(axis=grid, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(BASE)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(length=2, width=0.5, colors=INK3)


def load(p):
    return json.loads((ROOT / p).read_text())


# ======================================================================
# fig06: garments entering the tub per cycle, 12 takes
# ======================================================================
TAKES = (["outputs/multi/r3_s0.json", "outputs/remote/multi/r3_s1.json", "outputs/multi/r3_s2.json"]
         + [f"outputs/remote/multi/r4_s{s}.json" for s in (3, 4, 5)]
         + [f"outputs/multi/r4_s{s}.json" for s in (6, 7, 8)]
         + [f"outputs/remote/multi/r5_s{s}.json" for s in (9, 10, 11)])
NAMES = ["R3 s0", "R3 s1", "R3 s2"] + [f"R4 s{s}" for s in range(3, 9)] + [f"R5 s{s}" for s in (9, 10, 11)]


def fig06():
    rows, ncyc = [], 0
    for p in TAKES:
        d = load(p)
        after = [0] + [c["n_in_tub_after"] for c in d["cycles"]]
        per = [after[i + 1] - after[i] for i in range(len(after) - 1)]
        assert all(x >= 0 for x in per), (p, per)
        out = sum(1 for o in d["objects_final"] if o["where"] != "tub")
        assert sum(per) + out == 3, (p, per, out)
        assert sum(per) == sum(1 for o in d["objects_final"] if o["where"] == "tub"), p
        rows.append((per, out))
        ncyc = max(ncyc, len(per))
    total = sum(sum(r[0]) for r in rows)
    assert total == 32 and len(rows) == 12, total
    print(f"fig06: loaded {total} of {3 * len(rows)}; max cycles with a card = {ncyc}; per-take: "
          + ", ".join(f"{n}:{r[0]}+out{r[1]}" for n, r in zip(NAMES, rows)))
    n_slots = max(max((i + 1 for i, v in enumerate(r[0]) if v), default=0) for r in rows)
    n_slots = max(n_slots, 3)
    print("fig06: segments used (cycles with loading):", n_slots)

    fig, ax = plt.subplots(figsize=(W, 8.4 / 2.54 + 0.9))
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    handles = []
    series = [([r[0][k] if k < len(r[0]) else 0 for r in rows], CAT[k], f"{k + 1} 回目の把持サイクルで入った", None)
              for k in range(n_slots)]
    series.append(([r[1] for r in rows], NEUTRAL, "籠の外に落ちた", "///"))
    for vals, col, lab, hatch in series:
        vals = np.array(vals, float)
        if vals.sum() == 0:
            continue
        ax.barh(y, vals, left=left, height=0.68, color=col, edgecolor=SURFACE, linewidth=1.44,
                hatch=hatch)
        for yi, (l, v) in enumerate(zip(left, vals)):
            if v > 0:
                ax.text(l + v / 2, yi, f"{int(v)}", ha="center", va="center", fontsize=8,
                        color=INK if hatch else label_color(col),
                        bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none") if hatch else None)
        left += vals
        handles.append(Patch(facecolor=col, edgecolor=SURFACE if not hatch else "white", hatch=hatch, label=lab))
    ax.set_yticks(y)
    ax.set_yticklabels(NAMES)
    ax.invert_yaxis()
    ax.set_xlim(0, 3)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("着数（各テイク 3 着の行き先）")
    style_axes(ax, "x")
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
              frameon=False, fontsize=7, labelcolor=INK2, handlelength=1.0, handletextpad=0.5, columnspacing=0.8)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.93, bottom=0.13)
    fig.savefig(OUT / "fig07_multi_cycles.png", dpi=DPI)
    plt.close(fig)


# ======================================================================
# fig07: wall time per phase group
# ======================================================================
FIG7_TAKES = [("タオル vlm_s2", "outputs/remote/integrated_semi/vlm_s2.json"),
              ("タオル vlm_s3", "outputs/remote/integrated_semi/vlm_s3.json"),
              ("3 着 R3 s0", "outputs/multi/r3_s0.json"),
              ("3 着 R3 s1", "outputs/remote/multi/r3_s1.json"),
              ("3 着 R3 s2", "outputs/multi/r3_s2.json")]
# phase_wall_s key -> category (every key must be mapped; unmapped keys fail loudly)
GROUPS = [
    ("布の落下・静置", ["drop", "settle", "settle2"]),
    ("走行", ["drive"]),
    ("認識（VLM）", ["vlm"]),
    ("把持・持ち上げ", ["lower", "grasp", "lift"]),
    ("運搬・解放", ["carry", "release"]),
    ("その他（腕の退避・籠内）", ["retract", "in tub"]),
]


def fig07():
    mapped = {k for _, ks in GROUPS for k in ks}
    data = []
    for name, p in FIG7_TAKES:
        pw = load(p)["phase_wall_s"]
        assert set(pw) <= mapped, (p, set(pw) - mapped)
        g = [sum(pw.get(k, 0.0) for k in ks) / 60 for _, ks in GROUPS]
        assert abs(sum(g) - sum(pw.values()) / 60) < 1e-9
        data.append((name, g, sum(pw.values()), pw["drive"]))
        print(f"fig07: {name}: total {sum(pw.values()):.1f} s = {sum(pw.values()) / 60:.1f} min; drive share "
              f"{100 * pw['drive'] / sum(pw.values()):.0f} %")
    fig, ax = plt.subplots(figsize=(W, 6.2 / 2.54 + 0.9))
    y = np.arange(len(data))
    left = np.zeros(len(data))
    handles = []
    for gi, (gname, _) in enumerate(GROUPS):
        last = gi == len(GROUPS) - 1
        col = NEUTRAL if last else CAT[gi]
        hatch = "///" if last else None
        vals = np.array([d[1][gi] for d in data])
        ax.barh(y, vals, left=left, height=0.66, color=col, edgecolor=SURFACE, linewidth=1.44, hatch=hatch)
        for yi, (l, v) in enumerate(zip(left, vals)):
            if v >= 5:
                ax.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color=INK if (hatch or col == "#eda100") else "white")
        left += vals
        handles.append(Patch(facecolor=col, edgecolor=SURFACE if not hatch else "white", hatch=hatch, label=gname))
    for yi, d in enumerate(data):
        ax.text(left[yi] + 1.2, yi, f"{d[2] / 60:.0f} 分", ha="left", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([d[0] for d in data])
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.12)
    ax.set_xlabel("実時間 [分]")
    style_axes(ax, "x")
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False,
              fontsize=7.5, labelcolor=INK2, handlelength=1.2, columnspacing=1.0)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.84, bottom=0.17)
    fig.savefig(OUT / "fig09_phase_wall.png", dpi=DPI)
    plt.close(fig)


# ======================================================================
# fig08: cloth drift during the 1.2 m drive test
# ======================================================================
ARMS = [("base", "既定（早期終了あり）"), ("semi64", "早期終了を切る"),
        ("beta1e-4", "閾値 beta = 1e-4"), ("beta1e-5", "閾値 beta = 1e-5")]
QUOTED = {"base": -94.7, "beta1e-4": -100.4, "beta1e-5": -108.3}


def fig08():
    fig, ax = plt.subplots(figsize=(W, 7.2 / 2.54 + 0.6))
    ends = []
    for k, (tag, lab) in enumerate(ARMS):
        j = load(f"outputs/remote/drift_probe_{tag}.json")["drift_probe"]
        z = np.load(ROOT / f"outputs/remote/drift_probe_{tag}.npz", allow_pickle=True)
        cols, rows = list(z["cols"]), z["rows"]
        t, cx = rows[:, cols.index("t")], rows[:, cols.index("cx_rel")] * 1000.0  # cx_rel is in m -> mm
        t0 = j["drive_t0"]
        i0 = int(np.argmin(abs(t - t0)))
        sel = t >= t0 - 1e-9
        ts, d = t[sel] - t0, cx[sel] - cx[i0]
        # reproduce the JSON's drift_dcx_mm and the report's quoted values
        for key, v in j["drift_dcx_mm"].items():
            i = int(np.argmin(abs(ts - float(key[1:-1]))))
            assert abs(round(d[i], 1) - v) <= 0.11, (tag, key, d[i], v)
        v1 = d[int(np.argmin(abs(ts - 1.0)))]
        if tag in QUOTED:
            assert round(v1, 1) == QUOTED[tag], (tag, v1)
        print(f"fig08: {tag}: drift at +1.0 s = {v1:.1f} mm; drift_dcx_mm = {j['drift_dcx_mm']}; "
              f"line ends t = {t[-1]:.2f} s (+{ts[-1]:.2f} s), end drift {d[-1]:.1f} mm; aborted = {j['meta']['aborted']}")
        if tag == "semi64":
            assert abs(t[-1] - 7.60) < 0.011 and max(abs(d)) <= 0.5
        ax.plot(ts, d, color=CAT[k], linewidth=1.44, solid_capstyle="butt")
        ends.append((ts[-1], d[-1], tag, lab, k))
    ax.axhline(0, color=BASE, linewidth=0.8, zorder=0)
    ax.axvline(1.0, color=BASE, linewidth=0.5, linestyle=(0, (3, 3)), zorder=0)
    ax.text(1.04, 12, "走り出し 1 s", fontsize=7, color=INK3, va="bottom")
    xmax = max(e[0] for e in ends)
    # direct labels at line ends (manual vertical spread so labels never collide)
    ypos = {"base": -121.0, "semi64": 10.0, "beta1e-4": -102.0, "beta1e-5": -140.0}
    for x, yv, tag, lab, k in ends:
        short = {"base": "既定", "semi64": "早期終了を切る", "beta1e-4": "1e-4", "beta1e-5": "1e-5"}[tag]
        note = "（計算時間の上限で打ち切り）" if tag == "semi64" else ""
        xt = x + 0.12
        ax.text(xt, ypos[tag], f"{short}  " + f"{yv:.1f}".replace("-", "\u2212") + note, fontsize=7.5, color=INK,
                va="center", ha="left")
    ax.set_xlim(0, xmax + 1.3)
    ax.set_ylim(-185, 25)
    ax.set_xlabel("走り出しからの時刻 [s]")
    ax.set_ylabel("布のずれ [mm]")
    style_axes(ax, "y")
    handles = [Line2D([0], [0], color=CAT[k], linewidth=1.44, label=lab) for k, (_, lab) in enumerate(ARMS)]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=4, frameon=False,
              fontsize=7.5, labelcolor=INK2, handlelength=1.6, columnspacing=1.0)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.17)
    fig.savefig(OUT / "fig04_drift_probe.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    fig06()
    fig07()
    fig08()
