#!/usr/bin/env python3
"""VLM garment labels vs the true garment over the 12 three-garment takes (report section 5).

Run from the repo root: python3 docs/report/count_labels.py
Counts every matched pair in evaluation.objects of each <take>_vlm_c<k>.json (true garment from the ground-truth
mask, label from the VLM), in total and split by where the detection came from (first answer / recheck).
"""
import glob
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAKES = dict(local=["r3_s0", "r3_s2", "r4_s6", "r4_s7", "r4_s8"],
             remote=["r3_s1", "r4_s3", "r4_s4", "r4_s5", "r5_s9", "r5_s10", "r5_s11"])
DIRS = dict(local=ROOT / "outputs/multi", remote=ROOT / "outputs/remote/multi")
NAMES = ["towel", "t-shirt", "sock"]

pairs = Counter()   # (source, true, label)
for where, takes in TAKES.items():
    for take in takes:
        files = sorted(glob.glob(str(DIRS[where] / f"{take}_vlm_c*.json")))
        assert files, f"no vlm json for {take} in {DIRS[where]}"
        for f in files:
            v = json.load(open(f))
            for o in (v.get("evaluation") or {}).get("objects", []):
                pairs[v["detections"][o["det_index"]]["source"], o["gt_name"], o["label"]] += 1

for title, keep in (("all", None), ("first answer (vlm)", "vlm"), ("recheck", "recheck")):
    sub = Counter()
    for (src, t, l), n in pairs.items():
        if keep in (None, src):
            sub[t, l] += n
    total, ok = sum(sub.values()), sum(n for (t, l), n in sub.items() if t == l)
    print(f"{title}: {ok}/{total} correct")
    print("  true \\ label " + " ".join(f"{n:>8}" for n in NAMES))
    for t in NAMES:
        print(f"  {t:<12} " + " ".join(f"{sub[t, l]:>8}" for l in NAMES))
