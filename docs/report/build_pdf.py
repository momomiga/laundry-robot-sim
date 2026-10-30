#!/usr/bin/env python3
"""Build outputs/report/report.pdf from docs/report/report.md (title + sections 1-8).

Run: uv run --no-sync --with markdown-it-py python docs/report/build_pdf.py
Needs headless google-chrome. Appendices (from "## 付録 A") and HTML comments are dropped.
"""
import html
import os
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="report.md", help="markdown file in docs/report/")
ap.add_argument("--out", default="outputs/report/report.pdf", help="PDF path (relative to the repo root)")
args = ap.parse_args()
OUT = (ROOT / args.out).resolve()
CHROME = os.environ.get("CHROME", "/usr/bin/google-chrome")

text = (HERE / args.src).read_text(encoding="utf-8")
text = re.split(r"^## 付録 A", text, maxsplit=1, flags=re.M)[0]
text = re.sub(r"\n-{3,}\s*$", "\n", text.rstrip())  # the rule that separated the appendices
text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
title = re.search(r"^# (.+)$", text, re.M).group(1).strip()

todos = [l.strip() for l in text.splitlines() if "TODO" in l]
for t in todos:
    print(f"WARNING: TODO remains: {t}", file=sys.stderr)

md = MarkdownIt("commonmark").enable("table")
body = md.render(text)
# standalone image paragraph -> figure with caption
body = re.sub(
    r'<p><img src="([^"]+)" alt="([^"]*)"\s*/?></p>',
    lambda m: f'<figure><img src="{m.group(1)}" alt=""><figcaption>{m.group(2)}</figcaption></figure>',
    body,
)
# short table cells (ids, GPU names, numbers) never wrap
body = re.sub(r"<(t[dh])>([^<]{1,16})</\1>", r'<\1 style="white-space:nowrap">\2</\1>', body)
css = (HERE / "report.css").read_text(encoding="utf-8")
doc = (f'<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8">'
       f'<title>{html.escape(title)}</title><style>{css}</style></head><body class="{Path(args.src).stem}">{body}</body></html>')

OUT.parent.mkdir(parents=True, exist_ok=True)
tmp = HERE / f"_{Path(args.src).stem}_tmp.html"  # in docs/report so figures/... resolve
try:
    tmp.write_text(doc, encoding="utf-8")
    with tempfile.TemporaryDirectory() as prof:
        subprocess.run([CHROME, "--headless=new", "--no-sandbox", "--disable-gpu",
                        f"--user-data-dir={prof}", "--no-pdf-header-footer",
                        f"--print-to-pdf={OUT}", tmp.as_uri()], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
finally:
    tmp.unlink(missing_ok=True)
print(f"wrote {OUT} ({len(todos)} TODO line(s))")
