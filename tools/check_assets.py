#!/usr/bin/env python3
"""Syntax/consistency check of every URDF under assets/ (standard library only).

Checks: XML parses, links/joints named uniquely, joint parent/child exist, exactly
one root link, tree has no cycles, every mesh path exists, materials referenced
are defined (or inline), revolute/prismatic joints have limits.
Usage: python3 tools/check_assets.py [file.urdf ...]
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check(path: Path) -> list[str]:
    errs = []
    try:
        robot = ET.parse(path).getroot()
    except ET.ParseError as e:
        return [f"XML parse error: {e}"]
    if robot.tag != "robot":
        return ["root element is not <robot>"]
    links = [l.get("name") for l in robot.findall("link")]
    if len(set(links)) != len(links):
        errs.append("duplicate link names")
    joints = robot.findall("joint")
    jn = [j.get("name") for j in joints]
    if len(set(jn)) != len(jn):
        errs.append("duplicate joint names")
    children, parent_of = set(), {}
    for j in joints:
        p, c = j.find("parent").get("link"), j.find("child").get("link")
        for n in (p, c):
            if n not in links:
                errs.append(f"joint {j.get('name')}: unknown link {n}")
        if c in children:
            errs.append(f"link {c} has multiple parent joints")
        children.add(c)
        parent_of[c] = p
        if j.get("type") in ("revolute", "prismatic") and j.find("limit") is None:
            errs.append(f"joint {j.get('name')}: missing <limit>")
        if j.get("type") in ("revolute", "prismatic", "continuous") and j.find("axis") is None:
            errs.append(f"joint {j.get('name')}: missing <axis>")
    roots = [l for l in links if l not in children]
    if len(roots) != 1:
        errs.append(f"expected 1 root link, found {roots}")
    for c in parent_of:  # cycle check
        seen, n = set(), c
        while n in parent_of:
            if n in seen:
                errs.append(f"cycle at {n}")
                break
            seen.add(n)
            n = parent_of[n]
    top_mats = {m.get("name") for m in robot.findall("material")}
    for m in robot.iter("material"):
        if m.find("color") is None and m.find("texture") is None and m.get("name") not in top_mats:
            errs.append(f"undefined material {m.get('name')}")
    n_mesh = 0
    for mesh in robot.iter("mesh"):
        fn = mesh.get("filename", "")
        n_mesh += 1
        if "://" in fn:
            errs.append(f"non-relative mesh path: {fn}")
            continue
        if not (path.parent / fn).is_file():
            errs.append(f"missing mesh: {fn}")
    for l in robot.findall("link"):
        if l.find("inertial") is None and l.find("collision") is not None:
            errs.append(f"link {l.get('name')}: collision without inertial")
    print(f"  {len(links)} links, {len(joints)} joints, {n_mesh} mesh refs, root={roots}")
    return errs


def main() -> int:
    files = [Path(a) for a in sys.argv[1:]] or sorted(ROOT.glob("assets/**/*.urdf"))
    bad = 0
    for f in files:
        print(f.relative_to(ROOT) if f.is_absolute() and ROOT in f.parents else f)
        e = check(f)
        for m in e:
            print("  ERROR:", m)
        bad += bool(e)
    print("FAILED" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
