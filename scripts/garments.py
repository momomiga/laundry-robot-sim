"""Flat garment meshes (towel, T-shirt, sock) for the IPC basket scene, plus a CPU z-buffer rasterizer (CPU only:
numpy, no Genesis / GPU).

Real garments are two layers of cloth. Here each garment is a SINGLE-LAYER flat outline: a deliberate simplification
(the mesh is also the IPC rest shape, so it must be flat, z = 0, and a closed two-layer shape would need self contact
between the layers from the start). The outline is a regular grid with ~1 cm spacing masked to whole cells inside the
outline, so the boundary is clean and the triangles are near equilateral-ish right triangles (alternating diagonals,
the same pattern as `cloth_ipc_grasp_test.grid_cloth`).

  towel   0.30 m square (identical to the former single towel: 31 x 31 vertices)
  tshirt  body 0.18 x 0.24 m, short sleeves 0.07 m out of each side, 0.08 m deep at the top (total width 0.32 m)
  sock    L shape: leg 0.07 x 0.15 m, foot 0.07 m wide, 0.09 m beyond the leg

Meshes are centred on their area centroid, x = width direction, y = length direction (T-shirt neck / sock top at +y).

`rasterize_ids(V, F, face_obj, K, E_world2cam, W, H)` renders per-pixel object ids (-1 = none) with a z-buffer, using
the camera convention of `vlm_grasp.project` / `robot_sensors.project_points`: pc = E[:3,:3] @ p + E[:3,3] (OpenCV
axes, z forward), (u, v) = K @ pc / z, pixel (row i, col j) has its centre at (u, v) = (j, i) (integer pixel
coordinates, as `vlm_grasp.pixel_to_point` rounds).

  python scripts/garments.py --selftest
"""
import argparse
import sys
from pathlib import Path

import numpy as np

SPACING = 0.01


# ---------------- meshes ----------------
def mesh_from_cells(cells, spacing=SPACING):
    """cells: bool (nx, ny), cell (i, j) covers x in [i, i+1) * spacing, y in [j, j+1) * spacing. Returns (V, F) with
    V (N, 3) float64 (z = 0, centred on the area centroid) and F (M, 3) int, counter-clockwise seen from +z, only the
    vertices of kept cells."""
    cells = np.asarray(cells, bool)
    nx, ny = cells.shape
    vid = -np.ones((nx + 1, ny + 1), int)
    used = np.zeros((nx + 1, ny + 1), bool)
    for di in (0, 1):
        for dj in (0, 1):
            used[di:di + nx, dj:dj + ny] |= cells
    vid[used] = np.arange(used.sum())           # row-major order, like grid_cloth's i * n + j
    ii, jj = np.nonzero(used)
    V = np.stack([ii * spacing, jj * spacing, np.zeros(len(ii))], 1)
    F = []
    for i, j in zip(*np.nonzero(cells)):
        a, b, c, d = vid[i, j], vid[i + 1, j], vid[i + 1, j + 1], vid[i, j + 1]   # ccw: a b c d
        F += [[a, b, c], [a, c, d]] if (i + j) % 2 == 0 else [[a, b, d], [b, c, d]]
    F = np.array(F, int)
    ci, cj = np.nonzero(cells)
    V[:, 0] -= (ci.mean() + 0.5) * spacing
    V[:, 1] -= (cj.mean() + 0.5) * spacing
    return V, F


def _rect(mask, x0, x1, y0, y1):
    mask[x0:x1, y0:y1] = True


def towel(size=0.30, spacing=SPACING):
    n = int(round(size / spacing))
    return mesh_from_cells(np.ones((n, n), bool), spacing)


def tshirt(body_w=0.18, body_l=0.24, sleeve_out=0.07, sleeve_d=0.08, spacing=SPACING):
    r = lambda v: int(round(v / spacing))
    bw, bl, so, sd = r(body_w), r(body_l), r(sleeve_out), r(sleeve_d)
    m = np.zeros((bw + 2 * so, bl), bool)
    _rect(m, so, so + bw, 0, bl)                    # body
    _rect(m, 0, so, bl - sd, bl)                    # sleeves at the top (+y)
    _rect(m, so + bw, bw + 2 * so, bl - sd, bl)
    return mesh_from_cells(m, spacing)


def sock(leg_w=0.07, leg_l=0.15, foot_w=0.07, foot_l=0.09, spacing=SPACING):
    r = lambda v: int(round(v / spacing))
    lw, ll, fw, fl = r(leg_w), r(leg_l), r(foot_w), r(foot_l)
    m = np.zeros((lw + fl, ll), bool)
    _rect(m, 0, lw, 0, ll)                          # leg (up = +y)
    _rect(m, lw, lw + fl, 0, fw)                    # foot, pointing +x from the bottom of the leg
    return mesh_from_cells(m, spacing)


def analytic_area(name):
    return dict(towel=0.30 * 0.30, tshirt=0.18 * 0.24 + 2 * 0.07 * 0.08, sock=0.07 * 0.15 + 0.09 * 0.07)[name]


GARMENTS = dict(towel=towel, tshirt=tshirt, sock=sock)
# display name used in the observation files, and RGB colour (0-1) of the rendered garment
NAMES = dict(towel="towel", tshirt="t-shirt", sock="sock")
COLORS = dict(towel=(0.95, 0.55, 0.15), tshirt=(0.20, 0.62, 0.32), sock=(0.92, 0.92, 0.90))


def write_obj(path, V, F):
    with open(path, "w") as f:
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in F:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def edge_list(F):
    """Unique undirected edges (E, 2) of a triangle list."""
    F = np.asarray(F)
    return np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)


def mesh_report(V, F):
    """Checks used by the selftest: area, edge length range, edge-manifold, component count, degenerate / duplicate
    faces."""
    V, F = np.asarray(V, float), np.asarray(F, int)
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    E = edge_list(F)
    ln = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    ec = {}
    for t in F:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            k = (min(a, b), max(a, b))
            ec[k] = ec.get(k, 0) + 1
    parent = list(range(len(V)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in E:
        parent[find(a)] = find(b)
    return dict(area=float(area.sum()), min_area=float(area.min()), edge_min=float(ln.min()), edge_max=float(ln.max()),
                max_edge_faces=max(ec.values()), n_components=len({find(i) for i in range(len(V))}),
                n_unreferenced=int(len(V) - len(np.unique(F))),
                n_dup_faces=int(len(F) - len({tuple(sorted(t)) for t in F.tolist()})),
                n_degenerate=int((np.array([len(set(t)) for t in F.tolist()]) < 3).sum()),
                all_up=bool(((np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])[:, 2]) > 0).all()))


# ---------------- rasterizer ----------------
def rasterize_ids(V, F, face_obj, K, E_world2cam, W, H):
    """Per-pixel object id (H, W) int16 of the nearest triangle (-1 = none). V (N,3) world points, F (M,3), face_obj
    (M,) ids. Perspective-correct depth (1/z interpolated linearly in the image), nearest z wins; triangles with a
    vertex at z <= 1 mm are skipped."""
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    face_obj = np.asarray(face_obj)
    E = np.asarray(E_world2cam, float)
    K = np.asarray(K, float)
    pc = V @ E[:3, :3].T + E[:3, 3]
    z = pc[:, 2]
    uv = (K @ pc.T).T[:, :2] / np.where(z > 1e-3, z, 1.0)[:, None]
    ids = -np.ones((H, W), np.int16)
    zbuf = np.full((H, W), np.inf)
    for f in range(len(F)):
        a, b, c = F[f]
        if min(z[a], z[b], z[c]) <= 1e-3:
            continue
        p0, p1, p2 = uv[a], uv[b], uv[c]
        u0, u1 = int(np.floor(min(p0[0], p1[0], p2[0]))), int(np.ceil(max(p0[0], p1[0], p2[0])))
        v0, v1 = int(np.floor(min(p0[1], p1[1], p2[1]))), int(np.ceil(max(p0[1], p1[1], p2[1])))
        u0, u1, v0, v1 = max(u0, 0), min(u1, W - 1), max(v0, 0), min(v1, H - 1)
        if u0 > u1 or v0 > v1:
            continue
        den = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(den) < 1e-12:
            continue
        uu, vv = np.meshgrid(np.arange(u0, u1 + 1), np.arange(v0, v1 + 1))
        l0 = ((p1[1] - p2[1]) * (uu - p2[0]) + (p2[0] - p1[0]) * (vv - p2[1])) / den
        l1 = ((p2[1] - p0[1]) * (uu - p2[0]) + (p0[0] - p2[0]) * (vv - p2[1])) / den
        l2 = 1.0 - l0 - l1
        eps = -1e-9
        inside = (l0 >= eps) & (l1 >= eps) & (l2 >= eps)
        if not inside.any():
            continue
        inv_z = l0 / z[a] + l1 / z[b] + l2 / z[c]
        zz = 1.0 / np.where(inv_z > 0, inv_z, np.nan)
        sub = zbuf[v0:v1 + 1, u0:u1 + 1]
        win = inside & (zz < sub)
        sub[win] = zz[win]
        ids[v0:v1 + 1, u0:u1 + 1][win] = face_obj[f]
    return ids


# ---------------- selftest ----------------
def _selftest():
    bad = []

    def chk(name, ok, info=""):
        print(("ok   " if ok else "FAIL ") + name + (f"  {info}" if info else ""))
        if not ok:
            bad.append(name)
    for name, fn in GARMENTS.items():
        V, F = fn()
        r = mesh_report(V, F)
        ana = analytic_area(name)
        chk(f"{name}: area within 5% of analytic", abs(r["area"] / ana - 1) < 0.05,
            f"{r['area']:.5f} vs {ana:.5f} (nV {len(V)}, nF {len(F)})")
        chk(f"{name}: edge lengths in [0.5, 1.6] x spacing", r["edge_min"] >= 0.5 * SPACING - 1e-9
            and r["edge_max"] <= 1.6 * SPACING + 1e-9, f"{r['edge_min']:.4f}..{r['edge_max']:.4f}")
        chk(f"{name}: edge-manifold", r["max_edge_faces"] <= 2)
        chk(f"{name}: one component, no unreferenced vertices", r["n_components"] == 1 and r["n_unreferenced"] == 0)
        chk(f"{name}: no degenerate / duplicate faces, all facing +z", r["n_degenerate"] == 0
            and r["n_dup_faces"] == 0 and r["min_area"] > 1e-9 and r["all_up"])
        chk(f"{name}: flat, centred", np.allclose(V[:, 2], 0) and np.allclose(
            V[:, :2].mean(0), 0, atol=0.05))
    Vt, _ = towel()
    chk("towel: 961 vertices like grid_cloth(31, 0.30)", len(Vt) == 961)
    Vs, _ = tshirt()
    chk("tshirt: width 0.32 m, length 0.24 m", np.allclose(np.ptp(Vs[:, 0]), 0.32) and np.allclose(np.ptp(Vs[:, 1]), 0.24))
    # rasterizer: top-down camera 1 m above the origin, image up = world +x (as robot_sensors / the basket camera)
    W, H = 640, 480
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    E = np.array([[0, -1.0, 0, 0], [-1.0, 0, 0, 0], [0, 0, -1.0, 1.0], [0, 0, 0, 1.0]])
    p = np.array([[0.10, 0.05, 0.0]])
    pc = p @ E[:3, :3].T + E[:3, 3]
    uv = (K @ pc.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    chk("projection of a known point", np.allclose(uv[0], [295.0, 190.0], atol=1e-6), str(uv[0]))

    def square(cx, cy, z, h=0.1):
        V = np.array([[cx - h, cy - h, z], [cx + h, cy - h, z], [cx + h, cy + h, z], [cx - h, cy + h, z]])
        return V, np.array([[0, 1, 2], [0, 2, 3]])
    V0, F0 = square(0.0, 0.0, 0.10)
    V1, F1 = square(0.0, 0.10, 0.20)        # higher, shifted +y (image left): overlaps for y in [0, 0.1]
    V = np.concatenate([V0, V1])
    F = np.concatenate([F0, F1 + 4])
    ids = rasterize_ids(V, F, np.array([0, 0, 1, 1]), K, E, W, H)

    def at(x, y, z=0.0):
        q = np.array([x, y, z]) @ E[:3, :3].T + E[:3, 3]
        return int(round(500 * q[0] / q[2] + 320)), int(round(500 * q[1] / q[2] + 240))
    u, v = at(0.0, 0.05)
    chk("overlap: the higher square wins", ids[v, u] == 1, f"id {ids[v, u]} at ({u},{v})")
    u, v = at(0.0, -0.05)
    chk("only the lower square", ids[v, u] == 0)
    u, v = at(0.0, 0.15)
    chk("only the higher square", ids[v, u] == 1)
    u, v = at(0.3, 0.0)
    chk("background is -1", ids[v, u] == -1 and ids.dtype == np.int16)
    ys, xs = np.nonzero(ids >= 0)
    # analytic extents: u from -0.1 m at depth 0.9 to +0.2 m at depth 0.8, v +-0.1 m at depth 0.8 (the higher square)
    eu, ev = 500 * 0.1 / 0.9 + 500 * 0.2 / 0.8, 2 * 500 * 0.1 / 0.8
    chk("footprint extent (px)", abs(xs.max() - xs.min() - eu) <= 2 and abs(ys.max() - ys.min() - ev) <= 2,
        f"{xs.max() - xs.min()} x {ys.max() - ys.min()} vs {eu:.1f} x {ev:.1f}")
    # known vertex pixel: the corner (x=0.1, y=-0.1, z=0.1) of the lower square is at the image bottom-left-ish
    u, v = at(0.099, -0.099, 0.10)
    chk("point near a corner of the lower square is inside (+-1 px)", ids[v, u] in (0, 1))
    # reversing the heights flips the winner in the overlap
    V2 = V.copy()
    V2[:4, 2], V2[4:, 2] = 0.20, 0.10
    ids2 = rasterize_ids(V2, F, np.array([0, 0, 1, 1]), K, E, W, H)
    u, v = at(0.0, 0.05)
    chk("overlap: swapping heights swaps the winner", ids2[v, u] == 0)
    print("selftest FAILED: " + ", ".join(bad) if bad else "selftest OK")
    return not bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--write", metavar="DIR", help="write towel/tshirt/sock .obj into DIR")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.write:
        Path(a.write).mkdir(parents=True, exist_ok=True)
        for n, fn in GARMENTS.items():
            write_obj(str(Path(a.write) / f"{n}.obj"), *fn())
    else:
        ap.print_help()
