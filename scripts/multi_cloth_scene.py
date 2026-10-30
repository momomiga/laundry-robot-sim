"""IPC basket scene with several garments (towel, T-shirt, sock) instead of one towel, for evaluating a VLM multi-object
detector. The garments (scripts/garments.py, flat single-layer meshes) are dropped into the robot's basket one above
the other (the first of --garments is lowest), settle until still, and `--save-obs PATH` writes the observation file:

  all keys of cloth_ipc_washer_test.save_obs (cloth_verts = all garments concatenated, rgb, depth, pc_world, pc_mask,
  intrinsics, extrinsics (world -> camera), cam_transform, base_xy0) plus
    obj_names   (k,) unicode   e.g. ["towel", "t-shirt", "sock"]
    obj_slices  (k, 2) int     vertex range [start, stop) of object k in cloth_verts
    faces       (F, 3) int     triangles, indices into cloth_verts
    face_obj    (F,) int       object index of each face
    obj_colors  (k, 3) float   rendered colour (0-1) of each object
    seg_gt      (H, W) int16   CPU-rasterized ground truth with the saved intrinsics / extrinsics: object index per
                               pixel, -1 = no garment (the basket walls / other robot parts do not occlude it)
  plus <stem>_rgb.png, <stem>_seg.png (seg_gt overlaid on the rgb to check the alignment) next to the npz, and the result
  JSON (--json) with `objects` (per object: n_vertices, in_basket_frac, zmin / zmax above the basket floor,
  max edge stretch, n_below_floor, visible pixels) and the base numbers (settle_s, sim_steps_per_s, build_s, nan, ...).

Drops: the towel as in the single-towel test (standing, leaning, buckles into a heap); other garments flat with a small
random tilt (--tilt-max), a seed-dependent yaw and offset, the footprint kept inside the basket inner area. Each garment
starts --garment-gap above the previous one's highest vertex, so the initial volumes never overlap.

  flock /tmp/house_robot-gpu.lock .venv-ipc/bin/python scripts/multi_cloth_scene.py --seed 0 --stop-after settle \\
      --save-obs outputs/multi/obs_s0.npz --json outputs/multi/settle_s0.json --out outputs/multi/settle_s0.mp4
  CPU check of the drop planning: .venv-ipc/bin/python scripts/multi_cloth_scene.py --selftest
"""
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))
import cloth_ipc_washer_test as cwt  # noqa: E402
import garments as G  # noqa: E402
import params as P  # noqa: E402

FIT_MARGIN = 0.01     # the dropped footprint stays this far inside the basket inner walls [m]
OFFSET_MAX = 0.03     # max random xy offset of a garment's footprint centre from the basket centre [m]
YAW_MAX = dict(towel=20.0, tshirt=20.0, sock=180.0)   # flat garments: max |yaw| [deg] (the T-shirt is wide: shrunk to fit)


def _rot(euler):
    return Rotation.from_euler("xyz", euler, degrees=True)   # = Genesis `euler` (scipy extrinsic x-y-z, degrees)


def plan_drops(keys, rng, basket_xy, floor, drop_h, gap, lean_max, tilt_max, yaw_max=None):
    """Initial pose of every garment. keys: garment keys in drop order (lowest first). Returns a list of dicts with
    pos (3,), euler (3,), V (the mesh), bottom / top (z of the lowest / highest initial vertex) and the drawn numbers.
    RNG draws per garment, in this order: towel: yaw, lean sign, lean, dx, dy (as `Test.add_cloth`); others: yaw,
    tilt x, tilt y, dx, dy."""
    yaw_max = {**YAW_MAX, **(yaw_max or {})}
    ix, iy, _ = P.BASKET_INNER
    bx, by = basket_xy
    out = []
    z = floor + drop_h
    for key in keys:
        V, F = G.GARMENTS[key]()
        if key == "towel":
            yaw = rng.uniform(-yaw_max[key], yaw_max[key])
            lean = rng.choice([-1, 1]) * rng.uniform(0, lean_max)
            dx, dy = rng.uniform(-0.01, 0.01), rng.uniform(-0.03, 0.03)
            euler = (90 + lean, 0.0, yaw)
            info = dict(yaw=yaw, lean=lean)
        else:
            yaw = rng.uniform(-yaw_max[key], yaw_max[key])
            tx, ty = rng.uniform(-tilt_max, tilt_max), rng.uniform(-tilt_max, tilt_max)
            dx, dy = rng.uniform(-OFFSET_MAX, OFFSET_MAX), rng.uniform(-OFFSET_MAX, OFFSET_MAX)
            for _ in range(12):      # shrink the yaw until the rotated footprint fits the basket
                R = _rot((tx, ty, yaw))
                xy = R.apply(V)[:, :2]
                if (np.ptp(xy[:, 0]) <= ix - 2 * FIT_MARGIN) and (np.ptp(xy[:, 1]) <= iy - 2 * FIT_MARGIN):
                    break
                yaw *= 0.7
            euler = (tx, ty, yaw)
            info = dict(yaw=yaw, tilt_x=tx, tilt_y=ty)
        R = _rot(euler)
        W = R.apply(V)
        mid = 0.5 * (W.min(0) + W.max(0))
        if key != "towel":   # keep the footprint inside: clamp the offset to the slack
            sx = max(0.0, (ix - 2 * FIT_MARGIN - np.ptp(W[:, 0])) / 2)
            sy = max(0.0, (iy - 2 * FIT_MARGIN - np.ptp(W[:, 1])) / 2)
            dx, dy = float(np.clip(dx, -sx, sx)), float(np.clip(dy, -sy, sy))
            pos_xy = np.array([bx + dx, by + dy]) - mid[:2]
        else:                # as the single-towel test: the mesh origin (= centre) at the basket centre + offset
            pos_xy = np.array([bx + dx, by + dy])
        pos = np.array([*pos_xy, z - W[:, 2].min()])
        top = float(z + np.ptp(W[:, 2]))
        out.append(dict(key=key, pos=pos, euler=tuple(float(e) for e in euler), V=V, F=F, bottom=float(z), top=top,
                        dx=float(dx), dy=float(dy), **{k: float(v) for k, v in info.items()}))
        z = top + gap
    return out


class MultiCloth(cwt.Test):
    def add_cloth(self, scene):
        a = self.args
        keys = [k.strip() for k in a.garments.split(",") if k.strip()]
        for k in keys:
            if k not in G.GARMENTS:
                raise ValueError(f"unknown garment {k!r} (known: {', '.join(G.GARMENTS)})")
        self.keys = keys
        plans = plan_drops(keys, self.rng, cwt._basket_xy(self), cwt.BASKET_FLOOR, a.drop_h, a.garment_gap,
                           a.lean_max, a.tilt_max)
        tmp = tempfile.mkdtemp(prefix="multi_cloth_")
        self.plans, self.entities, self.obj_slices, self.faces, self.face_obj = plans, [], [], [], []
        edges, rest, n0 = [], [], 0
        for i, p in enumerate(plans):
            path = os.path.join(tmp, f"{p['key']}.obj")
            G.write_obj(path, p["V"], p["F"])
            e = G.edge_list(p["F"])
            edges.append(e + n0)
            rest.append(np.linalg.norm(p["V"][e[:, 0]] - p["V"][e[:, 1]], axis=1))
            self.obj_slices.append((n0, n0 + len(p["V"])))
            self.faces.append(p["F"] + n0)
            self.face_obj.append(np.full(len(p["F"]), i))
            n0 += len(p["V"])
            col = G.COLORS[p["key"]]
            ent = scene.add_entity(
                morph=cwt.gs.morphs.Mesh(file=path, pos=tuple(p["pos"]), euler=p["euler"]),
                material=cwt.gs.materials.FEM.Cloth(E=a.E, nu=0.3, rho=a.rho, thickness=a.thickness,
                                                    bending_stiffness=a.bend, friction_mu=a.mu),
                surface=cwt.gs.surfaces.Default(color=(*col, 1.0)))
            self.entities.append(ent)
        self.cloth = self.entities[0]
        self.edges, self.rest_len = np.concatenate(edges), np.concatenate(rest)
        self.faces, self.face_obj = np.concatenate(self.faces), np.concatenate(self.face_obj)
        self.drop = {p["key"]: {k: round(v, 3) for k, v in p.items() if isinstance(v, float)} |
                     dict(start_z=[round(p["bottom"] - cwt.BASKET_FLOOR, 3), round(p["top"] - cwt.BASKET_FLOOR, 3)])
                     for p in plans}

    def cloth_pos(self):
        parts = [e.get_state().pos.cpu().numpy().reshape(-1, 3) for e in self.entities]
        for p, (s0, s1) in zip(parts, self.obj_slices):
            if len(p) != s1 - s0:
                raise RuntimeError(f"garment vertex count {len(p)} != mesh {s1 - s0} (Genesis re-indexed the mesh?)")
        return np.concatenate(parts)

    # ---------------- observation ----------------
    def save_obs(self, path, x):
        info = super().save_obs(path, x)
        a = self.args
        z = dict(np.load(path, allow_pickle=False))
        names = [G.NAMES[k] for k in self.keys]
        z["obj_names"] = np.array(names, dtype="U16")
        z["obj_slices"] = np.array(self.obj_slices, int).reshape(-1, 2)
        z["faces"] = self.faces.astype(np.int32)
        z["face_obj"] = self.face_obj.astype(np.int32)
        z["obj_colors"] = np.array([G.COLORS[k] for k in self.keys], float)
        seg = None
        if "intrinsics" in z:
            rgb = z.get("rgb")
            Hh, Ww = rgb.shape[:2] if rgb is not None else (P.BASKET_CAM_RES[1], P.BASKET_CAM_RES[0])
            seg = G.rasterize_ids(np.asarray(x, float), self.faces, self.face_obj, z["intrinsics"], z["extrinsics"],
                                  Ww, Hh)
            z["seg_gt"] = seg
        np.savez_compressed(path, **z)
        info["keys"] = sorted(z)
        stem = str(Path(path).with_suffix(""))
        if seg is not None and z.get("rgb") is not None:
            self._write_pngs(stem, z["rgb"], seg)
            info["png"] = [stem + "_rgb.png", stem + "_seg.png"]
        self.res["objects"] = self.object_stats(x, seg)
        return info

    def _write_pngs(self, stem, rgb, seg):
        from PIL import Image
        Image.fromarray(rgb).save(stem + "_rgb.png")
        ov = rgb.astype(float)
        # overlay colours differ from the garment colours so that a misalignment is visible: magenta / cyan / yellow
        hi = np.array([[255, 0, 255], [0, 255, 255], [255, 255, 0], [255, 0, 0], [0, 0, 255]], float)
        m = seg >= 0
        ov[m] = 0.5 * ov[m] + 0.5 * hi[seg[m] % len(hi)]
        edge = m & ~(np.roll(seg, 1, 0) == seg) | m & ~(np.roll(seg, 1, 1) == seg)
        ov[edge] = hi[seg[edge] % len(hi)]
        Image.fromarray(ov.clip(0, 255).astype(np.uint8)).save(stem + "_seg.png")

    def object_stats(self, x, seg):
        rows = []
        for i, (key, (s0, s1)) in enumerate(zip(self.keys, self.obj_slices)):
            xo = x[s0:s1]
            ed = (self.edges[:, 0] >= s0) & (self.edges[:, 0] < s1)
            e = self.edges[ed]
            ln = np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1)
            rows.append(dict(
                name=G.NAMES[key], n_vertices=int(s1 - s0), n_nonfinite=int((~np.isfinite(xo)).any(1).sum()),
                in_basket_frac=round(float(self.in_basket(xo).mean()), 4),
                zmin_above_floor=round(float(xo[:, 2].min() - cwt.BASKET_FLOOR), 4),
                zmax_above_floor=round(float(xo[:, 2].max() - cwt.BASKET_FLOOR), 4),
                max_edge_stretch=round(float((ln / self.rest_len[ed]).max()), 4),
                n_below_floor=int((xo[:, 2] < cwt.BASKET_FLOOR).sum()),
                visible_px=None if seg is None else int((seg == i).sum())))
        return rows


def make_parser():
    ap = cwt.make_parser()
    ap.add_argument("--garments", default="towel,tshirt,sock", help="comma list of %s, drop order (lowest first)"
                    % "/".join(G.GARMENTS))
    ap.add_argument("--garment-gap", type=float, default=0.04, help="initial z gap between a garment's highest vertex "
                    "and the next one's lowest [m]")
    ap.add_argument("--tilt-max", type=float, default=10.0, help="max random tilt of a flat garment about x / y [deg]")
    ap.add_argument("--selftest", action="store_true", help="CPU checks of the drop planning, no simulation")
    ap.set_defaults(out="outputs/multi/settle.mp4", json="outputs/multi/settle.json", settle_max=15.0,
                    stop_after="settle", semi_implicit="off")
    return ap


def _selftest():
    bad = []

    def chk(name, ok, info=""):
        print(("ok   " if ok else "FAIL ") + name + (f"  {info}" if info else ""))
        if not ok:
            bad.append(name)
    bxy = (P.BASKET_CENTER_XY[0], P.BASKET_CENTER_XY[1])
    ix, iy, _ = P.BASKET_INNER
    floor = cwt.BASKET_FLOOR
    for seed in range(200):
        plans = plan_drops(["towel", "tshirt", "sock"], np.random.default_rng(seed), bxy, floor, 0.12, 0.04, 8.0, 10.0)
        for p in plans:
            W = _rot(p["euler"]).apply(p["V"]) + p["pos"]
            ok_xy = (np.abs(W[:, 0] - bxy[0]).max() <= ix / 2 - 0.005) and (np.abs(W[:, 1] - bxy[1]).max() <= iy / 2 - 0.005)
            ok_z = abs(W[:, 2].min() - p["bottom"]) < 1e-9 and abs(W[:, 2].max() - p["top"]) < 1e-9
            if not (ok_xy and ok_z):
                chk(f"seed {seed} {p['key']} footprint inside / z extent consistent", False,
                    f"{np.abs(W[:, 0] - bxy[0]).max():.3f} {np.abs(W[:, 1] - bxy[1]).max():.3f}")
        gaps = [plans[i + 1]["bottom"] - plans[i]["top"] for i in range(2)]
        if min(gaps) < 0.039:
            chk(f"seed {seed} vertical gaps", False, str(gaps))
    chk("200 seeds: footprints inside the basket, z extents and gaps (>= 4 cm) consistent", not bad)
    # the towel draws match Test.add_cloth (same generator, same order)
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    p = plan_drops(["towel"], r1, bxy, floor, 0.12, 0.04, 8.0, 10.0)[0]
    yaw = r2.uniform(-20, 20)
    lean = r2.choice([-1, 1]) * r2.uniform(0, 8.0)
    dx, dy = r2.uniform(-0.01, 0.01), r2.uniform(-0.03, 0.03)
    chk("towel RNG order / euler / pos as Test.add_cloth", np.allclose(p["euler"], (90 + lean, 0, yaw))
        and np.allclose(p["pos"], (bxy[0] + dx, bxy[1] + dy, floor + 0.15 * math.cos(math.radians(lean)) + 0.12)))
    p = plan_drops(["towel", "tshirt", "sock"], np.random.default_rng(0), bxy, floor, 0.12, 0.04, 8.0, 10.0)
    print("seed 0 start heights above floor:", [(q["key"], round(q["bottom"] - floor, 3), round(q["top"] - floor, 3))
                                                for q in p])
    print("selftest FAILED: " + ", ".join(bad) if bad else "selftest OK")
    return not bad


def main():
    ap = make_parser()
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)
    a, _ = ap.parse_known_args()
    if not a.save_obs:
        ap.error("--save-obs PATH is required (the per-object stats and the observation file are written there)")
    cwt.main(MultiCloth, ap)


if __name__ == "__main__":
    main()
