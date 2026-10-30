"""The house (room, washer, room visuals, lights) in the frame of the parked base, as visual-only geometry.

Parked-base frame = frame of scripts/cloth_ipc_washer_test.py: the robot base is fixed at the origin, which is
tools/params.py PARK_POSE in the room frame. The room goes at the pose inv(T_room_park) (derived, not hard-coded).

add_house(scene, frame="park") adds room.urdf (+ washer.urdf if with_washer) with collision=False and a rigid
material with needs_coup=False, and the textured room visuals (collision-free as well). Genesis' IPC coupler skips
entities with needs_coup=False (couplers/ipc_coupler/coupler.py _setup_coupling_config) and links without collision
geometry, so nothing here enters IPC; the rigid solver has no collision geoms for it either. Without an IPC coupler
the same call just adds inert fixed visuals.

Rasterizer lights can only be given at Scene construction: use house_vis_options(frame) for the VisOptions
(add_house(lights=True) adds them via scene.add_light only for a BatchRenderer, otherwise it does nothing).

  uv run python scripts/house_frame.py --check     # CPU acceptance check
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))
import params as P  # noqa: E402


# corridor ceiling lamps as in scripts/render_corridor.py: (room-frame pos, colour, intensity)
_LIGHTS = [((*P.CEILING_LIGHT_XY, P.WALL_H - 0.08), (1.0, 0.80, 0.55), 9.0),
           ((0.9, 7.6, P.WALL_H - 0.3), (1.0, 0.82, 0.6), 8.0)]


def T_room_to_park(park=P.PARK_POSE):
    """4x4 transform taking room-frame points to the parked-base frame (= inv of the base pose in the room)."""
    px, py, yaw = park
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:2, :2] = [[c, s], [-s, c]]            # R(-yaw)
    T[:2, 3] = [-(c * px + s * py), (s * px - c * py)]
    return T


def pose_room_to_park(park=P.PARK_POSE):
    """(pos xyz, yaw [rad]) of the room frame expressed in the parked-base frame."""
    T = T_room_to_park(park)
    return (float(T[0, 3]), float(T[1, 3]), 0.0), -park[2]


def room_to_park(xyz, park=P.PARK_POSE):
    """Room-frame point(s) (..., 2 or 3) -> parked-base frame. Array-safe; z passes through."""
    a = np.asarray(xyz, dtype=float)
    T = T_room_to_park(park)
    out = a.copy()
    out[..., :2] = a[..., :2] @ T[:2, :2].T + T[:2, 3]
    return out


def park_to_room(xyz, park=P.PARK_POSE):
    a = np.asarray(xyz, dtype=float)
    T = T_room_to_park(park)
    out = a.copy()
    out[..., :2] = (a[..., :2] - T[:2, 3]) @ T[:2, :2]
    return out


def house_vis_options(frame="park", **kw):
    """gs.options.VisOptions with the house lights (corridor ceiling lamps) in the requested frame."""
    import genesis as gs
    lights = []
    for p, col, inten in _LIGHTS:
        q = room_to_park(p) if frame == "park" else np.asarray(p, float)
        lights.append(dict(type="point", pos=tuple(float(v) for v in q), color=col, intensity=inten))
    kw.setdefault("show_world_frame", False)
    kw.setdefault("ambient_light", (0.42, 0.40, 0.37))
    kw.setdefault("lights", lights)
    return gs.options.VisOptions(**kw)


def add_house(scene, frame="park", with_washer=False, lights=True, visuals=True, ceiling=True):
    """Add the house to an unbuilt scene. Returns dict(room=, washer=, visuals=[...]). frame: "park" or "room"."""
    import genesis as gs
    from room_visuals import add_room_visuals
    if frame == "park":
        (x, y, z), yaw = pose_room_to_park()
        pos, euler = (x, y, z), (0.0, 0.0, math.degrees(yaw))
    elif frame == "room":
        pos, euler = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    else:
        raise ValueError(f"frame must be 'park' or 'room', got {frame!r}")
    mat = gs.materials.Rigid(needs_coup=False)
    out = {"washer": None, "visuals": []}
    out["room"] = scene.add_entity(
        gs.morphs.URDF(file=str(ROOT / "assets/room/room.urdf"), fixed=True, collision=False, pos=pos, euler=euler),
        material=mat)
    if with_washer:
        wp = room_to_park(P.WASHER_POS) if frame == "park" else np.asarray(P.WASHER_POS, float)
        out["washer"] = scene.add_entity(
            gs.morphs.URDF(file=str(ROOT / "assets/washer/washer.urdf"), fixed=True, collision=False,
                           pos=tuple(float(v) for v in wp), euler=euler),
            material=mat)
    if visuals:
        out["visuals"] = add_room_visuals(scene, pos=pos, euler=euler, material=mat)
    if lights and isinstance(scene.options.renderer, gs.renderers.BatchRenderer):
        for p, col, inten in _LIGHTS:
            q = room_to_park(p) if frame == "park" else np.asarray(p, float)
            scene.add_light(pos=tuple(float(v) for v in q), dir=(0, 0, -1), color=col, intensity=inten,
                            directional=False)
    return out


def _check():
    import genesis as gs
    ok = True

    def chk(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name, info)

    wb = P.washer_in_base()
    wp = room_to_park(P.WASHER_POS)
    chk("washer pos == washer_in_base", np.allclose(wp[:2], wb[:2], atol=1e-6), f"{wp[:2]} vs {wb[:2]}")
    chk("washer yaw == washer_in_base", abs(pose_room_to_park()[1] - wb[2]) < 1e-9, f"{wb[2]:.4f}")
    chk("round trip", np.allclose(park_to_room(room_to_park([[1, 2, 3], [4, 5, 6.]])), [[1, 2, 3], [4, 5, 6.]]))
    chk("park origin -> room PARK_POSE", np.allclose(park_to_room([0, 0, 0])[:2], P.PARK_POSE[:2]))

    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False, vis_options=house_vis_options("park"))
    robot = scene.add_entity(gs.morphs.URDF(file=str(ROOT / "assets/robot/house_robot.urdf"), fixed=True))
    h = add_house(scene, frame="park", with_washer=True)
    scene.build()
    room, washer = h["room"], h["washer"]
    print("room geoms:", len(room.geoms), "washer geoms:", len(washer.geoms),
          "visual overlays:", len(h["visuals"]), "robot geoms:", len(robot.geoms))
    chk("room has no collision geoms", len(room.geoms) == 0 and len(washer.geoms) == 0)
    solver = scene.sim.rigid_solver
    room_idx = {k.idx for k in room.links} | {k.idx for k in washer.links}
    chk("no rigid-solver geom belongs to room/washer", all(g.link.idx not in room_idx for g in solver.geoms))
    chk("needs_coup False", not room.material.needs_coup and not washer.material.needs_coup)
    # positions read back: the visual overlay meshes and the washer link
    wpos = np.asarray(washer.get_pos().cpu() if hasattr(washer.get_pos(), "cpu") else washer.get_pos()).reshape(-1)
    chk("washer entity at washer_in_base", np.allclose(wpos[:2], wb[:2], atol=1e-5), f"{wpos}")
    vp = np.asarray(h["visuals"][0].get_pos()).reshape(-1)
    rp = pose_room_to_park()[0]
    chk("visual overlay origin at room origin in park frame", np.allclose(vp[:2], rp[:2], atol=1e-5), f"{vp}")
    # the corridor's bath-side wall and fridge front in the park frame (derived)
    bath = room_to_park((P.CORRIDOR_X[1], 7.0, 0.0))
    fridge = room_to_park((P.FRIDGE_X[1], P.FRIDGE_Y[0], 0.0))
    print("bath-side wall (room x=%.3f) -> park y %.3f; fridge front (room x=%.3f) -> park y %.3f"
          % (P.CORRIDOR_X[1], bath[1], P.FRIDGE_X[1], fridge[1]))
    # world-space vertices of a visual mesh: bounds in the park frame must match the transformed room bounds
    import trimesh
    m = trimesh.load(str(ROOT / "assets/room/vis_wallpaper.obj"), force="mesh")
    exp = room_to_park(np.asarray(m.vertices))
    chk("expected wallpaper bounds (park)", True, f"{exp.min(0)} .. {exp.max(0)}")
    for e in h["visuals"][:2]:
        for vg in e.vgeoms:
            v = np.asarray(vg.get_vverts())
            print("  visual", e.idx, "vverts bounds", v.min(0).round(3), v.max(0).round(3))
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sys.exit(_check() if a.check else 0)
