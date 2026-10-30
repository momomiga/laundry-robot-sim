#!/usr/bin/env python3
"""Generate assets/room/room.urdf from tools/params.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import params as P
from urdf_util import *
import gen_room_visuals

OUT = Path(__file__).resolve().parent.parent / "assets/room/room.urdf"


EXTRA_COLORS = {
    "white": (0.93, 0.93, 0.91, 1),
    "black": (0.06, 0.06, 0.07, 1),
    "steel": (0.65, 0.66, 0.68, 1),
    "frame_wood": (0.60, 0.34, 0.14, 1),
    "dark_wood": (0.25, 0.11, 0.05, 1),
    "door_wood": (0.58, 0.32, 0.12, 1),
    "lattice": (0.90, 0.90, 0.88, 1),
    "lamp": (1.0, 0.93, 0.78, 1),
    "pan": (0.92, 0.93, 0.94, 1),
}


def wall(x0, x1, y0, y1, h=P.WALL_H, z0=0.0, color="wall"):
    return box_shape(((x0 + x1) / 2, (y0 + y1) / 2, z0 + h / 2), (x1 - x0, y1 - y0, h), color=color)


def build() -> str:
    t, W = P.WALL_T, P.ROOM_W
    y_end = P.UPPER_Y[1]
    mats = material("wall", P.WALL_COLOR) + material("furniture", P.FURNITURE_COLOR)
    for k, c in EXTRA_COLORS.items():
        mats += material(k, c)
    for k, c in P.FLOOR_COLORS.items():
        mats += material(f"floor_{k}", c)
    b = inertial(1000.0, (1000, 1000, 1000))
    # collision floor (no visual), top face at z=0
    b += box_shape(((W) / 2, y_end / 2, -0.05), (W + 2 * t, y_end + 2 * t, 0.1), visual=False)

    def floor(name, xr, yr, top=0.0):
        return box_shape(((xr[0] + xr[1]) / 2, (yr[0] + yr[1]) / 2, top - 0.01),
                         (xr[1] - xr[0], yr[1] - yr[0], 0.02), color=f"floor_{name}", collision=False)

    drop = P.GENKAN_DROP

    full = (0.0, W)
    b += floor("main", full, P.MAIN_Y)
    b += floor("main", full, P.PARTITION_Y)
    b += floor("corridor", P.CORRIDOR_X, (P.UPPER_Y[0], P.GENKAN_Y[0]))
    b += floor("genkan", P.CORRIDOR_X, P.GENKAN_Y, -drop)
    b += floor("kitchen", P.LEFT_COL_X, P.KITCHEN_Y)
    b += floor("washer_bay", P.LEFT_COL_X, (P.FRIDGE_Y[0], P.GENKAN_Y[0]))
    b += floor("genkan", P.LEFT_COL_X, (P.GENKAN_Y[0], P.GENKAN_Y[1]), -drop)
    b += box_shape(((P.CORRIDOR_X[0] + P.CORRIDOR_X[1]) / 2, P.GENKAN_Y[0] - 0.005, -drop / 2),
                   (P.CORRIDOR_X[1] - P.CORRIDOR_X[0], 0.01, drop), color="furniture", collision=False)   # riser
    b += floor("closet", P.RIGHT_COL_X, P.CLOSET_Y)
    b += floor("toilet", P.RIGHT_COL_X, P.TOILET_Y)
    b += floor("bath", P.RIGHT_COL_X, P.BATH_Y)

    # outer walls
    lo = dict(z0=-P.GENKAN_DROP, h=P.WALL_H + P.GENKAN_DROP)   # reach down to the lowered entrance
    b += wall(-t, 0, -t, y_end + t, **lo)                # left
    b += wall(W, W + t, -t, y_end + t, **lo)             # right
    b += wall(-t, W + t, -t, 0)                          # window wall (y=0)
    b += wall(-t, W + t, y_end, y_end + t, **lo)         # entrance wall (y=8.4)
    # partition with door opening
    b += wall(0, P.DOOR_X[0], *P.PARTITION_Y)
    b += wall(P.DOOR_X[1], W, *P.PARTITION_Y)
    # right column: closed boxes, wall on corridor side (inside right column)
    b += wall(P.RIGHT_COL_X[0], P.RIGHT_COL_X[0] + t, P.UPPER_Y[0], y_end, **lo)
    it = P.INNER_WALL_T
    for y in (P.CLOSET_Y[1], P.TOILET_Y[1]):
        b += wall(P.RIGHT_COL_X[0] + t, W, y - it / 2, y + it / 2)
    # lintel over the main-room door opening + the (open) door leaf, hinged at x = DOOR_X[0]
    b += wall(P.DOOR_X[0], P.DOOR_X[1], *P.PARTITION_Y, h=P.WALL_H - P.ROOM_DOOR_H, z0=P.ROOM_DOOR_H)
    yh = P.PARTITION_Y[0]
    b += wall(P.DOOR_X[0], P.DOOR_X[0] + P.ROOM_DOOR_T, yh - P.ROOM_DOOR_W, yh, h=P.ROOM_DOOR_H, color="door_wood")

    # right column: door frames on the corridor wall (x = RIGHT_COL_X[0])
    xr = P.RIGHT_COL_X[0]
    fd, fw = 0.03, 0.06
    a, c = P.BATH_DOOR_Y
    zt = P.DOOR_H
    for ya, yb in ((a - fw, a), (c, c + fw)):
        b += box_shape((xr - fd / 2, (ya + yb) / 2, zt / 2), (fd, yb - ya, zt), color="frame_wood", collision=False)
    b += box_shape((xr - fd / 2, (a + c) / 2, zt + fw / 2), (fd, c - a + 2 * fw, fw), color="frame_wood", collision=False)
    b += box_shape((xr - fd / 2, (a + c) / 2, 0.03), (fd, c - a, 0.06), color="frame_wood", collision=False)
    # white lattice of the folding door: 1 vertical centre bar, 2 horizontal bars
    for yy in (a + 0.04, (a + c) / 2, c - 0.04):
        b += box_shape((xr - 0.012, yy, zt / 2), (0.01, 0.025, zt - 0.1), color="lattice", collision=False)
    for zz in (0.65, 1.25):
        b += box_shape((xr - 0.012, (a + c) / 2, zz), (0.01, c - a - 0.06, 0.025), color="lattice", collision=False)
    a, c = P.TOILET_DOOR_Y
    fw2 = 0.04
    for ya, yb in ((a - fw2, a), (c, c + fw2)):
        b += box_shape((xr - 0.007, (ya + yb) / 2, zt / 2), (0.014, yb - ya, zt), color="dark_wood", collision=False)
    b += box_shape((xr - 0.007, (a + c) / 2, zt + fw2 / 2), (0.014, c - a + 2 * fw2, fw2), color="dark_wood", collision=False)
    b += box_shape((xr - 0.012, a + 0.06, 0.95), (0.02, 0.1, 0.02), color="steel", collision=False)   # lever

    # left column: kitchen counter (white body, black top, hob, sink), fridge + microwave, wall cabinet
    lx = P.LEFT_COL_X[1]
    ky0, ky1 = P.KITCHEN_Y
    top = 0.03
    b += box_shape((lx / 2, (ky0 + ky1) / 2, (P.KITCHEN_H - top) / 2), (lx, ky1 - ky0, P.KITCHEN_H - top), color="white")
    b += box_shape((lx / 2, (ky0 + ky1) / 2, P.KITCHEN_H - top / 2), (lx, ky1 - ky0, top), color="black")
    for yc in (ky1 - 0.30,):        # hob: 2 burners
        b += box_shape((lx / 2, yc, P.KITCHEN_H + 0.004), (0.5, 0.52, 0.008), color="steel", collision=False)
        for dy in (-0.13, 0.13):
            b += cyl_shape((lx / 2, yc + dy, P.KITCHEN_H + 0.01), 0.09, 0.004, color="black", collision=False)
    b += box_shape((lx / 2, ky0 + 0.35, P.KITCHEN_H + 0.002), (0.42, 0.45, 0.006), color="steel", collision=False)  # sink
    fx0, fx1 = P.FRIDGE_X
    fy0, fy1 = P.FRIDGE_Y
    b += box_shape(((fx0 + fx1) / 2, (fy0 + fy1) / 2, P.FRIDGE_H / 2), (fx1 - fx0, fy1 - fy0, P.FRIDGE_H), color="black")
    b += box_shape((fx1 - 0.01, (fy0 + fy1) / 2, 0.80), (0.012, 0.006, 0.7), color="steel", collision=False)  # door seam
    b += box_shape((0.30, (fy0 + fy1) / 2, P.FRIDGE_H + P.MICROWAVE_H / 2), (0.45, 0.42, P.MICROWAVE_H), color="white")
    hx0, hx1 = P.HANGING_X
    hy0, hy1 = P.HANGING_Y
    hz0, hz1 = P.HANGING_Z
    b += box_shape(((hx0 + hx1) / 2, (hy0 + hy1) / 2, (hz0 + hz1) / 2), (hx1 - hx0, hy1 - hy0, hz1 - hz0), color="white")
    for dy in (-0.13, 0.13):        # round knobs on the two doors (front face x = hx1, near the centre seam)
        b += cyl_shape((hx1 + 0.015, (hy0 + hy1) / 2 + dy * 0.45, hz0 + 0.08), 0.018, 0.03,
                       rpy=(0, 1.5708, 0), color="steel", collision=False)
    b += cyl_shape((*P.CEILING_LIGHT_XY, P.WALL_H - 0.01), 0.17, 0.02, color="lamp", collision=False)   # ceiling light
    # waterproof pan under the washer (visual only) and a water tap on the wall behind it
    py0, py1 = P.PAN_Y
    b += box_shape((P.PAN_X_FRONT / 2, (py0 + py1) / 2, P.PAN_H / 2), (P.PAN_X_FRONT, py1 - py0, P.PAN_H),
                   color="pan", collision=False)
    b += cyl_shape((0.04, P.WASHER_POS[1] + 0.2, 1.10), 0.012, 0.08, rpy=(0, 1.5708, 0), color="steel", collision=False)
    b += box_shape((P.LEFT_COL_X[1] / 2, sum(P.SHOES_Y) / 2, (P.SHOES_H - P.GENKAN_DROP) / 2),
                   (P.LEFT_COL_X[1], P.SHOES_Y[1] - P.SHOES_Y[0], P.SHOES_H + P.GENKAN_DROP), color="furniture")
    gen_room_visuals.build_all()      # side effect: textured overlay OBJs next to room.urdf
    return robot("room", mats + link("room", b),
                 "generated by tools/gen_room.py from tools/params.py; fixed rigid body; units m")


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build())
    print("wrote", gen_room_visuals.build_all())
    print("wrote", OUT)
