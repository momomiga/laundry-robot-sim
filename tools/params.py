"""Single source of truth for the model dimensions (metres). Edit here, then
run `python tools/gen_room.py`, `gen_washer.py`, `gen_robot.py` (or gen_all).

Room frame: x = width (0 = inner face of left wall), y = depth (0 = inner face of
the window wall of the main room), z up. Values are estimates from a hand-made
sketch of a 1K flat.
"""
import math

# ---------------- room ----------------
WALL_T = 0.10
WALL_H = 2.40
ROOM_W = 2.70
MAIN_Y = (0.0, 4.9)            # main (western-style) room
PARTITION_Y = (4.9, 5.0)       # wall between main room and corridor zone
DOOR_X = (0.6, 1.2)            # opening in the partition (door assumed open); = corridor width
UPPER_Y = (5.0, 8.4)           # corridor zone
LEFT_COL_X = (0.0, 0.6)
CORRIDOR_X = (0.6, 1.2)        # bath-side wall at 1.20: 0.575 clear in front of the fridge (approx. estimate)
RIGHT_COL_X = (1.2, 2.7)
GENKAN_Y = (7.6, 8.4)          # entrance (same floor height)
# Along y everything is anchored at the genkan step edge GENKAN_Y[0] (approx. estimate):
# fridge kitchen-side face 1.27 before the edge, fridge 0.58 wide, washer pan ends at the edge.
FRIDGE_WIDTH = 0.58
FRIDGE_Y = (GENKAN_Y[0] - 1.27, GENKAN_Y[0] - 1.27 + FRIDGE_WIDTH)   # (6.33, 6.91)
KITCHEN_Y = (5.0, FRIDGE_Y[0])  # counter (sink/hob) ends at the fridge, which sits between counter and washer
FRIDGE_X = (0.0, 0.625)        # front x: estimate, pending (left wall -> fridge front approx. only)
FRIDGE_H = 1.20
MICROWAVE_H = 0.28
HANGING_Y = FRIDGE_Y         # white wall cabinet above the fridge (NOT above the washer)
HANGING_X = (0.0, 0.35)
HANGING_Z = (1.75, 2.35)
GENKAN_DROP = 0.12             # entrance floor is lower than the corridor floor
PAN_Y = (GENKAN_Y[0] - 0.67, GENKAN_Y[0])   # washer waterproof pan (visual only), 0.67 long, reaches the step edge
PAN_X_FRONT = 0.64             # pan depth: estimate, pending (front 0.015 ahead of the fridge)
PAN_H = 0.05
BATH_DOOR_Y = (7.1, 8.2)       # folding door on the corridor wall
TOILET_DOOR_Y = (5.85, 6.45)
DOOR_H = 1.95
ROOM_DOOR_W = 0.6              # leaf of the main-room door, hinged at x = DOOR_X[0], swung open into the main room
ROOM_DOOR_H = 2.0
ROOM_DOOR_T = 0.035
CEILING_LIGHT_XY = (0.9, 5.6)   # corridor centre line
KITCHEN_H = 0.85
SHOES_Y = (7.5, 8.4)
SHOES_H = 0.90
CLOSET_Y = (5.0, 5.7)
TOILET_Y = (5.7, 6.6)
BATH_Y = (6.6, 8.4)
INNER_WALL_T = 0.05            # thin walls between closet / toilet / bath

FLOOR_COLORS = {
    "main": (0.78, 0.62, 0.42, 1),
    "corridor": (0.70, 0.72, 0.76, 1),
    "kitchen": (0.85, 0.80, 0.60, 1),
    "washer_bay": (0.55, 0.75, 0.85, 1),
    "genkan": (0.45, 0.42, 0.40, 1),
    "closet": (0.80, 0.70, 0.80, 1),
    "toilet": (0.75, 0.85, 0.75, 1),
    "bath": (0.60, 0.80, 0.80, 1),
}
WALL_COLOR = (0.93, 0.91, 0.84, 1)
FURNITURE_COLOR = (0.55, 0.40, 0.28, 1)

# ---------------- washer (vertical type) ----------------
# Local frame: origin = opening centre on the floor, front (operation side) = +x. Approx. estimate:
# top 0.90 m, 0.51 wide, round opening 0.32, centre 0.225 behind the front edge, lid hinge 0.395 behind the front.
# Total depth, lid size: estimates (approx. only).
WASHER_W = 0.53      # x (front is +x): total depth, estimate
WASHER_D = 0.51      # y: width across
WASHER_H = 0.90
WASHER_OPEN_D = 0.32         # round opening diameter (visual)
WASHER_TUB_SIDE = 0.30       # collision hole / tub: square, inner side (inscribed-ish)
WASHER_OPEN_FRONT = 0.225    # opening centre -> front edge of the top
WASHER_HINGE_FRONT = 0.395   # lid hinge line -> front edge of the top
WASHER_LID = (0.40, 0.45)    # lid size (front-back when closed, across): estimate
WASHER_BODY_CX = WASHER_OPEN_FRONT - WASHER_W / 2   # body centre in the washer frame (x), behind the opening
WASHER_TUB_DEPTH = 0.60
WASHER_FRONT_X = 0.63        # washer front in the room frame (pan front 0.64 - small step): estimate, pending
# placement of the washer origin (= opening centre) in the room frame: centred on the pan along y
WASHER_POS = (WASHER_FRONT_X - WASHER_OPEN_FRONT, sum(PAN_Y) / 2, 0.0)
WASHER_MASS = 30.0

# Park pose of the robot base at the washer (room frame: x, y, yaw [rad]). The arm side (base +y) faces the washer.
# Centred across the corridor between the washer front (x 0.63) and the opposite wall (x 1.20) for the footprint incl.
# the stowed arm (world x extents px-0.242 .. px+0.225 at yaw pi/2: arm side faces the washer);
# opening centre at base x = 0.10 (py = washer y - 0.10).
PARK_POSE = (0.924, 7.165, math.pi / 2)


def washer_in_base(park=PARK_POSE, washer_pos=WASHER_POS):
    """Washer opening centre and yaw in the frame of the parked base: (x, y, yaw [rad]).
    The washer's front (+x in its own frame) points to base -y; its lid stands on the far side (base +y)."""
    px, py, pyaw = park
    dx, dy = washer_pos[0] - px, washer_pos[1] - py
    c, s = math.cos(pyaw), math.sin(pyaw)
    return (c * dx + s * dy, -s * dx + c * dy, -pyaw)


def base_to_washer(x, y, wxy=None, wyaw=None):
    """Point(s) (x, y) of the parked-base frame -> washer frame (origin = opening centre, front = +x). Array-safe."""
    wxy = WASHER_BASE_XY if wxy is None else wxy
    wyaw = WASHER_BASE_YAW if wyaw is None else wyaw
    c, s = math.cos(wyaw), math.sin(wyaw)
    dx, dy = x - wxy[0], y - wxy[1]
    return c * dx + s * dy, -s * dx + c * dy


def washer_to_base(lx, ly, wxy=None, wyaw=None):
    """Point(s) (lx, ly) of the washer frame -> parked-base frame. Array-safe."""
    wxy = WASHER_BASE_XY if wxy is None else wxy
    wyaw = WASHER_BASE_YAW if wyaw is None else wyaw
    c, s = math.cos(wyaw), math.sin(wyaw)
    return wxy[0] + c * lx - s * ly, wxy[1] + s * lx + c * ly


WASHER_BASE_XY = tuple(round(v, 6) for v in washer_in_base()[:2])   # (0.10, ~0.52)
WASHER_BASE_YAW = washer_in_base()[2]                                 # -pi/2
WASHER_BASE_YAW_DEG = math.degrees(WASHER_BASE_YAW)

# ---------------- robot ----------------
BASE_SIZE = (0.45, 0.45)
BASE_Z0, BASE_Z1 = 0.05, 0.25    # platform slab bottom/top
BASE_MASS = 15.0
WHEEL_R, WHEEL_W = 0.05, 0.04
COLUMN_XY = (-0.19, 0.17)
COLUMN_SIZE = 0.05
COLUMN_H = 0.65                  # above the platform top
LIFT_RANGE = (0.0, 0.5)
CARRIAGE_XY = (-0.07, 0.17)      # arm base position (robot frame: +x forward, +y left)
CARRIAGE_THK = 0.03
BASKET_INNER = (0.35, 0.30, 0.20)  # x (forward), y, depth
BASKET_WALL = 0.01
BASKET_CENTER_XY = (0.04, -0.05)

# Basket camera: virtual (no mesh, so it cannot collide with the arm). It is attached to
# base_link via Genesis camera.attach, looking straight down; image up = robot +x.
BASKET_CAM_POS = (BASKET_CENTER_XY[0], BASKET_CENTER_XY[1], 1.0)   # robot frame [m]
BASKET_CAM_RES = (640, 480)
BASKET_CAM_FOV = 60.0            # vertical fov [deg]
