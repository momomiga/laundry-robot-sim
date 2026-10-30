# assets
Hand-made models (URDF, units in metres). Dimensions are estimates and live in `tools/params.py`; regenerate with `python3 tools/gen_all.py`, validate with `python3 tools/check_assets.py`.

- `room/room.urdf`: fixed room. x = width (0 = left wall inner face), y = depth (0 = window wall inner face). Floor top at z=0.
- `washer/washer.urdf`: vertical washer, local origin = footprint centre on the floor, front = +x. Place at `pos=(0.30, 7.10, 0)`. Tub collision is a square prism of 0.40 m side so anything that fits a 0.40 m circle drops in; the visual rim is round. Lid is open and fixed.
- `robot/house_robot.urdf`: see `robot/README.md`. Load with `fixed=True` (the root is a virtual anchor; drive `base_x_joint`, `base_y_joint`, `base_yaw_joint`).
