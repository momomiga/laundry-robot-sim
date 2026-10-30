"""Load room + washer + robot (+ dummy laundry) into one Genesis scene.

Headless by default: steps 300 times, saves outputs/scene_top.png and
outputs/scene_persp.png, asserts the robot stayed put and laundry is in the
basket. Use --viewer for the GUI.
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import genesis as gs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import params as P  # noqa: E402

ROBOT_XY = (1.35, 2.0)
ROBOT_YAW = math.pi / 2
LAUNDRY_SIZE = (0.12, 0.12, 0.03)
N_STEPS = 300


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=args.viewer,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.2, -1.5, 2.2), camera_lookat=(1.35, 3.0, 0.5), camera_fov=50
        ),
    )
    scene.add_entity(gs.morphs.URDF(file=str(ROOT / "assets/room/room.urdf"), fixed=True))
    scene.add_entity(
        gs.morphs.URDF(file=str(ROOT / "assets/washer/washer.urdf"), pos=P.WASHER_POS, fixed=True)
    )
    robot = scene.add_entity(
        gs.morphs.URDF(file=str(ROOT / "assets/robot/house_robot.urdf"), fixed=True)
    )

    # basket centre in world (robot frame -> world by yaw)
    bx, by = P.BASKET_CENTER_XY
    c, s = math.cos(ROBOT_YAW), math.sin(ROBOT_YAW)
    basket_c = (ROBOT_XY[0] + c * bx - s * by, ROBOT_XY[1] + s * bx + c * by)
    floor_z = P.BASE_Z1 + 0.01  # basket floor top (urdf: slab 0.255 +- 0.005)
    laundry = []
    for i in range(3):
        e = scene.add_entity(
            gs.morphs.Box(
                size=LAUNDRY_SIZE,
                pos=(basket_c[0] + (i - 1) * 0.02, basket_c[1] + (i - 1) * 0.03,
                     floor_z + 0.05 + i * LAUNDRY_SIZE[2] * 1.5),
            ),
            surface=gs.surfaces.Default(color=(0.9, 0.4 + 0.2 * i, 0.5, 1)),
        )
        laundry.append(e)

    cam_top = scene.add_camera(
        res=(800, 1000), pos=(1.35, 4.2, 14.0), lookat=(1.35, 4.2, 0.0), up=(0, 1, 0),
        fov=45, GUI=False,
    )
    cam_persp = scene.add_camera(
        res=(1000, 700), pos=(0.5, 0.3, 1.6), lookat=(1.35, 2.6, 0.45), fov=60, GUI=False,
    )
    scene.build()

    names = ["base_x_joint", "base_y_joint", "base_yaw_joint", "lift_joint"] + [
        f"joint{i}" for i in range(1, 9)
    ]
    dofs = [robot.get_joint(n).dofs_idx_local[0] for n in names]
    q = np.zeros(len(names))
    q[:3] = [ROBOT_XY[0], ROBOT_XY[1], ROBOT_YAW]
    robot.set_dofs_position(q, dofs)
    robot.control_dofs_position(q, dofs)

    for _ in range(N_STEPS):
        scene.step()

    os.makedirs(ROOT / "outputs", exist_ok=True)
    for cam, name in ((cam_top, "scene_top.png"), (cam_persp, "scene_persp.png")):
        rgb = np.asarray(cam.render()[0]).astype(np.uint8)
        Image.fromarray(rgb).save(ROOT / "outputs" / name)
        print("saved outputs/" + name, rgb.shape)

    qn = robot.get_dofs_position(dofs).cpu().numpy().reshape(-1)
    drift = float(np.linalg.norm(qn[:2] - np.array(ROBOT_XY)))
    print("base drift [m]:", drift)
    assert drift < 0.01, f"robot base drifted {drift:.4f} m"

    hx, hy = P.BASKET_INNER[0] / 2, P.BASKET_INNER[1] / 2
    for i, e in enumerate(laundry):
        p = e.get_pos().cpu().numpy().reshape(-1)
        # basket inner bounds in world (robot yawed by pi/2: x-extent -> world y)
        dx, dy = p[0] - basket_c[0], p[1] - basket_c[1]
        lx, ly = c * dx + s * dy, -s * dx + c * dy  # world -> robot frame offsets
        print(f"laundry{i} pos", np.round(p, 3), "local", round(lx, 3), round(ly, 3))
        assert p[2] > floor_z, f"laundry{i} below basket floor"
        assert abs(lx) < hx and abs(ly) < hy, f"laundry{i} outside basket xy"
    print("OK")

    if args.viewer:
        while scene.viewer.is_alive():
            scene.step()
    return 0


if __name__ == "__main__":
    sys.exit(main())
