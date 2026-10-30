"""Render the corridor from the entrance.

Outputs: outputs/corridor_view.png (1280x960), outputs/corridor_view_portrait.png (960x1280),
outputs/corridor_view_robot.png (robot parked beside the washer).
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import genesis as gs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))
import params as P  # noqa: E402
from room_visuals import add_room_visuals  # noqa: E402

CAM_POS = (1.0, 8.25, 1.38)         # entrance floor (z=-0.12) + ~1.5 m
CAM_LOOKAT = (0.85, 5.0, 0.95)
PARK = P.PARK_POSE[:2]            # robot base position at the washer (params.py, shared with the IPC test)
YAW = P.PARK_POSE[2]


def make_scene(with_robot, views):
    lx, ly = P.CEILING_LIGHT_XY
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
        vis_options=gs.options.VisOptions(
            show_world_frame=False, ambient_light=(0.42, 0.40, 0.37),
            lights=[
                dict(type="point", pos=(lx, ly, P.WALL_H - 0.08), color=(1.0, 0.80, 0.55), intensity=9.0),
                dict(type="point", pos=(0.9, 7.6, P.WALL_H - 0.3), color=(1.0, 0.82, 0.6), intensity=8.0),
            ],
        ),
    )
    scene.add_entity(gs.morphs.URDF(file=str(ROOT / "assets/room/room.urdf"), fixed=True))
    add_room_visuals(scene)
    scene.add_entity(gs.morphs.URDF(file=str(ROOT / "assets/washer/washer.urdf"), pos=P.WASHER_POS, fixed=True))
    robot = None
    if with_robot:
        robot = scene.add_entity(gs.morphs.URDF(file=str(ROOT / "assets/robot/house_robot.urdf"), fixed=True))
    cams = [scene.add_camera(res=r, pos=p, lookat=la, fov=f, GUI=False) for r, p, la, f in views]
    scene.build()
    if robot is not None:
        names = ["base_x_joint", "base_y_joint", "base_yaw_joint", "lift_joint"] + [f"joint{i}" for i in range(1, 9)]
        dofs = [robot.get_joint(n).dofs_idx_local[0] for n in names]
        q = np.zeros(len(names))
        q[:3] = [*PARK, YAW]
        robot.set_dofs_position(q, dofs)
        robot.control_dofs_position(q, dofs)
    for _ in range(5):
        scene.step()
    return scene, cams


def shot(cam, path):
    rgb = np.asarray(cam.render()[0]).astype(np.uint8)
    Image.fromarray(rgb).save(path)
    print("saved", path)
    return rgb


def main():
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    gs.init(backend=gs.gpu, logging_level="warning")
    views = [((1280, 960), CAM_POS, CAM_LOOKAT, 80.0), ((960, 1280), CAM_POS, CAM_LOOKAT, 95.0)]
    scene, (c1, c2) = make_scene(False, views)
    shot(c1, out / "corridor_view.png")
    shot(c2, out / "corridor_view_portrait.png")
    del scene
    rviews = [((1280, 960), (1.0, 8.25, 1.55), (0.85, 5.0, 0.8), 80.0)]
    scene, (c3,) = make_scene(True, rviews)
    shot(c3, out / "corridor_view_robot.png")


if __name__ == "__main__":
    main()
