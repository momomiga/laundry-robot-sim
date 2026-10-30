"""Shared robot sensors: the basket camera (looks straight down into the basket).

The camera is a Genesis camera attached to the robot's `base_link` with `Camera.attach`
(virtual mount; no mesh, so it cannot interfere with the arm). Image up = robot +x.
Call `add_basket_camera` before `scene.build()`.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import params as P  # noqa: E402

# camera frame (OpenGL: x right, y up, z back) expressed in base_link: looks -z, image up = +x
_R = np.array([[0.0, 1.0, 0.0],
               [-1.0, 0.0, 0.0],
               [0.0, 0.0, 1.0]])


def add_basket_camera(scene, robot, res=None, fov=None):
    cam = scene.add_camera(res=res or P.BASKET_CAM_RES, pos=(0, 0, 3), lookat=(0, 0, 0),
                           fov=fov or P.BASKET_CAM_FOV, GUI=False)
    T = np.eye(4)
    T[:3, :3] = _R
    T[:3, 3] = P.BASKET_CAM_POS
    cam.attach(robot.get_link("base_link"), T)
    return cam


def get_basket_image(cam):
    """Render the basket camera -> uint8 (H, W, 3)."""
    cam.move_to_attach()
    return np.asarray(cam.render()[0]).astype(np.uint8)


def project_points(cam, pts_world):
    """World points (N,3) -> pixel coordinates (N,2) (u right, v down) for the current pose."""
    pts = np.asarray(pts_world, float).reshape(-1, 3)
    E = np.asarray(cam.extrinsics, float)
    K = np.asarray(cam.intrinsics, float)
    pc = (E[:3, :3] @ pts.T).T + E[:3, 3]
    uv = (K @ pc.T).T
    return uv[:, :2] / uv[:, 2:3]
