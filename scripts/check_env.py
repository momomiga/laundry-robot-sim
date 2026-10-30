import os
import sys

import numpy as np
import torch
from PIL import Image

import genesis as gs


def main() -> int:
    print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1
    print("GPU:", torch.cuda.get_device_name(0))

    gs.init(backend=gs.gpu)
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 0.5)))
    cam = scene.add_camera(res=(640, 480), pos=(2.0, 2.0, 1.5), lookat=(0, 0, 0.2), fov=40, GUI=False)
    scene.build()
    for _ in range(100):
        scene.step()

    rgb = cam.render()[0]
    rgb = np.asarray(rgb)
    os.makedirs("outputs", exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8)).save("outputs/check_env.png")
    print("saved outputs/check_env.png", rgb.shape)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
