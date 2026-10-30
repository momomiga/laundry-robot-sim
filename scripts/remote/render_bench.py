"""Stage 2 of smoke.sh: which GL backend is used and how fast is a 640x480 render."""
import os
import sys
import time

import numpy as np

import genesis as gs  # imported first: it selects PYOPENGL_PLATFORM

backend = "OSMesa (CPU)" if os.environ.get("PYOPENGL_PLATFORM") == "osmesa" else (
    f"EGL/GL ({os.environ.get('PYOPENGL_PLATFORM', 'default')})")
gs.init(backend=gs.gpu, logging_level="warning")
scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 0.5)))
cam = scene.add_camera(res=(640, 480), pos=(2.0, 2.0, 1.5), lookat=(0, 0, 0.2), fov=40, GUI=False)
scene.build()
scene.step()
cam.render()  # warm-up
n = 20
t0 = time.time()
for _ in range(n):
    scene.step()
    rgb = np.asarray(cam.render()[0])
ms = (time.time() - t0) / n * 1000
print(f"render backend: {backend}")
print(f"{ms:.1f} ms/frame (640x480, step+render), {1000 / ms:.1f} fps, frame {rgb.shape}")
if "OSMesa" in backend:
    print("WARNING: CPU rendering. Recreate the instance with -e NVIDIA_DRIVER_CAPABILITIES=all", file=sys.stderr)
elif ms > 100:
    print("WARNING: slow rendering (>100 ms/frame); check the EGL vendor", file=sys.stderr)
