"""Textured visual-only overlays for the corridor (OBJ meshes from tools/gen_room_visuals.py).

add_room_visuals(scene) must be called before scene.build(). The meshes have no collision;
room.urdf keeps the plain-colour boxes as collision shapes and fallback visuals, and the
overlays sit a few mm proud of them. Pass ceiling=False to leave the ceiling out of view
(its quad lives in the wallpaper OBJ, so this builds a ceiling-free copy on the fly).
"""
from pathlib import Path

import genesis as gs

ROOT = Path(__file__).resolve().parents[1]
ROOM = ROOT / "assets/room"
TEX = ROOT / "assets/textures"

# obj name -> (texture file, roughness)
MATERIALS = {
    "wood": ("floor_wood.png", 0.45),
    "wallpaper": ("wallpaper.png", 0.9),
    "tile": ("tile_gray.png", 0.35),
    "door_light": ("door_wood_light.png", 0.5),
    "door_dark": ("door_wood_dark.png", 0.5),
    "glass_frosted": ("glass_frosted.png", 0.6),
    "glass_slit": ("glass_slit.png", 0.2),
}


def add_room_visuals(scene, pos=(0.0, 0.0, 0.0), euler=(0.0, 0.0, 0.0), material=None):
    """pos / euler [deg] place the whole overlay (default: room frame); material: optional rigid material
    (e.g. gs.materials.Rigid(needs_coup=False) to keep the entities out of the IPC coupler)."""
    ents = []
    extra = {} if material is None else {"material": material}
    for name, (tex, rough) in MATERIALS.items():
        surf = gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(image_path=str(TEX / tex)),
            roughness=rough,
        )
        ents.append(scene.add_entity(
            gs.morphs.Mesh(file=str(ROOM / f"vis_{name}.obj"), fixed=True, collision=False,
                           convexify=False, decimate=False, pos=tuple(pos), euler=tuple(euler)),
            surface=surf, **extra,
        ))
    return ents
