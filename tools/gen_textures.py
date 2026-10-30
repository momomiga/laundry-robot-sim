#!/usr/bin/env python3
"""Procedural, seamlessly tiling 512x512 textures -> assets/textures/*.png.

No photograph is used: every texture is synthesised from noise (FFT-filtered, hence
periodic) so nothing personal can leak into the repository. Needs numpy + Pillow.
"""
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent.parent / "assets/textures"
N = 512
rng = np.random.default_rng(7)


def noise(scale: float, shape=(N, N), aniso=(1.0, 1.0)) -> np.ndarray:
    """Periodic band-limited noise in [-1, 1]; `scale` = feature size in px."""
    w = rng.standard_normal(shape)
    fy = np.fft.fftfreq(shape[0])[:, None] * aniso[0]
    fx = np.fft.fftfreq(shape[1])[None, :] * aniso[1]
    f = np.sqrt(fx**2 + fy**2) * scale
    g = np.fft.ifft2(np.fft.fft2(w) * np.exp(-(f**2) * 2.0)).real
    return g / (np.abs(g).max() + 1e-9)


def save(name, rgb):
    a = np.clip(rgb, 0, 1)
    Image.fromarray((a * 255).astype(np.uint8), "RGB").save(OUT / name, optimize=True)  # no EXIF
    print("wrote", OUT / name)


def tint(base, v):
    return np.array(base)[None, None, :] * (1.0 + v[..., None])


def wood(name, base, n_planks, plank_contrast, grain=0.10, dark_gaps=True):
    """Planks run along image x (u). Grain streaks are elongated along x."""
    g = noise(60, aniso=(1.0, 0.04)) * grain + noise(8, aniso=(1.0, 0.1)) * grain * 0.5
    ph = N // n_planks
    plank_id = np.arange(N) // ph
    per_plank = rng.uniform(-plank_contrast, plank_contrast, n_planks + 1)
    shift = per_plank[plank_id][:, None] * np.ones((1, N))
    v = g + shift
    img = tint(base, v)
    if dark_gaps:
        gap = (np.arange(N) % ph) < 2
        img[gap] *= 0.55
    save(name, img)


def wallpaper():
    emb = noise(1.6) * 0.5 + noise(5) * 0.5          # fine embossed relief
    y, x = np.mgrid[0:N, 0:N]
    weave = 0.5 * np.sin(2 * np.pi * x / 8) * np.sin(2 * np.pi * y / 8)
    v = 0.035 * emb + 0.012 * weave + 0.02 * noise(40)
    save("wallpaper.png", tint((0.90, 0.87, 0.79), v))


def tile():
    t = 128                                           # 4x4 tiles per texture
    img = tint((0.50, 0.51, 0.52), 0.05 * noise(3) + 0.04 * noise(30))
    # per-tile brightness variation
    var = rng.uniform(-0.03, 0.03, (N // t, N // t))
    img *= (1 + np.kron(var, np.ones((t, t))))[..., None]
    gy, gx = np.mgrid[0:N, 0:N]
    grout = ((gx % t) < 3) | ((gy % t) < 3)
    img[grout] = np.array((0.30, 0.30, 0.30))
    save("tile_gray.png", img)


def frosted():
    v = 0.05 * noise(2) + 0.03 * noise(12)
    img = tint((0.50, 0.55, 0.56), v)
    save("glass_frosted.png", img)


def glass_slit():
    v = 0.02 * noise(20)
    save("glass_slit.png", tint((0.35, 0.55, 0.95), v))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    wood("floor_wood.png", (0.50, 0.28, 0.12), 6, 0.12)
    wood("door_wood_light.png", (0.58, 0.32, 0.12), 1, 0.0, grain=0.10, dark_gaps=False)
    wood("door_wood_dark.png", (0.27, 0.12, 0.06), 1, 0.0, grain=0.12, dark_gaps=False)
    wallpaper()
    tile()
    frosted()
    glass_slit()
