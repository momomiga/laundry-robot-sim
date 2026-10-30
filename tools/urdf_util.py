"""Tiny helpers to emit URDF text (standard library only)."""
from __future__ import annotations

import math


def f(v: float) -> str:
    return f"{v:.5f}".rstrip("0").rstrip(".") if v != 0 else "0"


def vec(*v: float) -> str:
    return " ".join(f(x) for x in v)


def box_inertia(m: float, sx: float, sy: float, sz: float):
    return (m / 12 * (sy**2 + sz**2), m / 12 * (sx**2 + sz**2), m / 12 * (sx**2 + sy**2))


def cyl_inertia(m: float, r: float, h: float):
    ixx = m / 12 * (3 * r**2 + h**2)
    return ixx, ixx, m / 2 * r**2


def material(name: str, rgba) -> str:
    return f'  <material name="{name}"><color rgba="{vec(*rgba)}"/></material>\n'


def inertial(mass: float, I, xyz=(0, 0, 0)) -> str:
    return (
        f'    <inertial><origin xyz="{vec(*xyz)}" rpy="0 0 0"/><mass value="{f(mass)}"/>'
        f'<inertia ixx="{I[0]:.6g}" ixy="0" ixz="0" iyy="{I[1]:.6g}" iyz="0" izz="{I[2]:.6g}"/></inertial>\n'
    )


def _geom_box(size):
    return f'<geometry><box size="{vec(*size)}"/></geometry>'


def _geom_cyl(r, h):
    return f'<geometry><cylinder radius="{f(r)}" length="{f(h)}"/></geometry>'


def box_shape(center, size, rpy=(0, 0, 0), color=None, visual=True, collision=True) -> str:
    o = f'<origin xyz="{vec(*center)}" rpy="{vec(*rpy)}"/>'
    s = ""
    if visual:
        m = f'<material name="{color}"/>' if color else ""
        s += f"    <visual>{o}{_geom_box(size)}{m}</visual>\n"
    if collision:
        s += f"    <collision>{o}{_geom_box(size)}</collision>\n"
    return s


def cyl_shape(center, r, h, rpy=(0, 0, 0), color=None, visual=True, collision=True) -> str:
    o = f'<origin xyz="{vec(*center)}" rpy="{vec(*rpy)}"/>'
    s = ""
    if visual:
        m = f'<material name="{color}"/>' if color else ""
        s += f"    <visual>{o}{_geom_cyl(r, h)}{m}</visual>\n"
    if collision:
        s += f"    <collision>{o}{_geom_cyl(r, h)}</collision>\n"
    return s


def link(name: str, body: str) -> str:
    return f'  <link name="{name}">\n{body}  </link>\n'


def joint(name, jtype, parent, child, xyz=(0, 0, 0), rpy=(0, 0, 0), axis=None, limit=None, damping=None) -> str:
    s = f'  <joint name="{name}" type="{jtype}">\n'
    s += f'    <origin xyz="{vec(*xyz)}" rpy="{vec(*rpy)}"/>\n'
    s += f'    <parent link="{parent}"/>\n    <child link="{child}"/>\n'
    if axis:
        s += f'    <axis xyz="{vec(*axis)}"/>\n'
    if limit:
        lo, hi, eff, vel = limit
        s += f'    <limit lower="{f(lo)}" upper="{f(hi)}" effort="{f(eff)}" velocity="{f(vel)}"/>\n'
    if damping is not None:
        s += f'    <dynamics damping="{f(damping)}" friction="0"/>\n'
    s += "  </joint>\n"
    return s


def robot(name: str, content: str, header: str = "") -> str:
    return f'<?xml version="1.0"?>\n<!-- {header} -->\n<robot name="{name}">\n{content}</robot>\n'
