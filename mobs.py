"""
mobs.py  –  3-D mob rendering using real Minecraft skin-sheet UV layout.

Each animal is rendered as a set of axis-aligned boxes (body, head, legs, etc.)
by extracting the correct UV region from the mob's texture sheet.

Minecraft texture sheet layout (64×32 or 64×64):
  Row 0 (top 16px):  head cuboid UVs at (0,0)
  Row 1 (next 16px): body / torso UVs
  Legs, tail, etc. at various offsets documented per-mob below.

We render each face of each box as a textured quad sampled from the correct
region of the sheet, using a dedicated per-mob ModernGL texture.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image
import moderngl

# ─────────────────────────────────────────────────────────────────────────────
_TEX = "textures/assets/minecraft/textures/entity"


class Vec3:
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x); self.y = float(y); self.z = float(z)

    def __sub__(self, o): return Vec3(self.x-o.x, self.y-o.y, self.z-o.z)
    def __add__(self, o): return Vec3(self.x+o.x, self.y+o.y, self.z+o.z)

    def length(self):
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

    def normalized(self):
        l = self.length()
        return Vec3(self.x/l, self.y/l, self.z/l) if l > 1e-9 else Vec3()

    def __repr__(self): return f"Vec3({self.x:.2f},{self.y:.2f},{self.z:.2f})"


# ─────────────────────────────────────────────────────────────────────────────
#  MOB TYPE IDS
# ─────────────────────────────────────────────────────────────────────────────
class MobType:
    PIG      = "pig"
    COW      = "cow"
    CHICKEN  = "chicken"
    SHEEP    = "sheep"
    WOLF     = "wolf"
    ZOMBIE   = "zombie"
    SKELETON = "skeleton"
    SPIDER   = "spider"
    CREEPER  = "creeper"


def _candidate_paths(mob_type: str) -> list[str]:
    n = mob_type
    return [
        f"{_TEX}/{n}/temperate_{n}.png",
        f"{_TEX}/{n}/{n}.png",
        f"{_TEX}/{n}.png",
        f"{_TEX}/{n}s/{n}.png",
        f"{_TEX}/passive/{n}.png",
        f"{_TEX}/hostile/{n}.png",
    ]


MOB_TEXTURES = {t: f"{_TEX}/{t}/{t}.png" for t in [
    MobType.PIG, MobType.COW, MobType.CHICKEN, MobType.SHEEP,
    MobType.WOLF, MobType.ZOMBIE, MobType.SKELETON, MobType.SPIDER, MobType.CREEPER,
]}

# ─────────────────────────────────────────────────────────────────────────────
#  MOB STATS
# ─────────────────────────────────────────────────────────────────────────────
MOB_DATA = {
    MobType.PIG:      {"health": 10, "speed": 2.0, "damage": 0,  "hostile": False, "w": 0.9,  "h": 0.9},
    MobType.COW:      {"health": 10, "speed": 2.0, "damage": 0,  "hostile": False, "w": 1.0,  "h": 1.4},
    MobType.CHICKEN:  {"health":  4, "speed": 2.5, "damage": 0,  "hostile": False, "w": 0.6,  "h": 0.8},
    MobType.SHEEP:    {"health":  8, "speed": 2.0, "damage": 0,  "hostile": False, "w": 0.9,  "h": 1.1},
    MobType.WOLF:     {"health":  8, "speed": 4.0, "damage": 2,  "hostile": False, "w": 0.8,  "h": 0.9},
    MobType.ZOMBIE:   {"health": 20, "speed": 2.0, "damage": 3,  "hostile": True,  "w": 0.8,  "h": 1.9},
    MobType.SKELETON: {"health": 20, "speed": 2.5, "damage": 4,  "hostile": True,  "w": 0.7,  "h": 1.9},
    MobType.SPIDER:   {"health": 16, "speed": 3.5, "damage": 2,  "hostile": True,  "w": 1.4,  "h": 0.9},
    MobType.CREEPER:  {"health": 20, "speed": 2.0, "damage": 25, "hostile": True,  "w": 0.7,  "h": 1.7},
}

MOB_DROPS = {
    MobType.PIG:      {"raw_porkchop": (1, 3)},
    MobType.COW:      {"raw_beef": (1, 3), "leather": (0, 2)},
    MobType.CHICKEN:  {"raw_chicken": (1, 2), "feather": (0, 2)},
    MobType.SHEEP:    {"raw_mutton": (1, 2), "white_wool": (1, 3)},
    MobType.WOLF:     {},
    MobType.ZOMBIE:   {"rotten_flesh": (0, 2)},
    MobType.SKELETON: {"bone": (0, 2), "arrow": (0, 2)},
    MobType.SPIDER:   {"string": (0, 2), "spider_eye": (0, 1)},
    MobType.CREEPER:  {"gunpowder": (0, 2)},
}

_FALLBACK_COLOR = {
    MobType.PIG:      (255, 182, 193, 255),
    MobType.COW:      ( 80,  50,  30, 255),
    MobType.CHICKEN:  (240, 230, 210, 255),
    MobType.SHEEP:    (210, 210, 210, 255),
    MobType.WOLF:     (110, 100,  90, 255),
    MobType.ZOMBIE:   ( 60, 140,  60, 255),
    MobType.SKELETON: (200, 200, 200, 255),
    MobType.SPIDER:   ( 50,  30,  30, 255),
    MobType.CREEPER:  ( 30, 160,  30, 255),
}

# ─────────────────────────────────────────────────────────────────────────────
#  3-D BOX MODEL DEFINITIONS
#
#  Each mob is described as a list of "parts" (axis-aligned boxes).
#  Each part has:
#    offset (x, y, z) relative to mob foot-centre in MODEL space (y-up, +z forward)
#    size   (w, h, d) in blocks × a scale factor
#    uv     (u, v)    pixel offset into the 64×32 (or 64×64) sheet for this box
#    sheet_w, sheet_h  texture sheet dimensions (to normalise UVs)
#
#  Minecraft UV convention for a box of pixel-size (px, py, pz):
#    The 6 faces are laid out as follows (starting from uv_offset):
#
#     col offset:  0     pz    pz+px  2pz+px
#     row offset:       (top/bottom strip)
#        +0        [back][top ][ front][ bottom] ... but this isn't quite right
#
#  Actual standard Minecraft box UV layout from (ox, oy):
#    top:    x=ox+pz,      y=oy,       w=px,  h=pz
#    bottom: x=ox+pz+px,   y=oy,       w=px,  h=pz
#    front:  x=ox+pz,      y=oy+pz,    w=px,  h=py
#    back:   x=ox+2pz+px,  y=oy+pz,    w=px,  h=py
#    right:  x=ox,         y=oy+pz,    w=pz,  h=py
#    left:   x=ox+pz+px,   y=oy+pz,    w=pz,  h=py
#
#  We store parts as (ox, oy, px, py, pz, sx, sy, sz, rel_x, rel_y, rel_z)
#  where (px,py,pz) are PIXEL dims, (sx,sy,sz) are WORLD sizes in blocks,
#  (rel_x,rel_y,rel_z) is the part's base position relative to foot-centre.
# ─────────────────────────────────────────────────────────────────────────────

# Sheet sizes per mob
_SHEET = {
    MobType.PIG:      (64, 32),
    MobType.COW:      (64, 32),
    MobType.CHICKEN:  (64, 32),
    MobType.SHEEP:    (64, 32),
    MobType.WOLF:     (64, 32),
    MobType.ZOMBIE:   (64, 64),
    MobType.SKELETON: (64, 64),
    MobType.SPIDER:   (64, 32),
    MobType.CREEPER:  (64, 64),
}

# Parts: (uv_ox, uv_oy, px, py, pz, world_w, world_h, world_d, foot_rel_x, foot_rel_y, foot_rel_z)
# foot_rel_y=0 is at the feet.  All dims in blocks (1 block = 1 world unit).
_PARTS = {

    MobType.PIG: [
        # body
        (28, 8,  8, 8, 4,   0.9, 0.55, 0.6,  -0.45, 0.35, -0.30),
        # head (slightly forward and up)
        ( 0, 0,  8, 8, 8,   0.55, 0.55, 0.55, -0.275, 0.50, 0.10),
        # front-left leg
        ( 0, 16, 4, 6, 4,   0.20, 0.35, 0.20, -0.30, 0.0, -0.15),
        # front-right leg
        ( 0, 16, 4, 6, 4,   0.20, 0.35, 0.20,  0.10, 0.0, -0.15),
        # back-left leg
        ( 0, 16, 4, 6, 4,   0.20, 0.35, 0.20, -0.30, 0.0,  0.15),
        # back-right leg
        ( 0, 16, 4, 6, 4,   0.20, 0.35, 0.20,  0.10, 0.0,  0.15),
    ],

    MobType.COW: [
        # body
        (18, 4,  6, 14, 10,  1.05, 0.85, 0.50,  -0.525, 0.45, -0.25),
        # head
        ( 0, 0,  8, 8,  8,   0.60, 0.60, 0.60,  -0.30,  0.70,  0.10),
        # front-left leg
        ( 0, 16, 4, 12, 4,   0.22, 0.45, 0.22,  -0.38,  0.0,  -0.18),
        # front-right leg
        ( 0, 16, 4, 12, 4,   0.22, 0.45, 0.22,   0.16,  0.0,  -0.18),
        # back-left leg
        ( 0, 16, 4, 12, 4,   0.22, 0.45, 0.22,  -0.38,  0.0,   0.18),
        # back-right leg
        ( 0, 16, 4, 12, 4,   0.22, 0.45, 0.22,   0.16,  0.0,   0.18),
    ],

    MobType.CHICKEN: [
        # body
        ( 0, 16, 8, 8, 6,    0.55, 0.50, 0.40,  -0.275, 0.28, -0.20),
        # head
        ( 0,  0, 8, 6, 6,    0.36, 0.36, 0.36,  -0.18,  0.58,  0.08),
        # left wing
        (24, 13, 8, 8, 4,    0.18, 0.42, 0.10,  -0.36,  0.28, -0.05),
        # right wing
        (24, 13, 8, 8, 4,    0.18, 0.42, 0.10,   0.18,  0.28, -0.05),
        # left leg
        (26,  0, 3, 5, 3,    0.14, 0.28, 0.14,  -0.16,  0.0,  -0.05),
        # right leg
        (26,  0, 3, 5, 3,    0.14, 0.28, 0.14,   0.02,  0.0,  -0.05),
    ],

    MobType.SHEEP: [
        # wool body (slightly bigger than cow body)
        (28,  8, 8, 10, 4,   0.95, 0.75, 0.60,  -0.475, 0.40, -0.30),
        # head
        ( 0,  0, 8,  6, 6,   0.52, 0.52, 0.52,  -0.26,  0.65,  0.08),
        # front-left leg
        ( 0, 16, 4,  6, 4,   0.22, 0.38, 0.22,  -0.32,  0.0,  -0.15),
        # front-right leg
        ( 0, 16, 4,  6, 4,   0.22, 0.38, 0.22,   0.10,  0.0,  -0.15),
        # back-left leg
        ( 0, 16, 4,  6, 4,   0.22, 0.38, 0.22,  -0.32,  0.0,   0.15),
        # back-right leg
        ( 0, 16, 4,  6, 4,   0.22, 0.38, 0.22,   0.10,  0.0,   0.15),
    ],

    MobType.WOLF: [
        # body
        (18,  5, 9, 9, 6,    0.80, 0.60, 0.44,  -0.40, 0.40, -0.22),
        # head
        ( 0,  0, 6, 6, 6,    0.50, 0.50, 0.50,  -0.25, 0.60,  0.12),
        # tail
        ( 9, 18, 4, 8, 4,    0.16, 0.42, 0.16,  -0.08, 0.40, -0.30),
        # front-left leg
        ( 0, 18, 4, 6, 4,    0.18, 0.40, 0.18,  -0.27, 0.0,  -0.14),
        # front-right leg
        ( 0, 18, 4, 6, 4,    0.18, 0.40, 0.18,   0.09, 0.0,  -0.14),
        # back-left leg
        ( 0, 18, 4, 6, 4,    0.18, 0.40, 0.18,  -0.27, 0.0,   0.14),
        # back-right leg
        ( 0, 18, 4, 6, 4,    0.18, 0.40, 0.18,   0.09, 0.0,   0.14),
    ],

    MobType.ZOMBIE: [
        # torso
        (16,  16, 8, 12, 4,   0.62, 0.75, 0.31,  -0.31, 0.75, -0.155),
        # head
        ( 0,   0, 8,  8, 8,   0.62, 0.62, 0.62,  -0.31, 1.42,  0.0),
        # left arm
        (40,  16, 4, 12, 4,   0.28, 0.75, 0.28,  -0.60, 0.75, -0.14),
        # right arm
        (40,  16, 4, 12, 4,   0.28, 0.75, 0.28,   0.31, 0.75, -0.14),
        # left leg
        ( 0,  16, 4, 12, 4,   0.28, 0.75, 0.28,  -0.28, 0.0,  -0.14),
        # right leg
        ( 0,  16, 4, 12, 4,   0.28, 0.75, 0.28,   0.00, 0.0,  -0.14),
    ],

    MobType.SKELETON: [
        # torso (thinner)
        (16, 16, 8, 12, 4,    0.50, 0.70, 0.25,  -0.25, 0.70, -0.125),
        # head
        ( 0,  0, 8,  8, 8,    0.55, 0.55, 0.55,  -0.275, 1.38, 0.0),
        # left arm
        (40, 16, 4, 12, 4,    0.22, 0.70, 0.22,  -0.50, 0.70, -0.11),
        # right arm
        (40, 16, 4, 12, 4,    0.22, 0.70, 0.22,   0.28, 0.70, -0.11),
        # left leg
        ( 0, 16, 4, 12, 4,    0.22, 0.70, 0.22,  -0.22, 0.0,  -0.11),
        # right leg
        ( 0, 16, 4, 12, 4,    0.22, 0.70, 0.22,   0.00, 0.0,  -0.11),
    ],

    MobType.SPIDER: [
        # abdomen (rear)
        (28, 15, 8, 8, 12,   0.80, 0.60, 1.0,  -0.40, 0.25, -0.60),
        # thorax (front)
        ( 0, 16, 8, 8,  6,   0.60, 0.55, 0.50,  -0.30, 0.20, 0.10),
        # head
        ( 0,  0, 8, 8,  8,   0.50, 0.50, 0.50,  -0.25, 0.25, 0.38),
        # leg FL
        (18,  0, 8, 4,  4,   0.50, 0.18, 0.18,  -0.70, 0.22,-0.08),
        # leg FR
        (18,  0, 8, 4,  4,   0.50, 0.18, 0.18,   0.20, 0.22,-0.08),
        # leg ML
        (18,  0, 8, 4,  4,   0.50, 0.18, 0.18,  -0.70, 0.22, 0.10),
        # leg MR
        (18,  0, 8, 4,  4,   0.50, 0.18, 0.18,   0.20, 0.22, 0.10),
        # leg BL
        (18,  0, 8, 4,  4,   0.50, 0.18, 0.18,  -0.70, 0.22, 0.28),
        # leg BR
        (18,  0, 8, 4,  4,   0.50, 0.18, 0.18,   0.20, 0.22, 0.28),
    ],

    MobType.CREEPER: [
        # body
        (16, 16, 8, 12, 4,   0.55, 0.75, 0.30,  -0.275, 0.75, -0.15),
        # head
        ( 0,  0, 8,  8, 8,   0.65, 0.65, 0.65,  -0.325, 1.46, 0.0),
        # front-left leg
        ( 0, 16, 4,  6, 4,   0.22, 0.38, 0.22,  -0.28, 0.0, -0.12),
        # front-right leg
        ( 0, 16, 4,  6, 4,   0.22, 0.38, 0.22,   0.06, 0.0, -0.12),
        # back-left leg
        ( 0, 16, 4,  6, 4,   0.22, 0.38, 0.22,  -0.28, 0.0,  0.12),
        # back-right leg
        ( 0, 16, 4,  6, 4,   0.22, 0.38, 0.22,   0.06, 0.0,  0.12),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
#  MOB DATACLASS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Mob:
    type:          str
    pos:           Vec3
    health:        float
    speed:         float
    damage:        float
    hostile:       bool
    width:         float
    height:        float
    yaw:           float = 0.0      # facing direction (degrees)
    anim_time:     float = 0.0      # walk cycle phase
    ai_timer:      float = 0.0
    attack_cd:     float = 0.0
    wander_target: Optional[Vec3] = None


# ─────────────────────────────────────────────────────────────────────────────
#  GLSL – per-vertex coloured + textured box face
# ─────────────────────────────────────────────────────────────────────────────
_BOX_VERT = """
#version 330 core
in vec3 in_pos;
in vec2 in_uv;
in float in_light;
uniform mat4 u_mvp;
out vec2 v_uv;
out float v_light;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_uv    = in_uv;
    v_light = in_light;
}
"""

_BOX_FRAG = """
#version 330 core
in vec2 v_uv;
in float v_light;
uniform sampler2D u_tex;
out vec4 frag_color;
void main() {
    vec4 c = texture(u_tex, v_uv);
    if (c.a < 0.05) discard;
    frag_color = vec4(c.rgb * v_light, c.a);
}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  UV HELPERS  –  extract one face's UV rect from the standard Minecraft layout
# ─────────────────────────────────────────────────────────────────────────────
def _face_uvs(ox: int, oy: int, px: int, py: int, pz: int,
              sw: int, sh: int, face: int):
    """
    Returns (u0,v0, u1,v1) normalised UVs for face `face` of a box.

    Face indices:
      0=top  1=bottom  2=front(+Z)  3=back(-Z)  4=right(+X)  5=left(-X)

    Standard Minecraft UV block layout starting at pixel (ox,oy):
      top:    rect(ox+pz,        oy,       px, pz)
      bottom: rect(ox+pz+px,     oy,       px, pz)
      right:  rect(ox,           oy+pz,    pz, py)
      front:  rect(ox+pz,        oy+pz,    px, py)
      left:   rect(ox+pz+px,     oy+pz,    pz, py)
      back:   rect(ox+2*pz+px,   oy+pz,    px, py)

    V is flipped (1-v) because PIL loads images top-left but OpenGL
    textures have origin at bottom-left.
    """
    rects = [
        (ox+pz,       oy,      px, pz),   # 0 top
        (ox+pz+px,    oy,      px, pz),   # 1 bottom
        (ox+pz,       oy+pz,   px, py),   # 2 front
        (ox+2*pz+px,  oy+pz,   px, py),   # 3 back
        (ox,          oy+pz,   pz, py),   # 4 right
        (ox+pz+px,    oy+pz,   pz, py),   # 5 left
    ]
    rx, ry, rw, rh = rects[face]
    u0 =  rx / sw
    u1 = (rx + rw) / sw
    # Flip V: PIL top-left → OpenGL bottom-left
    v0 = 1.0 - (ry + rh) / sh
    v1 = 1.0 -  ry / sh
    return u0, v0, u1, v1


def _box_quads(ox: int, oy: int, px: int, py: int, pz: int,
               sw: int, sh: int,
               wx: float, wy: float, wz: float,
               rx: float, ry: float, rz: float,
               yaw_rad: float) -> list:
    """
    Generate 6 quads (each a list of 6 vertices) for one box part.

    (rx, ry, rz) = bottom-left-back corner of the box in local mob space.
    The box is axis-aligned in LOCAL space then rotated by yaw around Y.

    Each vertex: [x, y, z, u, v, light]
    """
    hw = wx * 0.5   # not used for corner calc – we use explicit corners
    # 8 corners in local space (no yaw yet):
    #   x: rx … rx+wx,  y: ry … ry+wy,  z: rz … rz+wz
    x0, x1 = rx,      rx + wx
    y0, y1 = ry,      ry + wy
    z0, z1 = rz,      rz + wz

    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)

    def rot(lx, lz):
        return lx * cos_y - lz * sin_y, lx * sin_y + lz * cos_y

    def v(lx, ly, lz, u, vv, light):
        rx2, rz2 = rot(lx, lz)
        return [rx2, ly, rz2, u, vv, light]

    quads = []
    face_lights = [1.0, 0.55, 0.85, 0.85, 0.75, 0.75]  # top,bot,front,back,right,left

    # With V-flipped UVs: (u0,v0) = bottom-left of texture region, (u1,v1) = top-right
    # face 0: top  (y = y1, xz plane) — looking down, +X=right, +Z=down in texture
    u0,v0,u1,v1 = _face_uvs(ox,oy,px,py,pz, sw,sh, 0)
    lt = face_lights[0]
    quads.append([
        v(x0, y1, z1, u0,v0, lt), v(x1, y1, z1, u1,v0, lt),
        v(x1, y1, z0, u1,v1, lt), v(x0, y1, z1, u0,v0, lt),
        v(x1, y1, z0, u1,v1, lt), v(x0, y1, z0, u0,v1, lt),
    ])
    # face 1: bottom (y = y0)
    u0,v0,u1,v1 = _face_uvs(ox,oy,px,py,pz, sw,sh, 1)
    lt = face_lights[1]
    quads.append([
        v(x0, y0, z0, u0,v0, lt), v(x1, y0, z0, u1,v0, lt),
        v(x1, y0, z1, u1,v1, lt), v(x0, y0, z0, u0,v0, lt),
        v(x1, y0, z1, u1,v1, lt), v(x0, y0, z1, u0,v1, lt),
    ])
    # face 2: front (+Z, z = z1)
    u0,v0,u1,v1 = _face_uvs(ox,oy,px,py,pz, sw,sh, 2)
    lt = face_lights[2]
    quads.append([
        v(x0, y0, z1, u0,v0, lt), v(x1, y0, z1, u1,v0, lt),
        v(x1, y1, z1, u1,v1, lt), v(x0, y0, z1, u0,v0, lt),
        v(x1, y1, z1, u1,v1, lt), v(x0, y1, z1, u0,v1, lt),
    ])
    # face 3: back (-Z, z = z0)
    u0,v0,u1,v1 = _face_uvs(ox,oy,px,py,pz, sw,sh, 3)
    lt = face_lights[3]
    quads.append([
        v(x1, y0, z0, u0,v0, lt), v(x0, y0, z0, u1,v0, lt),
        v(x0, y1, z0, u1,v1, lt), v(x1, y0, z0, u0,v0, lt),
        v(x0, y1, z0, u1,v1, lt), v(x1, y1, z0, u0,v1, lt),
    ])
    # face 4: right (+X, x = x1)
    u0,v0,u1,v1 = _face_uvs(ox,oy,px,py,pz, sw,sh, 4)
    lt = face_lights[4]
    quads.append([
        v(x1, y0, z0, u0,v0, lt), v(x1, y0, z1, u1,v0, lt),
        v(x1, y1, z1, u1,v1, lt), v(x1, y0, z0, u0,v0, lt),
        v(x1, y1, z1, u1,v1, lt), v(x1, y1, z0, u0,v1, lt),
    ])
    # face 5: left (-X, x = x0)
    u0,v0,u1,v1 = _face_uvs(ox,oy,px,py,pz, sw,sh, 5)
    lt = face_lights[5]
    quads.append([
        v(x0, y0, z1, u0,v0, lt), v(x0, y0, z0, u1,v0, lt),
        v(x0, y1, z0, u1,v1, lt), v(x0, y0, z1, u0,v0, lt),
        v(x0, y1, z0, u1,v1, lt), v(x0, y1, z1, u0,v1, lt),
    ])
    return quads


# ─────────────────────────────────────────────────────────────────────────────
#  MOB RENDERER
# ─────────────────────────────────────────────────────────────────────────────
class MobRenderer:
    def __init__(self):
        self._ctx  = None
        self._prog = None
        self._textures: dict[str, moderngl.Texture] = {}
        self._vbo  = None
        self._vao  = None

    def init(self, ctx: moderngl.Context) -> None:
        self._ctx  = ctx
        self._prog = ctx.program(vertex_shader=_BOX_VERT, fragment_shader=_BOX_FRAG)
        # Each vertex: 3f pos + 2f uv + 1f light = 6 floats = 24 bytes
        # Max ~200 verts per mob × 20 mobs = 4000 verts
        self._vbo = ctx.buffer(reserve=4000 * 6 * 4)
        self._vao = ctx.vertex_array(self._prog,
                                      [(self._vbo, '3f 2f 1f', 'in_pos', 'in_uv', 'in_light')])
        for mob_type in MOB_DATA:
            self._load_tex(mob_type)

    def _make_fallback_tex(self, mob_type: str) -> moderngl.Texture:
        c    = _FALLBACK_COLOR.get(mob_type, (180, 60, 60, 255))
        dark = (max(0, c[0]-60), max(0, c[1]-60), max(0, c[2]-60), 255)
        sz   = 64
        data = bytearray()
        for y in range(sz):
            for x in range(sz):
                col = c if (x // 8 + y // 8) % 2 == 0 else dark
                data.extend(col)
        tex = self._ctx.texture((sz, sz), 4, bytes(data))
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return tex

    def _load_tex(self, mob_type: str) -> None:
        if mob_type in self._textures:
            return
        for path in _candidate_paths(mob_type):
            if not os.path.exists(path):
                continue
            try:
                img = Image.open(path).convert("RGBA")
                # Ensure the sheet is at least 64×32; scale up if tiny
                if img.width < 32:
                    scale = 64 // img.width
                    img = img.resize((img.width * scale, img.height * scale),
                                     resample=Image.NEAREST)
                # Flip vertically: PIL is top-left origin, OpenGL is bottom-left
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                tex = self._ctx.texture(img.size, 4, img.tobytes())
                tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
                self._textures[mob_type] = tex
                print(f"[Mobs3D] ✔ {mob_type}: {path}  ({img.size})")
                return
            except Exception as exc:
                print(f"[Mobs3D] ✘ {mob_type} @ {path}: {exc}")
        self._textures[mob_type] = self._make_fallback_tex(mob_type)
        print(f"[Mobs3D] ⚠ {mob_type}: using colour fallback")

    # ── build & upload vertex data for a single mob ───────────────────────────
    def _build_mob_verts(self, mob: Mob) -> np.ndarray:
        parts   = _PARTS.get(mob.type, [])
        sw, sh  = _SHEET.get(mob.type, (64, 32))
        yaw_rad = math.radians(mob.yaw)

        verts = []
        for part in parts:
            (uv_ox, uv_oy, px, py, pz,
             world_w, world_h, world_d,
             rel_x, rel_y, rel_z) = part

            # Apply simple leg-swing animation using anim_time
            extra_yaw = 0.0
            # (for simplicity we don't rotate individual parts — just the whole mob)

            quads = _box_quads(
                uv_ox, uv_oy, px, py, pz, sw, sh,
                world_w, world_h, world_d,
                rel_x, rel_y, rel_z,
                yaw_rad,
            )
            for quad in quads:
                for vtx in quad:
                    # translate from local mob space to world space
                    lx, ly, lz, u, v, light = vtx
                    verts.extend([
                        mob.pos.x + lx,
                        mob.pos.y + ly,
                        mob.pos.z + lz,
                        u, v, light,
                    ])
        return np.array(verts, dtype=np.float32)

    # ── draw all mobs ─────────────────────────────────────────────────────────
    def draw(self, mobs: list[Mob], mvp: np.ndarray) -> None:
        if self._ctx is None or not mobs:
            return

        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.enable(moderngl.DEPTH_TEST)

        self._prog['u_mvp'].write(mvp.T.astype(np.float32).tobytes())

        for mob in mobs:
            tex = self._textures.get(mob.type)
            if tex is None:
                continue
            vdata = self._build_mob_verts(mob)
            if vdata.size == 0:
                continue
            # Resize VBO if needed
            needed = vdata.nbytes
            if needed > self._vbo.size:
                self._vbo.release()
                self._vao.release()
                self._vbo = self._ctx.buffer(reserve=needed * 2)
                self._vao = self._ctx.vertex_array(
                    self._prog,
                    [(self._vbo, '3f 2f 1f', 'in_pos', 'in_uv', 'in_light')],
                )
            self._vbo.write(vdata.tobytes())
            tex.use(0)
            self._prog['u_tex'].value = 0
            n_verts = len(vdata) // 6
            self._vao.render(moderngl.TRIANGLES, vertices=n_verts)


# ─────────────────────────────────────────────────────────────────────────────
#  MOB MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class MobManager:
    MAX_MOBS               = 20
    PASSIVE_SPAWN_INTERVAL = 20.0
    HOSTILE_SPAWN_INTERVAL = 15.0
    DETECTION_RADIUS       = 24.0
    ATTACK_RADIUS          = 2.0

    def __init__(self):
        self.mobs: list[Mob]  = []
        self._passive_timer   = 0.0
        self._hostile_timer   = 0.0
        self._renderer        = MobRenderer()

    def init_gl(self, ctx: moderngl.Context) -> None:
        self._renderer.init(ctx)

    def spawn_mob(self, mob_type: str, pos: Vec3) -> Mob:
        d   = MOB_DATA[mob_type]
        mob = Mob(
            type    = mob_type,
            pos     = Vec3(pos.x, pos.y, pos.z),
            health  = d["health"],
            speed   = d["speed"],
            damage  = d["damage"],
            hostile = d["hostile"],
            width   = d["w"],
            height  = d["h"],
            yaw     = random.uniform(0, 360),
        )
        self.mobs.append(mob)
        return mob

    def spawn_initial_passive(self, player_pos: Vec3, world, count: int = 6) -> None:
        spawned, attempts = 0, 0
        while spawned < count and attempts < 40:
            attempts += 1
            angle = random.uniform(0, math.tau)
            dist  = random.uniform(8, 20)
            wx = int(player_pos.x + dist * math.cos(angle))
            wz = int(player_pos.z + dist * math.sin(angle))
            wy = self._surface_y(wx, wz, world) + 1
            if not self._has_sky(wx, wy, wz, world):
                continue
            mob_type = random.choice([MobType.PIG, MobType.COW,
                                       MobType.CHICKEN, MobType.SHEEP])
            self.spawn_mob(mob_type, Vec3(wx, wy, wz))
            spawned += 1
        print(f"🐄 Spawned {spawned} initial passive mobs")

    # ── helpers ───────────────────────────────────────────────────────────────
    def _surface_y(self, wx, wz, world) -> int:
        return world.surface_y(wx, wz) if world else 35

    def _has_sky(self, wx, wy, wz, world) -> bool:
        return world.has_sky_access(wx, wy, wz) if world else True

    def _try_spawn_passive(self, player_pos: Vec3, world) -> None:
        if len(self.mobs) >= self.MAX_MOBS:
            return
        angle = random.uniform(0, math.tau)
        dist  = random.uniform(16, 40)
        wx = int(player_pos.x + dist * math.cos(angle))
        wz = int(player_pos.z + dist * math.sin(angle))
        wy = self._surface_y(wx, wz, world) + 1
        if not self._has_sky(wx, wy, wz, world):
            return
        mob_type = random.choice([MobType.PIG, MobType.COW,
                                   MobType.CHICKEN, MobType.SHEEP])
        self.spawn_mob(mob_type, Vec3(wx, wy, wz))

    def _try_spawn_hostile(self, player_pos: Vec3, world, time_of_day: float) -> None:
        if len(self.mobs) >= self.MAX_MOBS:
            return
        angle = random.uniform(0, math.tau)
        dist  = random.uniform(16, 45)
        wx = int(player_pos.x + dist * math.cos(angle))
        wz = int(player_pos.z + dist * math.sin(angle))
        wy = self._surface_y(wx, wz, world) + 1
        is_night = 13_000 <= time_of_day <= 23_000
        in_cave  = not self._has_sky(wx, wy, wz, world)
        if not is_night and not in_cave:
            return
        mob_type = random.choice([MobType.ZOMBIE, MobType.SKELETON,
                                   MobType.SPIDER, MobType.CREEPER])
        self.spawn_mob(mob_type, Vec3(wx, wy, wz))
        print(f"🌙 Hostile spawn: {mob_type} at ({wx},{wy},{wz})")

    # ── AI update ─────────────────────────────────────────────────────────────
    def update_mobs(self, dt: float, player_pos: Vec3,
                    world=None, player=None, time_of_day: float = 0.0) -> None:
        for mob in self.mobs[:]:
            if mob.health <= 0:
                self._remove_mob(mob); continue
            mob.anim_time += dt
            mob.ai_timer  += dt
            if mob.attack_cd > 0:
                mob.attack_cd -= dt
            if mob.ai_timer >= 1.0:
                mob.ai_timer = 0.0
                dmg = self._run_ai(mob, player_pos, world)
                if dmg and player is not None:
                    player.take_damage(dmg)
            if world is not None:
                ground_y = self._surface_y(int(mob.pos.x), int(mob.pos.z), world) + 1
                mob.pos.y = float(ground_y)

        self._passive_timer += dt
        if self._passive_timer >= self.PASSIVE_SPAWN_INTERVAL:
            self._passive_timer = 0.0
            self._try_spawn_passive(player_pos, world)

        self._hostile_timer += dt
        if self._hostile_timer >= self.HOSTILE_SPAWN_INTERVAL:
            self._hostile_timer = 0.0
            for _ in range(random.randint(1, 2)):
                self._try_spawn_hostile(player_pos, world, time_of_day)

    def _run_ai(self, mob: Mob, player_pos: Vec3, world) -> Optional[float]:
        if mob.hostile:
            diff = Vec3(player_pos.x - mob.pos.x, 0, player_pos.z - mob.pos.z)
            dist = diff.length()
            if dist < self.DETECTION_RADIUS:
                if dist > self.ATTACK_RADIUS:
                    d = diff.normalized()
                    mob.pos.x += d.x * 0.4
                    mob.pos.z += d.z * 0.4
                    # Face the player
                    mob.yaw = math.degrees(math.atan2(d.x, d.z))
                if dist <= self.ATTACK_RADIUS and mob.attack_cd <= 0:
                    mob.attack_cd = 1.0
                    return mob.damage
        else:
            if mob.wander_target is None or random.random() < 0.15:
                mob.wander_target = Vec3(
                    mob.pos.x + random.uniform(-6, 6),
                    mob.pos.y,
                    mob.pos.z + random.uniform(-6, 6),
                )
            diff = Vec3(mob.wander_target.x - mob.pos.x, 0,
                        mob.wander_target.z - mob.pos.z)
            if diff.length() > 0.5:
                d = diff.normalized()
                mob.pos.x += d.x * 0.25
                mob.pos.z += d.z * 0.25
                mob.yaw    = math.degrees(math.atan2(d.x, d.z))
            else:
                mob.wander_target = None
        return None

    # ── draw (called from main GameWindow.on_draw) ────────────────────────────
    def draw_mobs(self, mvp: np.ndarray,
                  player_yaw: float = 0.0, player_pitch: float = 0.0) -> None:
        """Draw all mobs. mvp is the combined projection × view matrix."""
        if not self.mobs:
            return
        self._renderer.draw(self.mobs, mvp)

    # ── damage / drops ────────────────────────────────────────────────────────
    def damage_mob(self, mob: Mob, damage: float) -> Optional[dict]:
        mob.health -= damage
        return self.kill_mob(mob) if mob.health <= 0 else None

    def kill_mob(self, mob: Mob) -> dict:
        drops = self._get_drops(mob.type)
        self._remove_mob(mob)
        return drops

    @staticmethod
    def _get_drops(mob_type: str) -> dict:
        result = {}
        for item, (lo, hi) in MOB_DROPS.get(mob_type, {}).items():
            n = random.randint(lo, hi)
            if n > 0: result[item] = n
        return result

    def get_mob_drops(self, mob_type: str) -> dict:
        return self._get_drops(mob_type)

    def _remove_mob(self, mob: Mob) -> None:
        if mob in self.mobs: self.mobs.remove(mob)

    def remove_mob(self, mob: Mob) -> None:
        self._remove_mob(mob)

    def clear_all_mobs(self) -> None:
        self.mobs.clear()

    def get_nearby_hostile_mobs(self, position: Vec3, radius: float = 3.0) -> list:
        return [m for m in self.mobs if m.hostile and
                Vec3(m.pos.x - position.x, 0, m.pos.z - position.z).length() < radius]

    # legacy no-op shims
    def maybe_spawn_night_mobs(self, *a, **kw): pass
    def draw_hud_indicators(self, *a, **kw): pass