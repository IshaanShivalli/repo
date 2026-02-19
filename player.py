"""
player.py  –  Player entity: state, physics, collision detection, raycasting.
"""

import math
import numpy as np

from block_textures import BlockType
from settings import (
    CHUNK_SIZE, CHUNK_HEIGHT, N_BLOCK_TYPES, _BT_LIST,
    BT_SOLID, EYE_OFFSET, PLAYER_HEIGHT, PLAYER_SPEED,
    GRAVITY, JUMP_VEL, MOUSE_SENS, _lookat,
)
from terrain import _terrain_height

try:
    from pyglet.window import key
except ImportError:
    key = None


WATER_BUOYANCY = 4.0
WATER_SWIM_UP = 6.0
WATER_SWIM_DOWN = 4.0
WATER_DRAG = 0.6
SURFACE_DRAG = 0.8


class Player:
    def __init__(self, x: float, y: float, z: float):
        self.x = float(x);  self.y = float(y);  self.z = float(z)
        self.vx = self.vy = self.vz = 0.0
        self.yaw   = 0.0
        self.pitch = 0.0

        self.on_ground  = False
        self.in_water   = False
        self.was_in_air = False
        self.fall_dist  = 0.0

        self.health     = 20.0;  self.max_health  = 20.0
        self.hunger     = 20.0;  self.max_hunger  = 20.0
        self.saturation = 5.0

        self.xp      = 0
        self.level   = 0

        self.inv_time      = 0.0
        self.void_time     = 0.0
        self.game_mode     = "survival"
        self.regen_time    = 0.0
        self.death_msg_time = 0.0

    def take_damage(self, amt: float) -> None:
        if self.inv_time <= 0:
            self.health -= amt
            self.inv_time = 0.5
            if self.health <= 0:
                self.health = self.max_health
                self.death_msg_time = 2.0

    def eye(self):
        return self.x, self.y + EYE_OFFSET, self.z

    def forward(self):
        yr = math.radians(self.yaw)
        pr = math.radians(self.pitch)
        return (
            math.cos(pr) * math.sin(yr),
            math.sin(pr),
            math.cos(pr) * math.cos(yr),
        )

    def view_mat(self):
        ex, ey, ez = self.eye()
        fx, fy, fz = self.forward()
        fl = math.sqrt(fx*fx + fy*fy + fz*fz)
        fwd = np.array([fx/fl, fy/fl, fz/fl], dtype=np.float32)
        return _lookat(np.array([ex, ey, ez], np.float32), fwd)

    def feet_in_water(self, world):
        b = world.get_block(int(self.x), int(self.y), int(self.z))
        return b < N_BLOCK_TYPES and _BT_LIST[b] == BlockType.WATER

    def update(self, dt: float, world, keys: set, mx: float, mz: float):
        dt = min(dt, 0.05)

        if self.inv_time > 0:
            self.inv_time -= dt

        yr = math.radians(self.yaw)
        fw = (math.sin(yr), 0.0, math.cos(yr))
        ri = (-math.cos(yr), 0.0, math.sin(yr))

        self.vx = (fw[0]*mz + ri[0]*mx) * PLAYER_SPEED
        self.vz = (fw[2]*mz + ri[2]*mx) * PLAYER_SPEED

        block_here = world.get_block(int(self.x), int(self.y), int(self.z))
        self.in_water = (
            block_here < N_BLOCK_TYPES and
            _BT_LIST[block_here] == BlockType.WATER
        )

        if self.in_water:
            if key and key.SPACE in keys:
                self.vy = WATER_SWIM_UP
            elif key and key.LSHIFT in keys:
                self.vy = -WATER_SWIM_DOWN
            else:
                # gentle sinking when idle
                self.vy -= GRAVITY * 0.15 * dt

            self.vx *= WATER_DRAG
            self.vz *= WATER_DRAG
            self.fall_dist = 0.0
        else:
            self.vy -= GRAVITY * dt
            if key and key.SPACE in keys and self.on_ground:
                self.vy = JUMP_VEL


        if self.feet_in_water(world):
            self.vx *= SURFACE_DRAG
            self.vz *= SURFACE_DRAG

        self._collide(dt, world)

        if self.y < -20:
            self.void_time += dt
            if self.void_time > 2.0:
                sy = world.surface_y(int(self.x), int(self.z))
                self.y = sy + 2
                self.vy = 0.0
                self.void_time = 0.0
        else:
            self.void_time = 0.0

    def _collide(self, dt: float, world):
        W = 0.3

        def hit(px, py, pz):
            for bx in (px-W, px+W):
                for bz in (pz-W, pz+W):
                    for by in (py, py+PLAYER_HEIGHT*0.5, py+PLAYER_HEIGHT-0.05):
                        if world.is_solid(bx, by, bz):
                            return True
            return False

        self.x += self.vx * dt
        if hit(self.x, self.y, self.z):
            self.x -= self.vx * dt
            self.vx = 0.0

        self.z += self.vz * dt
        if hit(self.x, self.y, self.z):
            self.z -= self.vz * dt
            self.vz = 0.0

        prev_vy = self.vy
        self.y += self.vy * dt
        if hit(self.x, self.y, self.z):
            self.y -= prev_vy * dt
            if prev_vy < 0:
                self.on_ground = True
            self.vy = 0.0
        else:
            self.on_ground = False

        if not self.on_ground and not self.in_water and self.vy < 0:
            self.fall_dist += -self.vy * dt
        elif self.on_ground:
            if self.fall_dist > 3.0:
                self.take_damage(self.fall_dist - 3.0)
            self.fall_dist = 0.0

    def raycast(self, world, max_d: float = 6.0):
        ex, ey, ez = self.eye()
        dx, dy, dz = self.forward()
        l = math.sqrt(dx*dx + dy*dy + dz*dz)
        dx /= l; dy /= l; dz /= l

        step = 0.1
        dist = 0.0

        x, y, z = ex, ey, ez
        prev = (int(x), int(y), int(z))
        norm = (0, 0, 0)

        while dist <= max_d:
            bx, by, bz = int(x), int(y), int(z)
            bt = world.get_block(bx, by, bz)

            if bt and bt < N_BLOCK_TYPES and BT_SOLID[bt]:
                return (bx, by, bz), norm, prev

            prev = (bx, by, bz)

            x += dx * step
            y += dy * step
            z += dz * step
            dist += step

        return None