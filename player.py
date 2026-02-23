"""
player.py  –  Player entity: state, physics, collision, raycasting.

Water physics are handled by physics.py.
Key changes vs previous version:
  - No passive buoyancy – player sinks naturally, SPACE swims up
  - Smooth breath-metre transitions (no flicker)
  - Cave-water surface glitch fixed via tighter Y-collision guard
"""

import math
import numpy as np

from block_textures import BlockType
from settings import (
    CHUNK_SIZE, CHUNK_HEIGHT, N_BLOCK_TYPES, _BT_LIST,
    BT_SOLID, BT_HIT, EYE_OFFSET, PLAYER_HEIGHT, PLAYER_SPEED,
    GRAVITY, JUMP_VEL, MOUSE_SENS, _lookat,
)
from physics import (
    AIR_MAX, probe_water, update_breathing, apply_water_physics,
    block_is_water, block_is_swimmable,
)
from terrain import _terrain_height

try:
    from pyglet.window import key
except ImportError:
    key = None


class Player:
    def __init__(self, x: float, y: float, z: float):
        self.x = float(x);  self.y = float(y);  self.z = float(z)
        self.vx = self.vy = self.vz = 0.0
        self.yaw   = 0.0
        self.pitch = 0.0
        self.on_ground     = False
        self.in_water      = False
        self.head_in_water = False
        self.water_depth   = 0.0
        self.was_in_water  = False
        # Smooth breath bar – track a display value separately from raw air
        self.air           = AIR_MAX
        self.air_display   = AIR_MAX   # lerped value for HUD (no flicker)
        self.health        = 20.0;  self.max_health  = 20.0
        self.hunger        = 20.0;  self.max_hunger  = 20.0
        self.saturation    = 5.0
        self.xp            = 0
        self.level         = 0
        self.inv_time      = 0.0
        self.void_time     = 0.0
        self.game_mode     = "survival"
        self.fall_dist     = 0.0
        self.regen_time    = 0.0
        self.death_msg_time = 0.0
        self.sprinting     = False
        self.crouching     = False
        self.sprint_timer  = 0.0   # double-tap W detection
        self.w_tap_time    = 0.0   # time of last W tap

    # ── helpers ───────────────────────────────────────────────────────────────

    def take_damage(self, amt: float) -> None:
        if self.inv_time <= 0:
            self.health   = max(0.0, self.health - amt)
            self.inv_time = 0.5
            if self.health <= 0:
                self.health         = self.max_health
                self.death_msg_time = 2.0

    def eye(self) -> tuple:
        # Crouch lowers eye height by 0.4 blocks
        offset = EYE_OFFSET - 0.4 if self.crouching else EYE_OFFSET
        return self.x, self.y + offset, self.z

    def forward(self) -> tuple:
        yr = math.radians(self.yaw)
        pr = math.radians(self.pitch)
        return (math.cos(pr) * math.sin(yr),
                math.sin(pr),
                math.cos(pr) * math.cos(yr))

    def view_mat(self) -> np.ndarray:
        ex, ey, ez = self.eye()
        fx, fy, fz = self.forward()
        fl = math.sqrt(fx*fx + fy*fy + fz*fz)
        fwd = np.array([fx/fl, fy/fl, fz/fl], dtype=np.float32)
        return _lookat(np.array([ex, ey, ez], np.float32), fwd)

    # ── update ────────────────────────────────────────────────────────────────

    def update(self, dt: float, world, keys: set, mx: float, mz: float) -> None:
        dt = min(dt, 0.05)

        # Timers
        if self.inv_time > 0:
            self.inv_time -= dt
        if self.death_msg_time > 0:
            self.death_msg_time = max(0.0, self.death_msg_time - dt)

        # Hunger / regen
        if self.game_mode == "survival":
            self.hunger = max(0.0, self.hunger - 0.0001 * dt * 60)
            if self.hunger <= 0:
                self.take_damage(0.02 * dt * 60)
            if self.health < self.max_health and self.hunger > 0:
                self.regen_time += dt
                if self.regen_time >= 1.0:
                    self.health     = min(self.max_health, self.health + 1.0)
                    self.hunger     = max(0.0, self.hunger - 0.5)
                    self.regen_time = 0.0
            else:
                self.regen_time = 0.0

        # Water state probe
        self.in_water, self.head_in_water, self.water_depth = probe_water(
            self.x, self.y, self.z,
            PLAYER_HEIGHT, EYE_OFFSET,
            world, N_BLOCK_TYPES, _BT_LIST,
        )

        # Breathing (physics.py handles damage)
        self.air = update_breathing(
            self.air, self.head_in_water, dt,
            self.game_mode, self.take_damage,
        )

        # Smooth display value – lerp toward real air quickly but not instantly
        # This eliminates the single-frame flicker when crossing the surface.
        LERP_SPEED = 8.0
        self.air_display += (self.air - self.air_display) * min(1.0, LERP_SPEED * dt)

        # ── Movement direction ────────────────────────────────────────────────
        ctrl_pressed  = key and (key.LCTRL  in keys or key.RCTRL  in keys)
        shift_key     = key and (key.LSHIFT in keys or key.RSHIFT in keys)
        w_pressed     = key and key.W in keys

        # Sprinting: hold Ctrl (or double-tap W) while moving forward
        if ctrl_pressed and w_pressed and self.on_ground and not self.in_water:
            self.sprinting = True
        elif not w_pressed or self.in_water or not self.on_ground:
            self.sprinting = False

        # Crouching: hold Shift while on ground and not sprinting
        if shift_key and self.on_ground and not self.in_water:
            self.crouching = True
            self.sprinting = False
        else:
            self.crouching = False

        yr    = math.radians(self.yaw)
        fw    = (math.sin(yr), 0.0, math.cos(yr))
        ri    = (-math.cos(yr), 0.0, math.sin(yr))

        if self.in_water:
            speed = PLAYER_SPEED * 0.55
        elif self.sprinting:
            speed = PLAYER_SPEED * 1.6
        elif self.crouching:
            speed = PLAYER_SPEED * 0.3
        else:
            speed = PLAYER_SPEED

        self.vx = (fw[0]*mz + ri[0]*mx) * speed
        self.vz = (fw[2]*mz + ri[2]*mx) * speed

        # ── Gravity / water physics ───────────────────────────────────────────
        space_pressed = key and key.SPACE in keys
        shift_pressed = key and (key.LSHIFT in keys or key.RSHIFT in keys)  # swim down in water

        if self.in_water:
            self.vx, self.vy, self.vz = apply_water_physics(
                self.vx, self.vy, self.vz, dt,
                GRAVITY,
                self.in_water, self.head_in_water, self.water_depth,
                space_pressed, shift_pressed,
            )
        else:
            self.vy -= GRAVITY * dt
            if space_pressed and self.on_ground:
                self.vy = JUMP_VEL

        # Cancel fall damage when entering water
        if self.in_water and not self.was_in_water:
            self.fall_dist = 0.0

        self.was_in_water = self.in_water

        self._collide(dt, world)

        # Void rescue
        if self.y < -20:
            self.void_time += dt
            if self.void_time > 2.0:
                sy = world.surface_y(int(self.x), int(self.z))
                self.y     = float(sy + 2)
                self.vy    = 0.0
                self.void_time = 0.0
        else:
            self.void_time = 0.0

    # ── collision ─────────────────────────────────────────────────────────────

    def _collide(self, dt: float, world) -> None:
        W = 0.3

        def hit(px: float, py: float, pz: float) -> bool:
            """True if player AABB overlaps any solid block (water is non-solid)."""
            for bx in (px - W, px + W):
                for bz in (pz - W, pz + W):
                    for by in (py,
                               py + PLAYER_HEIGHT * 0.5,
                               py + PLAYER_HEIGHT - 0.05):
                        bt = world.get_block(
                            int(math.floor(bx)),
                            int(math.floor(by)),
                            int(math.floor(bz)))
                        if bt and bt < N_BLOCK_TYPES and BT_SOLID[bt]:
                            return True
            return False

        def has_support(px: float, py: float, pz: float) -> bool:
            """True if there's solid ground directly under any foot corner."""
            y = py - 0.05
            for bx in (px - W, px + W):
                for bz in (pz - W, pz + W):
                    bt = world.get_block(
                        int(math.floor(bx)),
                        int(math.floor(y)),
                        int(math.floor(bz)))
                    if bt and bt < N_BLOCK_TYPES and BT_SOLID[bt]:
                        return True
            return False

        def touch_cactus(px: float, py: float, pz: float) -> bool:
            for bx in (px - W, px + W):
                for bz in (pz - W, pz + W):
                    for by in (py,
                               py + PLAYER_HEIGHT * 0.5,
                               py + PLAYER_HEIGHT - 0.05):
                        bt = world.get_block(
                            int(math.floor(bx)),
                            int(math.floor(by)),
                            int(math.floor(bz)))
                        if bt and bt < N_BLOCK_TYPES and _BT_LIST[bt] == BlockType.CACTUS:
                            return True
            return False

        # X axis
        self.x += self.vx * dt
        if hit(self.x, self.y, self.z):
            self.x -= self.vx * dt;  self.vx = 0.0
        elif self.crouching and self.on_ground and not self.in_water:
            if not has_support(self.x, self.y, self.z):
                self.x -= self.vx * dt;  self.vx = 0.0

        # Z axis
        self.z += self.vz * dt
        if hit(self.x, self.y, self.z):
            self.z -= self.vz * dt;  self.vz = 0.0
        elif self.crouching and self.on_ground and not self.in_water:
            if not has_support(self.x, self.y, self.z):
                self.z -= self.vz * dt;  self.vz = 0.0

        # Y axis
        prev_vy   = self.vy
        self.y   += self.vy * dt

        # Fix for cave-water ceiling glitch:
        # If the player is inside a water column that has a solid ceiling and
        # the upward velocity clips them into it, pull back and zero vy.
        if hit(self.x, self.y, self.z):
            # Always push back exactly one step
            self.y       -= prev_vy * dt
            # Clamp: if we were moving up into a ceiling, stop; if falling, land
            self.on_ground = prev_vy < 0
            self.vy        = 0.0
        else:
            self.on_ground = False

        # Cactus damage
        if touch_cactus(self.x, self.y, self.z) and self.inv_time <= 0:
            self.take_damage(1.0)

        # Fall damage (suppressed in water)
        if not self.in_water:
            if self.on_ground:
                if self.fall_dist > 3.0:
                    self.take_damage((self.fall_dist - 3.0) * 1.0)
                self.fall_dist = 0.0
            else:
                if self.vy < 0:
                    self.fall_dist += -self.vy * dt
        else:
            self.fall_dist = 0.0

    # ── raycast ───────────────────────────────────────────────────────────────

    def raycast(self, world, max_d: float = 6.0):
        ex, ey, ez = self.eye()
        dx, dy, dz = self.forward()
        il = math.sqrt(dx*dx + dy*dy + dz*dz)
        dx /= il;  dy /= il;  dz /= il

        ix = int(math.floor(ex))
        iy = int(math.floor(ey))
        iz = int(math.floor(ez))

        sx = 1 if dx > 0 else -1
        sy = 1 if dy > 0 else -1
        sz = 1 if dz > 0 else -1

        def safe(a: float) -> float:
            return 1e18 if abs(a) < 1e-10 else a

        tmx = ((ix + (1 if dx > 0 else 0)) - ex) / safe(dx)
        tmy = ((iy + (1 if dy > 0 else 0)) - ey) / safe(dy)
        tmz = ((iz + (1 if dz > 0 else 0)) - ez) / safe(dz)
        tdx = abs(1 / safe(dx))
        tdy = abs(1 / safe(dy))
        tdz = abs(1 / safe(dz))

        norm = (0, 0, 0)
        prev = (ix, iy, iz)
        t    = 0.0

        for _ in range(int(max_d * 4)):
            bt = world.get_block(ix, iy, iz)
            if bt and BT_HIT[bt]:
                return (ix, iy, iz), norm, prev
            prev = (ix, iy, iz)
            if tmx < tmy and tmx < tmz:
                t = tmx;  tmx += tdx;  ix += sx;  norm = (-sx, 0, 0)
            elif tmy < tmz:
                t = tmy;  tmy += tdy;  iy += sy;  norm = (0, -sy, 0)
            else:
                t = tmz;  tmz += tdz;  iz += sz;  norm = (0, 0, -sz)
            if t > max_d:
                break
        return None
