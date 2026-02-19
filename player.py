"""
player.py  –  Player entity: state, physics (including full water physics),
              collision detection, raycasting.

Water physics implemented:
  - Buoyancy: upward force proportional to submersion depth
  - Drag: horizontal and vertical velocity damped in water
  - Swimming: SPACE near surface jumps out; SPACE submerged rises steadily
  - Breathing: air bar (10 bubbles = 10 s), depletes underwater, refills on surface
  - Drowning: 1 HP/s damage when air runs out
  - Water entry/exit: suppresses fall damage
  - Kelp / seagrass: non-solid, player moves through freely
"""

import math
import numpy as np

from block_textures import BlockType
from settings import (
    CHUNK_SIZE, CHUNK_HEIGHT, N_BLOCK_TYPES, _BT_LIST,
    BT_SOLID, EYE_OFFSET, PLAYER_HEIGHT, PLAYER_SPEED,
    GRAVITY, JUMP_VEL, MOUSE_SENS, _lookat,AIR_MAX, DROWN_DAMAGE_RATE,WATER_GRAVITY_SCALE,
    BUOYANCY_STRENGTH,WATER_SINK_CAP,WATER_RISE_CAP, WATER_HORIZ_DRAG,WATER_VERT_DRAG, WATER_SURFACE_JUMP,WATER_SWIM_FORCE,
)
from terrain import _terrain_height

try:
    from pyglet.window import key
except ImportError:
    key = None   # allow import without display (unit tests etc.)

# ── water tuning constants ────────────────────────────────────────────────────


def _block_is_water(bt_id: int) -> bool:
    return (0 < bt_id < N_BLOCK_TYPES and
            _BT_LIST[bt_id] == BlockType.WATER)


def _block_is_swimmable(bt_id: int) -> bool:
    """Non-solid liquid/plant blocks the player swims through."""
    if bt_id <= 0 or bt_id >= N_BLOCK_TYPES:
        return False
    name = _BT_LIST[bt_id]
    return name in (BlockType.WATER, BlockType.KELP, BlockType.SEAGRASS)


class Player:
    def __init__(self, x: float, y: float, z: float):
        self.x = float(x);  self.y = float(y);  self.z = float(z)
        self.vx = self.vy = self.vz = 0.0
        self.yaw   = 0.0
        self.pitch = 0.0
        self.on_ground  = False
        self.in_water   = False   # foot-level water
        self.head_in_water = False  # eye-level water (breathing)
        self.water_depth  = 0.0    # 0.0 = not in water, 1.0 = fully submerged
        self.was_in_water = False   # for fall-damage suppression
        self.air = AIR_MAX          # seconds of air remaining
        self.health     = 20.0;  self.max_health  = 20.0
        self.hunger     = 20.0;  self.max_hunger  = 20.0
        self.saturation = 5.0
        self.xp      = 0
        self.level   = 0
        self.inv_time      = 0.0
        self.void_time     = 0.0
        self.game_mode     = "survival"
        self.fall_dist     = 0.0
        self.regen_time    = 0.0
        self.death_msg_time = 0.0

    # ── helpers ───────────────────────────────────────────────────────────────

    def take_damage(self, amt: float) -> None:
        if self.inv_time <= 0:
            self.health   = max(0.0, self.health - amt)
            self.inv_time = 0.5
            if self.health <= 0:
                self.health         = self.max_health
                self.death_msg_time = 2.0

    def eye(self) -> tuple:
        return self.x, self.y + EYE_OFFSET, self.z

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

    # ── water probing ─────────────────────────────────────────────────────────

    def _probe_water(self, world) -> tuple:
        """
        Returns (in_water, head_in_water, depth_fraction).
        depth_fraction: 0=dry, 0.5=feet submerged, 1.0=fully under.
        """
        foot_y  = int(math.floor(self.y))
        mid_y   = int(math.floor(self.y + PLAYER_HEIGHT * 0.5))
        head_y  = int(math.floor(self.y + EYE_OFFSET))
        ix      = int(math.floor(self.x))
        iz      = int(math.floor(self.z))

        foot_bt = world.get_block(ix, foot_y, iz)
        mid_bt  = world.get_block(ix, mid_y,  iz)
        head_bt = world.get_block(ix, head_y, iz)

        foot_wet = _block_is_water(foot_bt)
        mid_wet  = _block_is_water(mid_bt)
        head_wet = _block_is_water(head_bt)

        if head_wet:
            depth = 1.0
        elif mid_wet:
            depth = 0.6
        elif foot_wet:
            depth = 0.3
        else:
            depth = 0.0

        return foot_wet or mid_wet, head_wet, depth

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
                    self.health   = min(self.max_health, self.health + 1.0)
                    self.hunger   = max(0.0, self.hunger - 0.5)
                    self.regen_time = 0.0
            else:
                self.regen_time = 0.0

        # Water state
        self.in_water, self.head_in_water, self.water_depth = (
            self._probe_water(world))

        # Breathing / drowning
        if self.head_in_water:
            self.air = max(0.0, self.air - dt)
            if self.air <= 0 and self.game_mode == "survival":
                self.take_damage(DROWN_DAMAGE_RATE * dt)
        else:
            # Refill air faster than depletion (1.5 s to full)
            self.air = min(AIR_MAX, self.air + dt * (AIR_MAX / 1.5))

        # ── Movement direction ────────────────────────────────────────────────
        yr  = math.radians(self.yaw)
        fw  = (math.sin(yr), 0.0, math.cos(yr))
        ri  = (-math.cos(yr), 0.0, math.sin(yr))
        speed = PLAYER_SPEED * (0.6 if self.in_water else 1.0)
        self.vx = (fw[0]*mz + ri[0]*mx) * speed
        self.vz = (fw[2]*mz + ri[2]*mx) * speed

        # ── Gravity & buoyancy ────────────────────────────────────────────────
        if self.in_water:
            # Reduced gravity
            grav = GRAVITY * WATER_GRAVITY_SCALE
            self.vy -= grav * dt

            # Buoyancy: push upward proportional to submersion
            buoy = BUOYANCY_STRENGTH * self.water_depth * dt
            self.vy += buoy

            # Terminal velocity clamps
            self.vy = max(WATER_SINK_CAP, min(WATER_RISE_CAP, self.vy))

            # Velocity drag (water resistance)
            self.vx *= WATER_HORIZ_DRAG
            self.vz *= WATER_HORIZ_DRAG
            self.vy *= WATER_VERT_DRAG

            # Swimming: SPACE rises, SHIFT sinks
            space_pressed = key and key.SPACE in keys
            shift_pressed = key and (key.LSHIFT in keys or key.RSHIFT in keys)

            if space_pressed:
                if not self.head_in_water:
                    # Surface jump – exit the water
                    self.vy = WATER_SURFACE_JUMP
                else:
                    # Propel upward smoothly
                    self.vy = min(WATER_RISE_CAP,
                                  self.vy + WATER_SWIM_FORCE * dt)
            elif shift_pressed:
                # Sink / dive
                self.vy = max(WATER_SINK_CAP,
                              self.vy - WATER_SWIM_FORCE * dt)
        else:
            self.vy -= GRAVITY * dt
            # Normal jump
            if key and key.SPACE in keys and self.on_ground:
                self.vy = JUMP_VEL

        # Suppress fall damage when entering water
        if self.in_water and not self.was_in_water:
            self.fall_dist = 0.0   # cancel accumulated fall distance

        self.was_in_water = self.in_water

        self._collide(dt, world)

        # Void rescue
        if self.y < -20:
            self.void_time += dt
            if self.void_time > 2.0:
                sy = world.surface_y(int(self.x), int(self.z))
                self.y  = float(sy + 2)
                self.vy = 0.0
                self.void_time = 0.0
        else:
            self.void_time = 0.0

    # ── collision ─────────────────────────────────────────────────────────────

    def _collide(self, dt: float, world) -> None:
        W = 0.3

        def hit(px: float, py: float, pz: float) -> bool:
            """True if the player AABB overlaps any solid block (not water/kelp)."""
            for bx in (px - W, px + W):
                for bz in (pz - W, pz + W):
                    for by in (py, py + PLAYER_HEIGHT * 0.5, py + PLAYER_HEIGHT - 0.05):
                        bt = world.get_block(int(math.floor(bx)),
                                             int(math.floor(by)),
                                             int(math.floor(bz)))
                        if bt and bt < N_BLOCK_TYPES and BT_SOLID[bt]:
                            return True
            return False

        def touch_cactus(px: float, py: float, pz: float) -> bool:
            for bx in (px - W, px + W):
                for bz in (pz - W, pz + W):
                    for by in (py, py + PLAYER_HEIGHT * 0.5, py + PLAYER_HEIGHT - 0.05):
                        bt = world.get_block(int(math.floor(bx)),
                                             int(math.floor(by)),
                                             int(math.floor(bz)))
                        if bt and bt < N_BLOCK_TYPES and _BT_LIST[bt] == BlockType.CACTUS:
                            return True
            return False

        # X
        self.x += self.vx * dt
        if hit(self.x, self.y, self.z):
            self.x -= self.vx * dt;  self.vx = 0.0

        # Z
        self.z += self.vz * dt
        if hit(self.x, self.y, self.z):
            self.z -= self.vz * dt;  self.vz = 0.0

        # Y
        prev_vy = self.vy
        self.y += self.vy * dt
        if hit(self.x, self.y, self.z):
            self.on_ground = prev_vy < 0
            self.y        -= prev_vy * dt
            self.vy        = 0.0
        else:
            self.on_ground = False

        # Cactus damage
        if touch_cactus(self.x, self.y, self.z) and self.inv_time <= 0:
            self.take_damage(1.0)

        # Fall damage (only outside water)
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
            if bt and BT_SOLID[bt]:
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