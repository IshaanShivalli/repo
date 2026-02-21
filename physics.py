"""
physics.py  –  Water physics constants and helper logic.

Water behaviour:
  - Player sinks naturally (no buoyancy push) until SPACE is pressed
  - SPACE propels the player upward; releasing SPACE lets them sink again
  - SHIFT dives faster
  - Breath metre depletes smoothly underwater; refills on surface
  - Drowning: 1 HP/s after air runs out
  - Fall damage cancelled on water entry
  - No random buoyancy flicker – buoyancy is ZERO unless SPACE held
"""

import math
from block_textures import BlockType

# ── Tuneable constants ────────────────────────────────────────────────────────

# Gravity scale inside water (feels heavy but not instant sink)
WATER_GRAVITY_SCALE  = 0.28      # fraction of normal gravity while submerged

# Horizontal drag (per physics step, multiplied each frame)
WATER_HORIZ_DRAG     = 0.80      # keeps movement snappy but with resistance

# Vertical drag – distinct from horizontal
WATER_VERT_DRAG      = 0.85

# Terminal velocities in water
WATER_SINK_CAP       = -3.5      # max downward speed in water (m/s)
WATER_RISE_CAP       =  4.0      # max upward speed while swimming (m/s)

# Upward impulse while SPACE is held underwater (m/s²)
WATER_SWIM_FORCE     =  18.0

# Jump velocity when exiting water at the surface
WATER_SURFACE_JUMP   =  8.0

# Breathing
AIR_MAX              = 10.0      # seconds (= 10 bubble icons)
DROWN_DAMAGE_RATE    =  1.0      # HP per second when air depleted
AIR_REFILL_TIME      =  1.5      # seconds to fully refill air bar on surface

# ── Block classification helpers ─────────────────────────────────────────────

def block_is_water(bt_id: int, n_block_types: int, bt_list: list) -> bool:
    """True only for WATER blocks (not kelp/seagrass)."""
    return (0 < bt_id < n_block_types and bt_list[bt_id] == BlockType.WATER)


def block_is_swimmable(bt_id: int, n_block_types: int, bt_list: list) -> bool:
    """Non-solid blocks the player moves through freely (water + plants)."""
    if bt_id <= 0 or bt_id >= n_block_types:
        return False
    return bt_list[bt_id] in (BlockType.WATER, BlockType.KELP, BlockType.SEAGRASS)


# ── Submersion probe ─────────────────────────────────────────────────────────

def probe_water(player_x, player_y, player_z,
                player_height, eye_offset,
                world, n_block_types, bt_list):
    """
    Returns (in_water, head_in_water, depth_fraction).

    depth_fraction:
        0.0  = completely dry
        0.3  = feet wet
        0.6  = mid-body wet
        1.0  = head submerged
    """
    foot_y = int(math.floor(player_y))
    mid_y  = int(math.floor(player_y + player_height * 0.5))
    head_y = int(math.floor(player_y + eye_offset))
    ix     = int(math.floor(player_x))
    iz     = int(math.floor(player_z))

    def is_water(y):
        return block_is_water(world.get_block(ix, y, iz), n_block_types, bt_list)

    foot_wet = is_water(foot_y)
    mid_wet  = is_water(mid_y)
    head_wet = is_water(head_y)

    if head_wet:
        depth = 1.0
    elif mid_wet:
        depth = 0.6
    elif foot_wet:
        depth = 0.3
    else:
        depth = 0.0

    return (foot_wet or mid_wet), head_wet, depth


# ── Breathing update ─────────────────────────────────────────────────────────

def update_breathing(air: float, head_in_water: bool,
                     dt: float, game_mode: str,
                     take_damage_fn):
    """
    Consume or refill air bar; deal drowning damage if needed.

    Returns updated air value.
    """
    if head_in_water:
        air = max(0.0, air - dt)
        if air <= 0.0 and game_mode == "survival":
            take_damage_fn(DROWN_DAMAGE_RATE * dt)
    else:
        # Refill at AIR_REFILL_TIME seconds to full
        air = min(AIR_MAX, air + dt * (AIR_MAX / AIR_REFILL_TIME))
    return air


# ── Water movement & drag ─────────────────────────────────────────────────────

def apply_water_physics(vx, vy, vz, dt,
                        gravity,
                        in_water, head_in_water, water_depth,
                        space_pressed, shift_pressed):
    """
    Apply in-water forces to velocity components.

    NO buoyancy is applied passively – the player sinks under gravity.
    SPACE applies an upward swim force; at the surface it launches the player.

    Returns (vx, vy, vz).
    """
    # Reduced gravity when wet
    grav = gravity * WATER_GRAVITY_SCALE
    vy -= grav * dt

    # Clamp to terminal velocities
    vy = max(WATER_SINK_CAP, min(WATER_RISE_CAP, vy))

    # Horizontal drag
    vx *= WATER_HORIZ_DRAG
    vz *= WATER_HORIZ_DRAG

    # Vertical drag (applied after swim force so player can exit water)
    if not space_pressed:
        vy *= WATER_VERT_DRAG

    # SPACE: swim up / exit
    if space_pressed:
        if not head_in_water:
            # At surface – pop out of the water
            vy = WATER_SURFACE_JUMP
        else:
            # Propel upward
            vy = min(WATER_RISE_CAP, vy + WATER_SWIM_FORCE * dt)

    # SHIFT: dive
    if shift_pressed:
        vy = max(WATER_SINK_CAP, vy - WATER_SWIM_FORCE * dt)

    return vx, vy, vz