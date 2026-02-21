"""
water_gen.py  –  All water-related terrain generation.

Two separate systems:
  1. OCEAN  – biome 4 deep water with sand floor, kelp, seagrass
  2. LAKES  – inland ponds/lakes filling natural terrain depressions

Both are called from gen_chunk() in terrain.py.
"""

import math
from numba import njit
from settings import (
    CHUNK_SIZE, CHUNK_HEIGHT, SEA_LEVEL, BEDROCK_LEVEL, OCEAN_FLOOR
)


# ═══════════════════════════════════════════════════════════════════════
#  SHARED NOISE
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _noise2(x: float, z: float, seed: int) -> float:
    a = math.sin(x * 0.08 + seed * 1e-9) * math.cos(z * 0.08 + seed * 1e-9)
    b = math.sin(x * 0.21 + seed * 2e-9) * math.cos(z * 0.13 + seed * 3e-9)
    c = math.sin(x * 0.05 + seed * 4e-9) * math.cos(z * 0.05 + seed * 5e-9)
    return max(-1.0, min(1.0, a * 0.5 + b * 0.35 + c * 0.15))


# ═══════════════════════════════════════════════════════════════════════
#  OCEAN GENERATION
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def ocean_floor_height(wx: int, wz: int, seed: int) -> int:
    """Varied ocean floor depth between OCEAN_FLOOR and SEA_LEVEL-4."""
    n = _noise2(float(wx) * 0.05, float(wz) * 0.05, seed ^ 0xABCD)
    h = OCEAN_FLOOR + int((n * 0.5 + 0.5) * (SEA_LEVEL - 6 - OCEAN_FLOOR))
    return max(OCEAN_FLOOR, min(SEA_LEVEL - 4, h))


@njit(cache=True)
def fill_ocean_column(blk, dx: int, dz: int, wx: int, wz: int,
                      seed: int, lcg: int,
                      ID_BEDROCK: int, ID_STONE: int, ID_SAND_OCEAN: int,
                      ID_WATER: int) -> int:
    """
    Fill a single ocean biome column with bedrock/stone/sand/water.
    Returns updated lcg.
    """
    H = CHUNK_HEIGHT
    floor_y = ocean_floor_height(wx, wz, seed)

    for y in range(H):
        if y <= BEDROCK_LEVEL:
            blk[dx, y, dz] = ID_BEDROCK
        elif y <= BEDROCK_LEVEL + 2:
            lcg = (lcg * 1664525 + 1013904223) & 0xFFFFFFFF
            blk[dx, y, dz] = ID_BEDROCK if (lcg & 0xFF) < 180 else ID_STONE
        elif y < floor_y - 2:
            blk[dx, y, dz] = ID_STONE
        elif y < floor_y:
            blk[dx, y, dz] = ID_SAND_OCEAN
        elif y == floor_y:
            blk[dx, y, dz] = ID_SAND_OCEAN
        elif y <= SEA_LEVEL:
            blk[dx, y, dz] = ID_WATER
        # above SEA_LEVEL → air (default 0)

    return lcg


@njit(cache=True)
def place_ocean_plants(blk, dx: int, dz: int, wx: int, wz: int,
                       seed: int, lcg: int,
                       ID_WATER: int, ID_KELP: int, ID_SEAGRASS: int) -> int:
    """
    Place kelp and seagrass on ocean floor. Returns updated lcg.
    """
    H = CHUNK_HEIGHT
    floor_y = ocean_floor_height(wx, wz, seed)
    above   = floor_y + 1

    if above >= SEA_LEVEL or above >= H:
        return lcg

    lcg = (lcg * 1664525 + 1013904223) & 0xFFFFFFFF
    rv  = lcg & 0xFF

    if rv < 40:
        # Kelp column — grows 2-6 blocks upward through water
        lcg = (lcg * 1664525 + 1013904223) & 0xFFFFFFFF
        kelp_h = 2 + int(lcg & 0x7) % 5
        for ky in range(above, min(above + kelp_h, SEA_LEVEL)):
            if blk[dx, ky, dz] == ID_WATER:
                blk[dx, ky, dz] = ID_KELP
    elif rv < 70:
        # Seagrass on the floor
        if blk[dx, above, dz] == ID_WATER:
            blk[dx, above, dz] = ID_SEAGRASS

    return lcg


# ═══════════════════════════════════════════════════════════════════════
#  INLAND LAKE / POND GENERATION
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def lake_water_top(wx: int, wz: int, height: int, seed: int) -> int:
    """
    Find the LOWEST point within radius of lake center.
    All water blocks share that exact Y as their surface.
    Only columns strictly below that Y get water placed.
    """
    CELL   = 80
    RADIUS = 4

    cell_x = wx // CELL
    cell_z = wz // CELL

    cx_f = float(cell_x * 7919 + seed)
    cz_f = float(cell_z * 6271 + seed)
    lake_roll = _noise2(cx_f * 0.001, cz_f * 0.001, seed ^ 0xCAFE)
    if lake_roll < 0.7:
        return -1

    off_x = _noise2(cx_f * 0.003, cz_f * 0.001, seed ^ 0x1111)
    off_z = _noise2(cx_f * 0.001, cz_f * 0.003, seed ^ 0x2222)
    center_wx = cell_x * CELL + 8 + int((off_x * 0.5 + 0.5) * (CELL - 16))
    center_wz = cell_z * CELL + 8 + int((off_z * 0.5 + 0.5) * (CELL - 16))

    dist_sq = (wx - center_wx) * (wx - center_wx) + (wz - center_wz) * (wz - center_wz)
    if dist_sq > RADIUS * RADIUS:
        return -1

    # Find the MINIMUM terrain height across all columns in the lake circle
    # This becomes the shared flat water surface for the whole lake
    min_h = 99999
    for ddx in range(-RADIUS, RADIUS + 1):
        for ddz in range(-RADIUS, RADIUS + 1):
            if ddx * ddx + ddz * ddz <= RADIUS * RADIUS:
                h = _terrain_height_for_lake(center_wx + ddx, center_wz + ddz, seed)
                if h < min_h:
                    min_h = h

    # Water surface is at the minimum height in the lake
    water_surface = min_h

    # This column only gets water if its floor is BELOW the water surface
    # (columns at or above water surface stay as dry land)
    if height >= water_surface:
        return -1

    return water_surface


@njit(cache=True)
def _terrain_height_for_lake(wx: int, wz: int, seed: int) -> int:
    """
    Minimal terrain height computation (duplicated here so water_gen.py
    is self-contained and can be imported independently).
    """
    n     = _noise2(float(wx) * 0.012, float(wz) * 0.012, seed)
    hills = _noise2(float(wx) * 0.04,  float(wz) * 0.04,  seed ^ 0x1234)
    h     = SEA_LEVEL + 5 + int(n * 8 + hills * 4)
    return max(BEDROCK_LEVEL + 3, min(CHUNK_HEIGHT - 4, h))