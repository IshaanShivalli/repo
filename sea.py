"""
sea.py  –  Ocean biome water generation.

Handles everything for the ocean biome (biome 4):
  - Varied ocean floor depth
  - Bedrock / stone / sand column fill
  - Kelp and seagrass placement

Called from gen_chunk() in terrain.py.
"""

import math
from numba import njit
from settings import SEA_LEVEL, BEDROCK_LEVEL, OCEAN_FLOOR, CHUNK_HEIGHT


# ── Noise ────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _sea_noise(x: float, z: float, seed: int) -> float:
    a = math.sin(x * 0.08 + seed * 1e-9) * math.cos(z * 0.08 + seed * 1e-9)
    b = math.sin(x * 0.21 + seed * 2e-9) * math.cos(z * 0.13 + seed * 3e-9)
    c = math.sin(x * 0.05 + seed * 4e-9) * math.cos(z * 0.05 + seed * 5e-9)
    return max(-1.0, min(1.0, a * 0.5 + b * 0.35 + c * 0.15))


# ── Public API ───────────────────────────────────────────────────────────────

@njit(cache=True)
def ocean_floor_height(wx: int, wz: int, seed: int) -> int:
    """Varied ocean floor depth between OCEAN_FLOOR and SEA_LEVEL-4."""
    n = _sea_noise(float(wx) * 0.05, float(wz) * 0.05, seed ^ 0xABCD)
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
        elif y <= floor_y:
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
        lcg = (lcg * 1664525 + 1013904223) & 0xFFFFFFFF
        kelp_h = 2 + int(lcg & 0x7) % 5
        for ky in range(above, min(above + kelp_h, SEA_LEVEL)):
            if blk[dx, ky, dz] == ID_WATER:
                blk[dx, ky, dz] = ID_KELP
    elif rv < 70:
        if blk[dx, above, dz] == ID_WATER:
            blk[dx, above, dz] = ID_SEAGRASS

    return lcg