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


# ── Noise functions ─────────────────────────────────────────────────────────

@njit(cache=True)
def _hash(x: int, z: int, seed: int) -> float:
    """Fast integer hash → float in [-1, 1]."""
    n = (x * 1664525 + z * 1013904223 + seed) & 0xFFFFFFFF
    n ^= (n >> 16)
    n = (n * 0x45d9f3b) & 0xFFFFFFFF
    n ^= (n >> 16)
    return (n / 0x7FFFFFFF) - 1.0


@njit(cache=True)
def _smooth(t: float) -> float:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(cache=True)
def _vn(x: float, z: float, seed: int) -> float:
    """Value noise, returns [-1, 1]."""
    ix = int(math.floor(x))
    iz = int(math.floor(z))
    fx = x - float(ix)
    fz = z - float(iz)
    ux = _smooth(fx)
    uz = _smooth(fz)
    v00 = _hash(ix, iz, seed)
    v10 = _hash(ix + 1, iz, seed)
    v01 = _hash(ix, iz + 1, seed)
    v11 = _hash(ix + 1, iz + 1, seed)
    a = v00 + ux * (v10 - v00)
    b = v01 + ux * (v11 - v01)
    return a + uz * (b - a)


@njit(cache=True)
def _sea_noise(x: float, z: float, seed: int) -> float:
    a = math.sin(x * 0.08 + seed * 1e-9) * math.cos(z * 0.08 + seed * 1e-9)
    b = math.sin(x * 0.21 + seed * 2e-9) * math.cos(z * 0.13 + seed * 3e-9)
    c = math.sin(x * 0.05 + seed * 4e-9) * math.cos(z * 0.05 + seed * 5e-9)
    return max(-1.0, min(1.0, a * 0.5 + b * 0.35 + c * 0.15))


# ── Public API ───────────────────────────────────────────────────────────────

@njit(cache=True)
def ocean_floor_height(wx: int, wz: int, seed: int) -> int:
    """Varied ocean floor with hills and trenches."""
    # Add more variation to ocean floor
    n1 = _vn(float(wx) * 0.03, float(wz) * 0.03, seed ^ 0xABCD)
    n2 = _vn(float(wx) * 0.08, float(wz) * 0.08, seed ^ 0x1234)
    n3 = _vn(float(wx) * 0.015, float(wz) * 0.015, seed ^ 0x5678)
    
    # Combine noises for varied terrain
    combined = n1 * 0.5 + n2 * 0.3 + n3 * 0.2
    
    # Range between OCEAN_FLOOR-8 and SEA_LEVEL-2
    min_floor = max(BEDROCK_LEVEL + 5, OCEAN_FLOOR - 8)
    max_floor = SEA_LEVEL - 2
    
    h = min_floor + int((combined + 1) * 0.5 * (max_floor - min_floor))
    return max(min_floor, min(max_floor, h))


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