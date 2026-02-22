"""
inland_water.py  –  Small inland rivers with a FIXED water surface elevation.

Key insight: river water must be at a FIXED world-Y (SEA_LEVEL - 2), not
height-relative. Per-column water placement at different Y values creates
stepped/floating water. A fixed Y means all river water is perfectly flat,
and is carved INTO any terrain above that level.

Columns below the river Y (deep valleys, ocean) are simply skipped.
"""

import math
from numba import njit
from settings import SEA_LEVEL, BEDROCK_LEVEL, CHUNK_HEIGHT

# River water surface — just below sea level so oceans don't bleed in
RIVER_Y = int(SEA_LEVEL) - 1

# ── Gradient noise (Perlin-style) ─────────────────────────────────────────────

@njit(cache=True)
def _ihash(x: int, z: int, seed: int) -> int:
    h = (x * 1664525 + z * 22695477 + seed) & 0xFFFFFFFF
    h ^= (h >> 16)
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h ^= (h >> 16)
    return h


@njit(cache=True)
def _grad(hx: int, hz: int, seed: int, px: float, pz: float) -> float:
    h = _ihash(hx, hz, seed) & 7
    if h == 0: return  (px - float(hx)) + (pz - float(hz))
    if h == 1: return  (px - float(hx)) - (pz - float(hz))
    if h == 2: return -(px - float(hx)) + (pz - float(hz))
    if h == 3: return -(px - float(hx)) - (pz - float(hz))
    if h == 4: return  (pz - float(hz))
    if h == 5: return -(pz - float(hz))
    if h == 6: return  (px - float(hx))
    return               -(px - float(hx))


@njit(cache=True)
def _grad_noise(x: float, z: float, seed: int) -> float:
    ix = int(math.floor(x)); iz = int(math.floor(z))
    fx = x - float(ix);     fz = z - float(iz)
    ux = fx * fx * (3.0 - 2.0 * fx)
    uz = fz * fz * (3.0 - 2.0 * fz)
    v00 = _grad(ix,   iz,   seed, x, z)
    v10 = _grad(ix+1, iz,   seed, x, z)
    v01 = _grad(ix,   iz+1, seed, x, z)
    v11 = _grad(ix+1, iz+1, seed, x, z)
    return (v00*(1.0-ux)*(1.0-uz) + v10*ux*(1.0-uz) +
            v01*(1.0-ux)*uz       + v11*ux*uz)


# ── Public API ────────────────────────────────────────────────────────────────

@njit(cache=True)
def river_info(wx: int, wz: int, height: int, seed: int) -> int:
    """
    Returns RIVER_Y if this column should have river water, else -1.

    River surface is always at RIVER_Y (fixed world coordinate).
    Only columns where terrain height > RIVER_Y get carved — columns
    already below RIVER_Y are skipped (no floating water possible).
    """
    FREQ = 0.010  # noise frequency — lower = fewer rivers, wider spacing
    BAND = 0.030  # zero-crossing band width — narrower rivers

    v = _grad_noise(float(wx) * FREQ, float(wz) * FREQ * 0.8, seed ^ 0x52AB)

    if v < -BAND or v > BAND:
        return -1

    # Only carve where terrain is above river level
    if height <= RIVER_Y:
        return -1

    return int(RIVER_Y)


@njit(cache=True)
def carve_river(blk, dx: int, dz: int, height: int,
                river_y: int, ID_WATER: int, ID_DIRT: int,
                H: int) -> None:
    """
    Carve a river channel:
      - Remove all terrain blocks from river_y+1 up to height (make channel)
      - Place water at river_y (flat surface, same Y every column)
      - Dirt river bed one block below
    """
    # Air channel above water
    ry = int(river_y)
    for y in range(ry + 1, min(height + 1, H)):
        blk[dx, y, dz] = 0

    # Water — always at the same world Y
    if 0 <= ry < H:
        blk[dx, ry, dz] = ID_WATER

    # Soft river bed
    bed_y = ry - 1
    if 0 <= bed_y < H:
        blk[dx, bed_y, dz] = ID_DIRT
