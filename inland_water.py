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
    FREQ = 0.009   # make rivers rarer and wider
    BAND = 0.045   

    v = _grad_noise(float(wx) * FREQ, float(wz) * FREQ * 0.8, seed ^ 0x52AB)

    if v < -BAND or v > BAND:
        return -1

    # Small vertical variation so rivers aren't perfectly flat
    v2 = _grad_noise(float(wx) * FREQ * 2.2, float(wz) * FREQ * 1.7, seed ^ 0x7F4A)
    ry = RIVER_Y + ( -1 if v2 > 0.35 else 0 )

    # Only carve where terrain is above river level
    if height <= ry:
        return -1

    return int(ry)


@njit(cache=True)
def river_depth(wx: int, wz: int, seed: int) -> int:
    # Variable depth between 2 and 6 blocks
    d1 = _grad_noise(float(wx) * 0.05, float(wz) * 0.05, seed ^ 0xBEEF)
    d2 = _grad_noise(float(wx) * 0.15, float(wz) * 0.15, seed ^ 0xDEAD)
    depth_var = (d1 * 0.7 + d2 * 0.3 + 1) * 0.5  # Range 0-1
    return 2 + int(depth_var * 5)  # 2-7 blocks deep

@njit(cache=True)
def carve_river(blk, dx: int, dz: int, height: int,
                river_y: int, depth: int, ID_WATER: int, ID_DIRT: int,
                H: int) -> None:
    """Carve a river channel with sloped banks."""
    ry = int(river_y)
    
    # Carve main channel
    for y in range(ry + 1, min(height + 1, H)):
        blk[dx, y, dz] = 0
    
    # Water at river level
    if 0 <= ry < H:
        blk[dx, ry, dz] = ID_WATER
    
    # Create sloped banks
    bank_depth = max(1, depth // 2)
    for y in range(ry - bank_depth, ry):
        if 0 <= y < H:
            blk[dx, y, dz] = ID_DIRT
    
    # Add sand/gravel on river bottom
    if ry - depth > 0:
        blk[dx, ry - depth, dz] = ID_DIRT