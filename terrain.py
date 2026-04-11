"""
terrain.py  –  Numba-JIT terrain generation (gen_chunk) and chunk mesh builder.

Biomes:
  0 = plains        (grass, scattered oak trees)
  1 = forest        (grass, dense oak trees, tall oaks)
  2 = desert        (sand, red sand patches, cactus)
  3 = snowy taiga   (snow, spruce trees, stone peaks)
  4 = ocean         (deep water, sand floor, kelp, seagrass)
  5 = birch forest  (grass ground, dense birch trees)
"""

import math
import numpy as np
from numba import njit

from settings import CHUNK_SIZE, CHUNK_HEIGHT, SEA_LEVEL, BEDROCK_LEVEL, OCEAN_FLOOR
from sea import fill_ocean_column, place_ocean_plants, ocean_floor_height
from inland_water import river_info, river_depth, carve_river


# ═══════════════════════════════════════════════════════════════════════
#  NOISE HELPERS
# ═══════════════════════════════════════════════════════════════════════

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
    ix = int(math.floor(x)); iz = int(math.floor(z))
    fx = x - float(ix);     fz = z - float(iz)
    ux = _smooth(fx);       uz = _smooth(fz)
    v00 = _hash(ix,   iz,   seed)
    v10 = _hash(ix+1, iz,   seed)
    v01 = _hash(ix,   iz+1, seed)
    v11 = _hash(ix+1, iz+1, seed)
    a = v00 + ux * (v10 - v00)
    b = v01 + ux * (v11 - v01)
    return a + uz * (b - a)


@njit(cache=True)
def _fbm(x: float, z: float, seed: int, octaves: int,
         lacunarity: float, gain: float) -> float:
    """Fractal Brownian Motion — layered value noise."""
    v = 0.0; amp = 0.5; freq = 1.0
    for _ in range(octaves):
        v   += _vn(x * freq, z * freq, seed) * amp
        freq *= lacunarity
        amp  *= gain
        seed  = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
    return v


@njit(cache=True)
def _noise2(x: float, z: float, seed: int) -> float:
    """Legacy 2-D noise wrapper (kept for biome blending)."""
    return _vn(x, z, seed)


@njit(cache=True)
def _noise3(x: float, y: float, z: float, seed: int) -> float:
    """3-D value noise for cave carving."""
    # XZ slice
    n1 = _vn(x * 0.08 + seed * 1e-9, z * 0.08 + seed * 2e-9, seed)
    # YZ slice (different seed offset)
    n2 = _vn(y * 0.12 + seed * 3e-9, z * 0.12 + seed * 4e-9, seed ^ 0xAB12)
    # XY slice
    n3 = _vn(x * 0.10 + seed * 5e-9, y * 0.10 + seed * 6e-9, seed ^ 0xCD34)
    return (n1 + n2 + n3) / 3.0


@njit(cache=True)
def _cave_noise_v(x: float, y: float, z: float, seed: int) -> float:
    """Secondary vertical-tunnel noise."""
    a = _vn(x * 0.06, z * 0.06, seed ^ 0x9911)
    b = _vn(x * 0.03, z * 0.03, seed ^ 0x2233)
    return a * 0.6 + b * 0.4


@njit(cache=True)
def _continent(wx: int, wz: int, seed: int) -> float:
    """Large-scale continent value: negative = ocean, positive = land."""
    wx_f = float(wx); wz_f = float(wz)
    warp_x = _fbm(wx_f * 0.003 + 1.7, wz_f * 0.003,       seed ^ 0x1111, 3, 2.0, 0.5) * 40.0
    warp_z = _fbm(wx_f * 0.003,       wz_f * 0.003 + 9.2, seed ^ 0x2222, 3, 2.0, 0.5) * 40.0
    return _fbm((wx_f + warp_x) * 0.005, (wz_f + warp_z) * 0.005,
                seed ^ 0x3333, 4, 2.0, 0.5)

# Change this line in _biome_at:


@njit(cache=True)
def _terrain_height(wx: int, wz: int, seed: int) -> int:
    wx_f = float(wx); wz_f = float(wz)

    # Domain warp for dramatic terrain variation
    warp_x = _fbm(wx_f * 0.008 + 3.1, wz_f * 0.008,       seed ^ 0xAAAA, 2, 2.0, 0.5) * 25.0
    warp_z = _fbm(wx_f * 0.008,       wz_f * 0.008 + 7.4,  seed ^ 0xBBBB, 2, 2.0, 0.5) * 25.0
    wx_w = wx_f + warp_x; wz_w = wz_f + warp_z

    # Base large-scale shape
    base   = _fbm(wx_w * 0.012, wz_w * 0.012, seed ^ 0xCC00, 5, 2.1, 0.48)
    # Mid-frequency hills
    hills  = _fbm(wx_w * 0.035, wz_w * 0.035, seed ^ 0xDD00, 4, 2.0, 0.50) * 0.4
    # Fine detail
    detail = _fbm(wx_w * 0.09,  wz_w * 0.09,  seed ^ 0xEE00, 3, 2.0, 0.45) * 0.15
    # Ridge noise for mountain ridges
    ridge_raw = abs(_fbm(wx_w * 0.018, wz_w * 0.018, seed ^ 0xFF00, 4, 2.0, 0.55))
    ridge = (1.0 - ridge_raw) * (1.0 - ridge_raw) * 0.5

    combined = base + hills + detail + ridge
    h = SEA_LEVEL + 6 + int(combined * 20.0)
    return max(BEDROCK_LEVEL + 3, min(CHUNK_HEIGHT - 4, h))


@njit(cache=True)
def _biome_at(wx: int, wz: int, seed: int) -> int:
    """
    Returns biome index:
      0 = plains   1 = forest   2 = desert   3 = snowy taiga
      4 = ocean    5 = birch forest
    """
    if _continent(wx, wz, seed) < -0.08:   # ← changed from -0.20 to -0.08
        return 4   # ocean

    # Temperature noise (smooth, large scale)
    temp  = _fbm(float(wx) * 0.018, float(wz) * 0.018, seed ^ 0x5A5A, 3, 2.0, 0.5)
    # Humidity noise (different frequency/seed)
    humid = _fbm(float(wx) * 0.022, float(wz) * 0.022, seed ^ 0xA5A5, 3, 2.0, 0.5)

    h = _terrain_height(wx, wz, seed)

    # Very high peaks → snowy taiga regardless
    if h >= SEA_LEVEL + 22:
        return 3

    if temp < -0.25:
        return 3   # cold → snowy taiga

    if temp > 0.30:
        return 2   # hot → desert

    # Temperate zone — split by humidity
    if humid > 0.25:
        return 5   # wet → birch forest
    if humid > -0.10:
        return 1   # moderate → oak forest
    return 0       # dry → plains


@njit(cache=True)
def _is_coast(wx: int, wz: int, seed: int) -> bool:
    """True when a land cell is close to the ocean boundary."""
    c = _continent(wx, wz, seed)
    if c < -0.20:
        return False
    for ddx in (-5, -2, 0, 2, 5):
        for ddz in (-5, -2, 0, 2, 5):
            if _continent(wx + ddx, wz + ddz, seed) < -0.20:
                return True
    return False


@njit(cache=True)
def ocean_floor_height(wx: int, wz: int, seed: int) -> int:
    n = _vn(float(wx) * 0.05, float(wz) * 0.05, seed ^ 0xABCD)
    h = OCEAN_FLOOR + int((n * 0.5 + 0.5) * (SEA_LEVEL - 6 - OCEAN_FLOOR))
    return max(OCEAN_FLOOR, min(SEA_LEVEL - 6, h))


# ═══════════════════════════════════════════════════════════════════════
#  CHUNK GENERATION
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def gen_chunk(cx: int, cz: int, seed: int,
              ID_GRASS: int, ID_SAND: int, ID_SNOW: int,
              ID_DIRT: int, ID_STONE: int, ID_BEDROCK: int,
              ID_WATER: int, ID_LOG: int, ID_LEAVES: int,
              ID_SPRUCE_LOG: int, ID_SPRUCE_LEAVES: int, ID_CACTUS: int,
              ID_COAL: int, ID_IRON: int, ID_GOLD: int, ID_DIAMOND: int,
              ID_SAND_OCEAN: int, ID_KELP: int, ID_SEAGRASS: int,
              ID_BIRCH_LOG: int, ID_BIRCH_LEAVES: int) -> np.ndarray:
    S   = CHUNK_SIZE
    H   = CHUNK_HEIGHT
    blk = np.zeros((S, H, S), dtype=np.uint8)
    lcg = (seed ^ (cx * 73856093) ^ (cz * 19349663)) & 0xFFFFFFFF

    for dx in range(S):
        for dz in range(S):
            wx    = cx * S + dx
            wz    = cz * S + dz
            biome = _biome_at(wx, wz, seed)

            # ── Ocean ────────────────────────────────────────────────────
            if biome == 4:
                lcg = fill_ocean_column(blk, dx, dz, wx, wz, seed, lcg,
                    ID_BEDROCK, ID_STONE, ID_SAND_OCEAN, ID_WATER)
                floor_y = ocean_floor_height(wx, wz, seed)
                for _ in range(3):
                    lcg = (lcg * 1664525 + 1013904223) & 0xFFFFFFFF
                    oy  = BEDROCK_LEVEL + 3 + int(lcg & 0x1F)
                    oy  = min(oy, floor_y - 4)
                    if oy >= BEDROCK_LEVEL + 3 and blk[dx, oy, dz] == ID_STONE:
                        lcg = (lcg * 1664525 + 1013904223) & 0xFFFFFFFF
                        rv  = lcg & 0xFF
                        if   rv < 20: blk[dx, oy, dz] = ID_COAL
                        elif rv < 32: blk[dx, oy, dz] = ID_IRON
                        elif rv < 35 and oy < SEA_LEVEL - 8:  blk[dx, oy, dz] = ID_GOLD
                        elif rv < 37 and oy < SEA_LEVEL - 16: blk[dx, oy, dz] = ID_DIAMOND
                continue

            # ── Land biomes ──────────────────────────────────────────────
            height = _terrain_height(wx, wz, seed)
            coast  = _is_coast(wx, wz, seed)

            _river_y = -1
            if biome not in (2, 3) and not coast:
                _river_y = river_info(wx, wz, height, seed)

            if coast:
                surf = ID_SAND
            elif biome == 2:
                surf = ID_SAND
            elif biome == 3:
                surf = ID_STONE if height >= SEA_LEVEL + 22 else ID_SNOW
            else:
                surf = ID_GRASS   # plains, forest, birch forest all use grass

            # ── Column fill ───────────────────────────────────────────────
            for y in range(H):
                if y <= BEDROCK_LEVEL:
                    blk[dx, y, dz] = ID_BEDROCK
                elif y <= BEDROCK_LEVEL + 2:
                    lcg = (lcg * 1664525 + 1013904223) & 0xFFFFFFFF
                    blk[dx, y, dz] = ID_BEDROCK if (lcg & 0xFF) < 180 else ID_STONE
                elif y < height - 3:
                    blk[dx, y, dz] = ID_STONE
                elif y < height:
                    if coast:
                        blk[dx, y, dz] = ID_SAND
                    elif biome == 2:
                        blk[dx, y, dz] = ID_SAND
                    elif biome == 3 and height >= SEA_LEVEL + 22:
                        blk[dx, y, dz] = ID_STONE
                    else:
                        blk[dx, y, dz] = ID_DIRT
                elif y == height:
                    blk[dx, y, dz] = surf

            # ── Cave carving ──────────────────────────────────────────────
            # Spaghetti caves: two noise fields — carve where both are near 0
            # (their product creates thin worm-like tubes)
            for y in range(BEDROCK_LEVEL + 3, height - 1):
                fy = float(y); fx = float(wx); fz = float(wz)

                # Spaghetti worm caves (main cave network)
                n1 = _noise3(fx * 1.0, fy * 1.0, fz * 1.0, seed ^ 0x1122)
                n2 = _noise3(fx * 1.0, fy * 1.0, fz * 1.0, seed ^ 0x3344)
                if n1 * n1 + n2 * n2 < 0.04:   # thin tubes where both near 0
                    blk[dx, y, dz] = 0
                    continue

                # Big chambers: single-field threshold (20% more open than before)
                n3 = _noise3(fx * 0.6, fy * 0.7, fz * 0.6, seed ^ 0x5566)
                big = _vn(fx * 0.05, fz * 0.05, seed ^ 0x7788) > 0.3
                thresh = 0.30 if big else 0.40
                if n3 > thresh:
                    blk[dx, y, dz] = 0
                    continue

                # Vertical shafts (ravine-style, ~8% of columns)
                shaft_v = _cave_noise_v(fx, fy, fz, seed)
                if shaft_v > 0.58 and y > 10:
                    shaft_w = _vn(fx * 0.25, fz * 0.25, seed ^ 0x9900)
                    if shaft_w > 0.10:
                        blk[dx, y, dz] = 0
                        continue

                # Near-surface cave openings
                if y >= height - 10:
                    open_n = _noise3(fx, fy, fz, seed ^ 0xBBCC)
                    if open_n > 0.32:
                        blk[dx, y, dz] = 0

            # River carving
            if _river_y > 0:
                _river_d = river_depth(wx, wz, seed)
                carve_river(blk, dx, dz, height, _river_y, _river_d,
                            ID_WATER, ID_DIRT, H)

            # Ore placement (more variety — 6 attempts per column)
            for _ in range(6):
                lcg = (lcg * 1664525 + 1013904223) & 0xFFFFFFFF
                oy  = BEDROCK_LEVEL + 3 + int(lcg & 0x3F) % (height - BEDROCK_LEVEL - 6)
                oy  = max(BEDROCK_LEVEL + 3, min(oy, height - 4))
                if blk[dx, oy, dz] == ID_STONE:
                    lcg = (lcg * 1664525 + 1013904223) & 0xFFFFFFFF
                    rv  = lcg & 0xFF
                    if   rv < 18: blk[dx, oy, dz] = ID_COAL
                    elif rv < 30: blk[dx, oy, dz] = ID_IRON
                    elif rv < 33 and oy < SEA_LEVEL - 6:  blk[dx, oy, dz] = ID_GOLD
                    elif rv < 35 and oy < SEA_LEVEL - 14: blk[dx, oy, dz] = ID_DIAMOND

    # ── Surface features ─────────────────────────────────────────────────────
    for dx in range(2, S - 3):
        for dz in range(2, S - 3):
            wx    = cx * S + dx
            wz    = cz * S + dz
            biome = _biome_at(wx, wz, seed)
            lcg   = (lcg * 1664525 + 1013904223) & 0xFFFFFFFF

            if biome == 4:
                lcg = place_ocean_plants(blk, dx, dz, wx, wz, seed, lcg,
                    ID_WATER, ID_KELP, ID_SEAGRASS)

            h  = _terrain_height(wx, wz, seed)
            ty = h + 1
            if ty + 7 >= H: continue
            if _is_coast(wx, wz, seed): continue
            if h <= SEA_LEVEL: continue
            if blk[dx, h, dz] == ID_WATER: continue
            if river_info(wx, wz, h, seed) > 0: continue

            # Desert cactus
            if biome == 2:
                if blk[dx, h, dz] != ID_SAND: continue
                if (lcg & 0xFF) >= 12: continue
                cact_h = 2 + (lcg & 0x3)
                for y in range(ty, min(ty + cact_h, H)):
                    blk[dx, y, dz] = ID_CACTUS
                continue

            # Oak trees (plains / forest)
            if biome in (0, 1):
                thresh = 5 if biome == 1 else 2
                if (lcg & 0xFF) >= thresh: continue
                trunk_h = 4 + (lcg & 0x3)   # variable trunk height 4-7
                for y in range(ty, min(ty + trunk_h, H)):
                    blk[dx, y, dz] = ID_LOG
                for ldy in range(trunk_h - 2, trunk_h + 3):
                    radius = 3 if ldy < trunk_h else 2
                    for ldx in range(-radius, radius + 1):
                        for ldz in range(-radius, radius + 1):
                            if abs(ldx) + abs(ldz) <= radius + 1:
                                lx_ = dx + ldx; lz_ = dz + ldz; ly_ = ty + ldy
                                if 0 <= lx_ < S and 0 <= lz_ < S and ly_ < H:
                                    if blk[lx_, ly_, lz_] == 0:
                                        blk[lx_, ly_, lz_] = ID_LEAVES
                continue

            # Birch trees (birch forest biome)
            if biome == 5:
                if (lcg & 0xFF) >= 8: continue   # dense birch
                trunk_h = 5 + (lcg & 0x3)        # birch is taller: 5-8
                for y in range(ty, min(ty + trunk_h, H)):
                    blk[dx, y, dz] = ID_BIRCH_LOG
                # Birch canopy: smaller, more oval than oak
                for ldy in range(trunk_h - 2, trunk_h + 2):
                    radius = 2 if ldy < trunk_h else 1
                    for ldx in range(-radius, radius + 1):
                        for ldz in range(-radius, radius + 1):
                            if abs(ldx) + abs(ldz) <= radius + 1:
                                lx_ = dx + ldx; lz_ = dz + ldz; ly_ = ty + ldy
                                if 0 <= lx_ < S and 0 <= lz_ < S and ly_ < H:
                                    if blk[lx_, ly_, lz_] == 0:
                                        blk[lx_, ly_, lz_] = ID_BIRCH_LEAVES
                continue

            # Spruce trees (snowy taiga)
            if biome == 3:
                if height >= SEA_LEVEL + 22: continue
                if (lcg & 0xFF) >= 4: continue
                trunk_h = 6 + (lcg & 0x3)
                for y in range(ty, min(ty + trunk_h, H)):
                    blk[dx, y, dz] = ID_SPRUCE_LOG
                for ldy in range(2, trunk_h + 1):
                    radius = max(1, 3 - (ldy * 2 // trunk_h))
                    for ldx in range(-radius, radius + 1):
                        for ldz in range(-radius, radius + 1):
                            if abs(ldx) + abs(ldz) <= radius + 1:
                                lx_ = dx + ldx; lz_ = dz + ldz; ly_ = ty + ldy
                                if 0 <= lx_ < S and 0 <= lz_ < S and ly_ < H:
                                    if blk[lx_, ly_, lz_] == 0:
                                        blk[lx_, ly_, lz_] = ID_SPRUCE_LEAVES

    return blk


# ═══════════════════════════════════════════════════════════════════════
#  MESH BUILDER  (unchanged — receives BT_OCCLUDE not BT_SOLID)
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def build_mesh(blk, nb_x_neg, nb_x_pos, nb_z_neg, nb_z_pos,
               render, occlude, trans, rot, colors, face_uv,
               cx, cz, greedy, show_all):
    """
    Build a flat float32 vertex buffer for one chunk.

    Vertex layout per vertex (12 floats):
        px py pz  nx ny nz  cr cg cb ca  u v

    occlude must be BT_OCCLUDE (solid AND NOT transparent) so transparent
    blocks like water, kelp, leaves never hide adjacent faces.
    """
    S  = CHUNK_SIZE;  H = CHUNK_HEIGHT
    ox = float(cx * S);  oz = float(cz * S)

    max_verts = S * H * S * 6 * 6
    buf = np.empty(max_verts * 12, dtype=np.float32)
    n   = 0

    FACES = (
        ( 0.0,  1.0,  0.0,  1, 0, 2, 1,  1),   # top    (+Y)
        ( 0.0, -1.0,  0.0,  1, 0, 2, 0, -1),   # bottom (-Y)
        ( 1.0,  0.0,  0.0,  0, 2, 1, 1,  1),   # right  (+X)
        (-1.0,  0.0,  0.0,  0, 2, 1, 0, -1),   # left   (-X)
        ( 0.0,  0.0,  1.0,  2, 0, 1, 1,  1),   # front  (+Z)
        ( 0.0,  0.0, -1.0,  2, 0, 1, 0, -1),   # back   (-Z)
    )

    for fi in range(6):
        fd   = FACES[fi]
        nx_  = fd[0]; ny_ = fd[1]; nz_ = fd[2]
        axis = int(fd[3]); a_ax = int(fd[4]); b_ax = int(fd[5])
        foff = int(fd[6]); ndir = int(fd[7])
        flip = (ndir < 0)
        sz   = [S, H, S]
        d_i  = sz[axis]; d_a = sz[a_ax]; d_b = sz[b_ax]

        for i in range(d_i):
            ni   = i + ndir
            mask = np.zeros((d_a, d_b), dtype=np.uint8)

            for a in range(d_a):
                for b in range(d_b):
                    idx3 = [0, 0, 0]
                    idx3[axis] = i; idx3[a_ax] = a; idx3[b_ax] = b
                    lx = idx3[0]; ly = idx3[1]; lz = idx3[2]
                    bt = blk[lx, ly, lz]
                    if bt == 0 or render[bt] == 0:
                        continue

                    # Transparent-only same-block suppression: remove internal
                    # faces between water-water (or leaf-leaf) neighbours only.
                    # This fixes X-ray grid lines without affecting solid blocks.
                    if trans[bt] == 1:
                        if 0 <= ni < d_i:
                            _ni2 = [0, 0, 0]
                            _ni2[axis] = ni; _ni2[a_ax] = a; _ni2[b_ax] = b
                            if blk[_ni2[0], _ni2[1], _ni2[2]] == bt:
                                continue
                        else:
                            if axis == 0:
                                _nb2 = nb_x_neg[a, b] if ndir < 0 else nb_x_pos[a, b]
                                # 0 = unloaded chunk: suppress face (avoids ghost walls)
                                if _nb2 == bt: continue
                            elif axis == 2:
                                _nb2 = nb_z_neg[a, b] if ndir < 0 else nb_z_pos[a, b]
                                if _nb2 == bt: continue

                    if show_all == 0:
                        if 0 <= ni < d_i:
                            nidx = [0, 0, 0]
                            nidx[axis] = ni; nidx[a_ax] = a; nidx[b_ax] = b
                            nbt = blk[nidx[0], nidx[1], nidx[2]]
                            if occlude[nbt] == 1 or nbt == bt:
                                continue
                        else:
                            if axis == 0:
                                nbt = nb_x_neg[a, b] if ndir < 0 else nb_x_pos[a, b]
                                if occlude[nbt] == 1 or nbt == bt:
                                    continue
                            elif axis == 2:
                                nbt = nb_z_neg[a, b] if ndir < 0 else nb_z_pos[a, b]
                                if occlude[nbt] == 1 or nbt == bt:
                                    continue

                    mask[a, b] = bt

            done = np.zeros((d_a, d_b), dtype=np.uint8)
            for a in range(d_a):
                for b in range(d_b):
                    bt = mask[a, b]
                    if bt == 0 or done[a, b]:
                        continue
                    bw = 1; ah = 1

                    # Transparent blocks (water, leaves) must NOT be greedy-merged
                    # because greedy stretches the UV across the merged quad, causing
                    # banding/stripe artifacts on water surfaces and leaf clusters.
                    use_greedy = greedy != 0 and trans[bt] == 0
                    if use_greedy:
                        while b + bw < d_b and mask[a, b + bw] == bt and done[a, b + bw] == 0:
                            bw += 1
                        ok2 = True
                        while ok2 and a + ah < d_a:
                            for bb in range(b, b + bw):
                                if mask[a + ah, bb] != bt or done[a + ah, bb] != 0:
                                    ok2 = False; break
                            if ok2: ah += 1
                        for aa in range(a, a + ah):
                            for bb in range(b, b + bw):
                                done[aa, bb] = 1
                    else:
                        done[a, b] = 1

                    cr = colors[bt, fi, 0]; cg = colors[bt, fi, 1]
                    cb2= colors[bt, fi, 2]; ca = colors[bt, fi, 3]
                    u0 = face_uv[bt, fi, 0]; v0 = face_uv[bt, fi, 1]
                    u1 = face_uv[bt, fi, 2]; v1 = face_uv[bt, fi, 3]

                    rdir = rot[bt, fi]
                    if rdir == 1:
                        uv0=(u0,v1); uv1=(u1,v1); uv2=(u1,v0); uv3=(u0,v0)
                    elif rdir == -1:
                        uv0=(u1,v0); uv1=(u0,v0); uv2=(u0,v1); uv3=(u1,v1)
                    else:
                        uv0=(u0,v0); uv1=(u0,v1); uv2=(u1,v1); uv3=(u1,v0)

                    face_i = float(i + foff)
                    if axis == 0:
                        v0x,v0y,v0z = face_i+ox, float(b),      float(a)    +oz
                        v1x,v1y,v1z = face_i+ox, float(b),      float(a+ah) +oz
                        v2x,v2y,v2z = face_i+ox, float(b+bw),   float(a+ah) +oz
                        v3x,v3y,v3z = face_i+ox, float(b+bw),   float(a)    +oz
                    elif axis == 1:
                        v0x,v0y,v0z = float(a)   +ox, face_i, float(b)    +oz
                        v1x,v1y,v1z = float(a+ah)+ox, face_i, float(b)    +oz
                        v2x,v2y,v2z = float(a+ah)+ox, face_i, float(b+bw) +oz
                        v3x,v3y,v3z = float(a)   +ox, face_i, float(b+bw) +oz
                    else:
                        v0x,v0y,v0z = float(a)   +ox, float(b),    face_i+oz
                        v1x,v1y,v1z = float(a+ah)+ox, float(b),    face_i+oz
                        v2x,v2y,v2z = float(a+ah)+ox, float(b+bw), face_i+oz
                        v3x,v3y,v3z = float(a)   +ox, float(b+bw), face_i+oz

                    if not flip:
                        for vx,vy,vz,uv in (
                            (v0x,v0y,v0z,uv0),(v1x,v1y,v1z,uv1),(v2x,v2y,v2z,uv2),
                            (v0x,v0y,v0z,uv0),(v2x,v2y,v2z,uv2),(v3x,v3y,v3z,uv3)):
                            if n + 12 <= buf.shape[0]:
                                buf[n]=vx; buf[n+1]=vy; buf[n+2]=vz
                                buf[n+3]=nx_; buf[n+4]=ny_; buf[n+5]=nz_
                                buf[n+6]=cr; buf[n+7]=cg; buf[n+8]=cb2; buf[n+9]=ca
                                buf[n+10]=uv[0]; buf[n+11]=uv[1]; n+=12
                    else:
                        for vx,vy,vz,uv in (
                            (v0x,v0y,v0z,uv0),(v2x,v2y,v2z,uv2),(v1x,v1y,v1z,uv1),
                            (v0x,v0y,v0z,uv0),(v3x,v3y,v3z,uv3),(v2x,v2y,v2z,uv2)):
                            if n + 12 <= buf.shape[0]:
                                buf[n]=vx; buf[n+1]=vy; buf[n+2]=vz
                                buf[n+3]=nx_; buf[n+4]=ny_; buf[n+5]=nz_
                                buf[n+6]=cr; buf[n+7]=cg; buf[n+8]=cb2; buf[n+9]=ca
                                buf[n+10]=uv[0]; buf[n+11]=uv[1]; n+=12

    return buf[:n].copy()

@njit(cache=True)
def build_mesh_split(blk, nb_x_neg, nb_x_pos, nb_z_neg, nb_z_pos,
                     render, occlude, trans, liquid, cross, rot, colors, face_uv,
                     cx, cz, greedy):
    """
    Same as build_mesh but returns (opaque_buf, trans_buf) as two separate
    float32 arrays. Opaque faces go into buf_o, transparent (water/leaves)
    into buf_t. This enables proper two-pass rendering:
      Pass 1: opaque with depth-write ON
      Pass 2: transparent with depth-write OFF (no self-occlusion artifacts)
    """
    S  = CHUNK_SIZE;  H = CHUNK_HEIGHT
    ox = float(cx * S);  oz = float(cz * S)

    max_verts = S * H * S * 6 * 6
    buf_o = np.empty(max_verts * 12, dtype=np.float32)
    buf_t = np.empty(max_verts * 12, dtype=np.float32)
    no = 0;  nt = 0

    FACES = (
        ( 0.0,  1.0,  0.0,  1, 0, 2, 1,  1),
        ( 0.0, -1.0,  0.0,  1, 0, 2, 0, -1),
        ( 1.0,  0.0,  0.0,  0, 2, 1, 1,  1),
        (-1.0,  0.0,  0.0,  0, 2, 1, 0, -1),
        ( 0.0,  0.0,  1.0,  2, 0, 1, 1,  1),
        ( 0.0,  0.0, -1.0,  2, 0, 1, 0, -1),
    )

    for fi in range(6):
        fd   = FACES[fi]
        nx_  = fd[0]; ny_ = fd[1]; nz_ = fd[2]
        axis = int(fd[3]); a_ax = int(fd[4]); b_ax = int(fd[5])
        foff = int(fd[6]); ndir = int(fd[7])
        flip = (ndir < 0)
        sz   = [S, H, S]
        d_i  = sz[axis]; d_a = sz[a_ax]; d_b = sz[b_ax]

        for i in range(d_i):
            ni   = i + ndir
            mask = np.zeros((d_a, d_b), dtype=np.uint8)

            for a in range(d_a):
                for b in range(d_b):
                    idx3 = [0, 0, 0]
                    idx3[axis] = i; idx3[a_ax] = a; idx3[b_ax] = b
                    lx = idx3[0]; ly = idx3[1]; lz = idx3[2]
                    bt = blk[lx, ly, lz]
                    if bt == 0 or render[bt] == 0:
                        continue

                    # Cross billboards (torch-like)
                    if cross[bt] == 1:
                        if fi != 0:
                            continue
                        u0 = face_uv[bt, 0, 0]; v0 = face_uv[bt, 0, 1]
                        u1 = face_uv[bt, 0, 2]; v1 = face_uv[bt, 0, 3]
                        cr = colors[bt, 0, 0]; cg = colors[bt, 0, 1]
                        cb2= colors[bt, 0, 2]; ca = colors[bt, 0, 3]
                        nx_ = 0.0; ny_ = 1.0; nz_ = 0.0
                        cx2 = float(lx) + 0.5 + ox
                        cz2 = float(lz) + 0.5 + oz
                        y0 = float(ly); y1 = float(ly) + 0.9
                        w = 0.6
                        x0 = cx2 - w * 0.5; x1 = cx2 + w * 0.5
                        z0 = cz2 - w * 0.5; z1 = cz2 + w * 0.5
                        uv0 = (u0, v1); uv1 = (u0, v0); uv2 = (u1, v0); uv3 = (u1, v1)

                        if trans[bt] == 1:
                            buf = buf_t; n = nt
                        else:
                            buf = buf_o; n = no

                        for vx, vy, vz, uv in (
                            (x0, y0, z0, uv0), (x0, y1, z0, uv1), (x1, y1, z1, uv2),
                            (x0, y0, z0, uv0), (x1, y1, z1, uv2), (x1, y0, z1, uv3),
                            (x0, y0, z1, uv0), (x0, y1, z1, uv1), (x1, y1, z0, uv2),
                            (x0, y0, z1, uv0), (x1, y1, z0, uv2), (x1, y0, z0, uv3),
                        ):
                            if n + 12 <= buf.shape[0]:
                                buf[n]=vx; buf[n+1]=vy; buf[n+2]=vz
                                buf[n+3]=nx_; buf[n+4]=ny_; buf[n+5]=nz_
                                buf[n+6]=cr; buf[n+7]=cg; buf[n+8]=cb2; buf[n+9]=ca
                                buf[n+10]=uv[0]; buf[n+11]=uv[1]; n+=12
                        if trans[bt] == 1:
                            nt = n
                        else:
                            no = n
                        continue

                    # Transparent: suppress only same-block faces
                    if trans[bt] == 1:
                        if 0 <= ni < d_i:
                            _ni2 = [0, 0, 0]
                            _ni2[axis] = ni; _ni2[a_ax] = a; _ni2[b_ax] = b
                            if blk[_ni2[0], _ni2[1], _ni2[2]] == bt: continue
                        else:
                            if axis == 0:
                                _nb2 = nb_x_neg[a, b] if ndir < 0 else nb_x_pos[a, b]
                                if _nb2 == bt: continue
                                if liquid[bt] == 1 and _nb2 == 0: continue
                            elif axis == 2:
                                _nb2 = nb_z_neg[a, b] if ndir < 0 else nb_z_pos[a, b]
                                if _nb2 == bt: continue
                                if liquid[bt] == 1 and _nb2 == 0: continue

                    # Opaque: normal face culling
                    if trans[bt] == 0:
                        if 0 <= ni < d_i:
                            nidx = [0, 0, 0]
                            nidx[axis] = ni; nidx[a_ax] = a; nidx[b_ax] = b
                            nbt = blk[nidx[0], nidx[1], nidx[2]]
                            if occlude[nbt] == 1 or nbt == bt:
                                continue
                        else:
                            if axis == 0:
                                nbt = nb_x_neg[a, b] if ndir < 0 else nb_x_pos[a, b]
                                if occlude[nbt] == 1 or nbt == bt:
                                    continue
                            elif axis == 2:
                                nbt = nb_z_neg[a, b] if ndir < 0 else nb_z_pos[a, b]
                                if occlude[nbt] == 1 or nbt == bt:
                                    continue

                    mask[a, b] = bt

            done = np.zeros((d_a, d_b), dtype=np.uint8)
            for a in range(d_a):
                for b in range(d_b):
                    bt = mask[a, b]
                    if bt == 0 or done[a, b]:
                        continue
                    bw = 1; ah = 1

                    # No greedy merge for transparent blocks
                    use_greedy = greedy != 0 and trans[bt] == 0
                    if use_greedy:
                        while b + bw < d_b and mask[a, b + bw] == bt and done[a, b + bw] == 0:
                            bw += 1
                        ok2 = True
                        while ok2 and a + ah < d_a:
                            for bb in range(b, b + bw):
                                if mask[a + ah, bb] != bt or done[a + ah, bb] != 0:
                                    ok2 = False; break
                            if ok2: ah += 1
                        for aa in range(a, a + ah):
                            for bb in range(b, b + bw):
                                done[aa, bb] = 1
                    else:
                        done[a, b] = 1

                    cr = colors[bt, fi, 0]; cg = colors[bt, fi, 1]
                    cb2= colors[bt, fi, 2]; ca = colors[bt, fi, 3]
                    u0 = face_uv[bt, fi, 0]; v0 = face_uv[bt, fi, 1]
                    u1 = face_uv[bt, fi, 2]; v1 = face_uv[bt, fi, 3]

                    rdir = rot[bt, fi]
                    if rdir == 1:
                        uv0=(u0,v1); uv1=(u1,v1); uv2=(u1,v0); uv3=(u0,v0)
                    elif rdir == -1:
                        uv0=(u1,v0); uv1=(u0,v0); uv2=(u0,v1); uv3=(u1,v1)
                    else:
                        uv0=(u0,v0); uv1=(u0,v1); uv2=(u1,v1); uv3=(u1,v0)

                    face_i = float(i + foff)
                    if axis == 0:
                        v0x,v0y,v0z = face_i+ox, float(b),      float(a)    +oz
                        v1x,v1y,v1z = face_i+ox, float(b),      float(a+ah) +oz
                        v2x,v2y,v2z = face_i+ox, float(b+bw),   float(a+ah) +oz
                        v3x,v3y,v3z = face_i+ox, float(b+bw),   float(a)    +oz
                    elif axis == 1:
                        v0x,v0y,v0z = float(a)   +ox, face_i, float(b)    +oz
                        v1x,v1y,v1z = float(a+ah)+ox, face_i, float(b)    +oz
                        v2x,v2y,v2z = float(a+ah)+ox, face_i, float(b+bw) +oz
                        v3x,v3y,v3z = float(a)   +ox, face_i, float(b+bw) +oz
                    else:
                        v0x,v0y,v0z = float(a)   +ox, float(b),    face_i+oz
                        v1x,v1y,v1z = float(a+ah)+ox, float(b),    face_i+oz
                        v2x,v2y,v2z = float(a+ah)+ox, float(b+bw), face_i+oz
                        v3x,v3y,v3z = float(a)   +ox, float(b+bw), face_i+oz

                    is_trans = trans[bt] == 1
                    if not flip:
                        for vx,vy,vz,uv in (
                            (v0x,v0y,v0z,uv0),(v1x,v1y,v1z,uv1),(v2x,v2y,v2z,uv2),
                            (v0x,v0y,v0z,uv0),(v2x,v2y,v2z,uv2),(v3x,v3y,v3z,uv3)):
                            if is_trans:
                                if nt + 12 <= buf_t.shape[0]:
                                    buf_t[nt]=vx; buf_t[nt+1]=vy; buf_t[nt+2]=vz
                                    buf_t[nt+3]=nx_; buf_t[nt+4]=ny_; buf_t[nt+5]=nz_
                                    buf_t[nt+6]=cr; buf_t[nt+7]=cg; buf_t[nt+8]=cb2; buf_t[nt+9]=ca
                                    buf_t[nt+10]=uv[0]; buf_t[nt+11]=uv[1]; nt+=12
                            else:
                                if no + 12 <= buf_o.shape[0]:
                                    buf_o[no]=vx; buf_o[no+1]=vy; buf_o[no+2]=vz
                                    buf_o[no+3]=nx_; buf_o[no+4]=ny_; buf_o[no+5]=nz_
                                    buf_o[no+6]=cr; buf_o[no+7]=cg; buf_o[no+8]=cb2; buf_o[no+9]=ca
                                    buf_o[no+10]=uv[0]; buf_o[no+11]=uv[1]; no+=12
                    else:
                        for vx,vy,vz,uv in (
                            (v0x,v0y,v0z,uv0),(v2x,v2y,v2z,uv2),(v1x,v1y,v1z,uv1),
                            (v0x,v0y,v0z,uv0),(v3x,v3y,v3z,uv3),(v2x,v2y,v2z,uv2)):
                            if is_trans:
                                if nt + 12 <= buf_t.shape[0]:
                                    buf_t[nt]=vx; buf_t[nt+1]=vy; buf_t[nt+2]=vz
                                    buf_t[nt+3]=nx_; buf_t[nt+4]=ny_; buf_t[nt+5]=nz_
                                    buf_t[nt+6]=cr; buf_t[nt+7]=cg; buf_t[nt+8]=cb2; buf_t[nt+9]=ca
                                    buf_t[nt+10]=uv[0]; buf_t[nt+11]=uv[1]; nt+=12
                            else:
                                if no + 12 <= buf_o.shape[0]:
                                    buf_o[no]=vx; buf_o[no+1]=vy; buf_o[no+2]=vz
                                    buf_o[no+3]=nx_; buf_o[no+4]=ny_; buf_o[no+5]=nz_
                                    buf_o[no+6]=cr; buf_o[no+7]=cg; buf_o[no+8]=cb2; buf_o[no+9]=ca
                                    buf_o[no+10]=uv[0]; buf_o[no+11]=uv[1]; no+=12

    return buf_o[:no].copy(), buf_t[:nt].copy()