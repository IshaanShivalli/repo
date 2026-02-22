"""
terrain.py  –  Numba-JIT terrain generation (gen_chunk) and chunk mesh builder.

Biomes:
  0 = plains      (grass, oak trees)
  1 = forest      (grass, dense oak trees)
  2 = desert      (sand, cactus)
  3 = snowy taiga (snow, spruce trees)
  4 = ocean       (deep water, sand floor, kelp, seagrass)
"""

import math
import numpy as np
from numba import njit

from settings import CHUNK_SIZE, CHUNK_HEIGHT, SEA_LEVEL, BEDROCK_LEVEL, OCEAN_FLOOR
from sea import fill_ocean_column, place_ocean_plants, ocean_floor_height
from inland_water import river_info, carve_river


# ═══════════════════════════════════════════════════════════════════════
#  NOISE HELPERS
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _noise2(x: float, z: float, seed: int) -> float:
    a = math.sin(x * 0.08 + seed * 1e-9) * math.cos(z * 0.08 + seed * 1e-9)
    b = math.sin(x * 0.21 + seed * 2e-9) * math.cos(z * 0.17 + seed * 2e-9)
    return a * 0.7 + b * 0.3


@njit(cache=True)
def _noise3(x: float, y: float, z: float, seed: int) -> float:
    # Multi-octave 3-D noise for natural, varied cave shapes
    n  = (math.sin(x * 0.12 + seed * 1e-9) *
          math.cos(z * 0.12 + seed * 1e-9) *
          math.sin(y * 0.18 + seed * 2e-9))
    n2 = (math.sin(x * 0.07 + seed * 3e-9) *
          math.cos(z * 0.09 + seed * 4e-9) *
          math.cos(y * 0.11 + seed * 5e-9)) * 0.6
    n3 = (math.sin(x * 0.23 + seed * 6e-9) *
          math.cos(z * 0.19 + seed * 7e-9) *
          math.sin(y * 0.27 + seed * 8e-9)) * 0.3
    return n + n2 + n3


@njit(cache=True)
def _cave_noise_v(x: float, y: float, z: float, seed: int) -> float:
    """Secondary vertical-tunnel noise — creates shafts that reach the surface."""
    a = math.sin(x * 0.15 + seed * 9e-9) * math.cos(z * 0.15 + seed * 1e-8)
    b = math.sin(x * 0.08 + seed * 1.1e-8) * math.cos(z * 0.08 + seed * 1.2e-8)
    # Y-independent: same value for a whole column → long vertical shaft
    return a * 0.6 + b * 0.4


@njit(cache=True)
def _continent(wx: int, wz: int, seed: int) -> float:
    """Large-scale continent value: negative = ocean, positive = land."""
    return (_noise2(float(wx) * 0.006, float(wz) * 0.006, seed ^ 0x1234) +
            _noise2(float(wx) * 0.002, float(wz) * 0.002, seed ^ 0x5678) * 0.5)


@njit(cache=True)
def _terrain_height(wx: int, wz: int, seed: int) -> int:
    n     = _noise2(float(wx), float(wz), seed)
    hills = (math.sin(float(wx) * 0.01 + seed * 1e-9) *
             math.cos(float(wz) * 0.01 + seed * 2e-9))
    h = SEA_LEVEL + 5 + int(n * 8 + hills * 4)
    return max(BEDROCK_LEVEL + 3, min(CHUNK_HEIGHT - 4, h))


@njit(cache=True)
def _biome_at(wx: int, wz: int, seed: int) -> int:
    """
    Returns biome index:
      0 = plains  1 = forest  2 = desert  3 = snowy taiga  4 = ocean
    """
    if _continent(wx, wz, seed) < -0.45:
        return 4   # ocean
    v = _noise2(float(wx) * 0.03, float(wz) * 0.03, seed)
    if v < -0.3: return 2   # desert
    if v <  0.2: return 0   # plains
    if v <  0.5: return 1   # forest
    # Snowy taiga only at high elevation
    h = _terrain_height(wx, wz, seed)
    if h >= SEA_LEVEL + 18: return 3  # snowy taiga (high ground only)
    return 1                 # forest instead of snow at low elevation


@njit(cache=True)
def _is_coast(wx: int, wz: int, seed: int) -> bool:
    """True when a land cell is close to the ocean boundary (within ~6 blocks)."""
    c = _continent(wx, wz, seed)
    if c < -0.45:
        return False   # already ocean
    # Sample a small neighbourhood; if any neighbour is ocean we're at the coast
    for ddx in (-6, -3, 0, 3, 6):
        for ddz in (-6, -3, 0, 3, 6):
            if _continent(wx + ddx, wz + ddz, seed) < -0.45:
                return True
    return False


@njit(cache=True)
def ocean_floor_height(wx: int, wz: int, seed: int) -> int:
    """Varied ocean floor between OCEAN_FLOOR and SEA_LEVEL-4."""
    n = _noise2(float(wx) * 0.05, float(wz) * 0.05, seed ^ 0xABCD)
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
              ID_SAND_OCEAN: int, ID_KELP: int, ID_SEAGRASS: int) -> np.ndarray:
    S   = CHUNK_SIZE
    H   = CHUNK_HEIGHT
    blk = np.zeros((S, H, S), dtype=np.uint8)
    lcg = (seed ^ (cx * 73856093) ^ (cz * 19349663)) & 0xFFFFFFFF

    for dx in range(S):
        for dz in range(S):
            wx    = cx * S + dx
            wz    = cz * S + dz
            biome = _biome_at(wx, wz, seed)

                        # ── Ocean biome (see water_gen.py) ───────────────────
            if biome == 4:
                lcg = fill_ocean_column(blk, dx, dz, wx, wz, seed, lcg,
                    ID_BEDROCK, ID_STONE, ID_SAND_OCEAN, ID_WATER)

                # Ore pockets in ocean stone
                floor_y = ocean_floor_height(wx, wz, seed)  # recompute for ores
                for _ in range(3):
                    lcg = (lcg * 1664525 + 1013904223) & 0xFFFFFFFF
                    oy  = BEDROCK_LEVEL + 3 + int(lcg & 0x1F)
                    oy  = min(oy, floor_y - 4)
                    if oy >= BEDROCK_LEVEL + 3 and blk[dx, oy, dz] == ID_STONE:
                        lcg = (lcg * 1664525 + 1013904223) & 0xFFFFFFFF
                        rv  = lcg & 0xFF
                        if   rv < 20: blk[dx, oy, dz] = ID_COAL
                        elif rv < 32: blk[dx, oy, dz] = ID_IRON
                        elif rv < 35 and oy < SEA_LEVEL - 8: blk[dx, oy, dz] = ID_GOLD
                        elif rv < 37 and oy < SEA_LEVEL - 16: blk[dx, oy, dz] = ID_DIAMOND
                continue

            # ── Land biomes ──────────────────────────────────────────────
            height = _terrain_height(wx, wz, seed)
            # Coast detection: land cells within ~6 blocks of ocean get sand
            # surface and subsurface, regardless of biome.
            coast = _is_coast(wx, wz, seed)
            # River carving (inland_water.py)
            _river_y = -1
            if biome not in (2, 3) and not coast:
                _river_y = river_info(wx, wz, height, seed)

            if coast:
                surf = ID_SAND
            elif biome == 2:
                surf = ID_SAND
            elif biome == 3:
                # High peaks: stone only above certain height
                if height >= SEA_LEVEL + 28:
                    surf = ID_STONE
                else:
                    surf = ID_SNOW
            else:
                surf = ID_GRASS

            for y in range(H):
                if y <= BEDROCK_LEVEL:
                    blk[dx, y, dz] = ID_BEDROCK
                elif y <= BEDROCK_LEVEL + 2:
                    lcg = (lcg * 1664525 + 1013904223) & 0xFFFFFFFF
                    blk[dx, y, dz] = ID_BEDROCK if (lcg & 0xFF) < 180 else ID_STONE
                elif y < height - 3:
                    blk[dx, y, dz] = ID_STONE
                elif y < height:
                    # Coast: sand; high snow peaks: stone; else dirt
                    if coast:
                        blk[dx, y, dz] = ID_SAND
                    elif biome == 3 and height >= SEA_LEVEL + 28:
                        blk[dx, y, dz] = ID_STONE
                    else:
                        blk[dx, y, dz] = ID_DIRT
                elif y == height:
                    blk[dx, y, dz] = surf

            # ── Cave carving (land only) ──────────────────────────────────
            # Big irregular chambers: low threshold on combined multi-octave noise.
            # Vertical shaft noise: columns where shaft_v > threshold become open
            # from BEDROCK up to near the surface, creating sky-shafts / ravines.
            shaft_v = _cave_noise_v(float(wx), 0.0, float(wz), seed)
            is_shaft = shaft_v > 0.60   # ~10% of columns get a shaft

            # Some caves are big (25% chance per column)
            big_cave = _noise2(float(wx) * 0.07, float(wz) * 0.07, seed ^ 0xC4FE) > 0.5
            cave_thresh = 0.36 if big_cave else 0.44
            # Cave opening: near-surface shaft that breaks through to sky
            opening_noise = _noise2(float(wx) * 0.15, float(wz) * 0.15, seed ^ 0xABCD)
            has_opening = opening_noise > 0.72

            for y in range(BEDROCK_LEVEL + 3, height):
                cv = _noise3(float(wx), float(y), float(wz), seed)
                if cv > cave_thresh:
                    blk[dx, y, dz] = 0
                    continue
                # Cave opening: carve from surface down 8 blocks in opening columns
                if has_opening and y >= height - 8 and y < height:
                    open_cv = _noise3(float(wx), float(y), float(wz), seed ^ 0x77)
                    if open_cv > 0.35:
                        blk[dx, y, dz] = 0
                        continue

                # Vertical shaft: narrow column carved from deep up toward surface.
                # Only activate above y=10 so we don't punch through bedrock.
                if is_shaft and y > 10:
                    # Shaft width varies with a slow horizontal noise
                    shaft_w = _noise2(float(wx) * 0.3, float(wz) * 0.3, seed ^ 0xCAFE)
                    # shaft_w in [-1,1]; only carve when it's above 0.15 (medium-wide shafts)
                    if shaft_w > 0.15:
                        blk[dx, y, dz] = 0

            # River carving — applied AFTER terrain y-loop
            if _river_y > 0:
                carve_river(blk, dx, dz, height, _river_y,
                            ID_WATER, ID_DIRT, H)

            # Ore placement
            for _ in range(4):
                lcg = (lcg * 1664525 + 1013904223) & 0xFFFFFFFF
                oy  = BEDROCK_LEVEL + 3 + int(lcg & 0x1F)
                oy  = min(oy, height - 4)
                if blk[dx, oy, dz] == ID_STONE:
                    lcg = (lcg * 1664525 + 1013904223) & 0xFFFFFFFF
                    rv  = lcg & 0xFF
                    if   rv < 16: blk[dx, oy, dz] = ID_COAL
                    elif rv < 28: blk[dx, oy, dz] = ID_IRON
                    elif rv < 31 and oy < SEA_LEVEL - 8:  blk[dx, oy, dz] = ID_GOLD
                    elif rv < 33 and oy < SEA_LEVEL - 16: blk[dx, oy, dz] = ID_DIAMOND

    # ── Surface features: trees, cactus, kelp, seagrass ─────────────────────
    for dx in range(2, S - 3):
        for dz in range(2, S - 3):
            wx    = cx * S + dx
            wz    = cz * S + dz
            biome = _biome_at(wx, wz, seed)
            lcg   = (lcg * 1664525 + 1013904223) & 0xFFFFFFFF

            # ── Ocean underwater plants (see water_gen.py) ────────────
            if biome == 4:
                lcg = place_ocean_plants(blk, dx, dz, wx, wz, seed, lcg,
                    ID_WATER, ID_KELP, ID_SEAGRASS)

            # ── Land surface features ────────────────────────────────────────
            h  = _terrain_height(wx, wz, seed)
            ty = h + 1
            if ty + 7 >= H: continue

            # No trees or cactus on coastal sand, water, or submerged terrain
            if _is_coast(wx, wz, seed): continue
            if h <= SEA_LEVEL: continue
            if blk[dx, h, dz] == ID_WATER: continue
            # No surface features on rivers
            if river_info(wx, wz, h, seed) > 0: continue

            # Desert cactus
            if biome == 2:
                if blk[dx, h, dz] != ID_SAND: continue
                if (lcg & 0xFF) >= 10: continue
                cact_h = 2 + (lcg & 0x3)
                for y in range(ty, min(ty + cact_h, H)):
                    blk[dx, y, dz] = ID_CACTUS
                continue

            # Oak tree (plains / forest)
            if biome in (0, 1):
                thresh = 4 if biome == 1 else 2
                if (lcg & 0xFF) >= thresh: continue
                for y in range(ty, ty + 5):
                    if 0 <= dx < S and 0 <= dz < S:
                        blk[dx, y, dz] = ID_LOG
                for ldy in range(4, 8):
                    for ldx in range(-3, 4):
                        for ldz in range(-3, 4):
                            if abs(ldx) + abs(ldz) <= 4:
                                lx_ = dx + ldx; lz_ = dz + ldz; ly_ = ty + ldy
                                if 0 <= lx_ < S and 0 <= lz_ < S and ly_ < H:
                                    if blk[lx_, ly_, lz_] == 0:
                                        blk[lx_, ly_, lz_] = ID_LEAVES
                continue

            # Tall oak tree (rare, forest biome only)
            if biome == 1:
                if (lcg & 0xFF) != 0: pass  # ~0.4% chance
                else:
                    # 4-wide trunk, 20 blocks tall
                    for y in range(ty, ty + 20):
                        for tx in range(dx, dx + 4):
                            for tz in range(dz, dz + 4):
                                if 0 <= tx < S and 0 <= tz < S and y < H:
                                    blk[tx, y, tz] = ID_LOG
                    # Big canopy at top
                    for ldy in range(16, 24):
                        radius = 6 - max(0, ldy - 20)
                        for ldx in range(-radius, radius + 1):
                            for ldz in range(-radius, radius + 1):
                                if ldx*ldx + ldz*ldz <= (radius+1)*(radius+1):
                                    lx_ = dx + 2 + ldx
                                    lz_ = dz + 2 + ldz
                                    ly_ = ty + ldy
                                    if 0<=lx_<S and 0<=lz_<S and ly_<H:
                                        if blk[lx_, ly_, lz_] == 0:
                                            blk[lx_, ly_, lz_] = ID_LEAVES
                    continue

            # Spruce tree (snowy taiga) - not on stone peaks
            if biome == 3:
                if height >= SEA_LEVEL + 28: continue  # stone peak, no trees
                if (lcg & 0xFF) >= 3: continue
                for y in range(ty, ty + 6):
                    if 0 <= y < H: blk[dx, y, dz] = ID_SPRUCE_LOG
                for ldy in range(3, 8):
                    radius = max(1, 4 - (ldy - 3))
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
                     render, occlude, trans, liquid, rot, colors, face_uv,
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
