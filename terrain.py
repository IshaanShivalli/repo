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
WATER_BLOCK_ID = None

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
    return (math.sin(x * 0.12 + seed * 1e-9) *
            math.cos(z * 0.12 + seed * 1e-9) *
            math.sin(y * 0.18 + seed * 2e-9))


@njit(cache=True)
def _terrain_height(wx: int, wz: int, seed: int) -> int:
    n     = _noise2(float(wx), float(wz), seed)
    hills = (math.sin(float(wx) * 0.01 + seed * 1e-9) *
             math.cos(float(wz) * 0.01 + seed * 2e-9))
    h = SEA_LEVEL + int(n * 10 + hills * 6)
    return max(BEDROCK_LEVEL + 3, min(CHUNK_HEIGHT - 4, h))


@njit(cache=True)
def _biome_at(wx: int, wz: int, seed: int) -> int:
    """
    Returns biome index:
      0 = plains  1 = forest  2 = desert  3 = snowy taiga  4 = ocean
    Ocean biome is determined by a separate large-scale "continent" noise —
    areas far below 0 become ocean, everything else uses the land biomes.
    """
    # Large-scale continent noise decides land vs ocean
    continent = (_noise2(float(wx) * 0.006, float(wz) * 0.006, seed ^ 0x1234) +
                 _noise2(float(wx) * 0.002, float(wz) * 0.002, seed ^ 0x5678) * 0.5)
    if continent < -0.45:
        return 4   # ocean

    # Land biome
    v = _noise2(float(wx) * 0.03, float(wz) * 0.03, seed)
    if v < -0.3: return 2   # desert
    if v <  0.2: return 0   # plains
    if v <  0.5: return 1   # forest
    return 3                 # snowy taiga


@njit(cache=True)
def _ocean_floor_height(wx: int, wz: int, seed: int) -> int:
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

            # ── Ocean biome ──────────────────────────────────────────────────
            if biome == 4:
                floor_y = _ocean_floor_height(wx, wz, seed)

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

                # Ore pockets in ocean stone
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

            # ── Land biomes ──────────────────────────────────────────────────
            height = _terrain_height(wx, wz, seed)
            surf   = ID_GRASS
            if biome == 2: surf = ID_SAND
            elif biome == 3: surf = ID_SNOW

            for y in range(H):
                if y <= BEDROCK_LEVEL:
                    blk[dx, y, dz] = ID_BEDROCK
                elif y <= BEDROCK_LEVEL + 2:
                    lcg = (lcg * 1664525 + 1013904223) & 0xFFFFFFFF
                    blk[dx, y, dz] = ID_BEDROCK if (lcg & 0xFF) < 180 else ID_STONE
                elif y < height - 3:
                    blk[dx, y, dz] = ID_STONE
                elif y < height:
                    blk[dx, y, dz] = ID_DIRT
                elif y == height:
                    blk[dx, y, dz] = surf
                elif y <= SEA_LEVEL and height < SEA_LEVEL:
                    blk[dx, y, dz] = ID_WATER

            # Cave carving (land only)
            for y in range(BEDROCK_LEVEL + 3, height - 2):
                v = _noise3(float(wx), float(y), float(wz), seed)
                if v > 0.55:
                    blk[dx, y, dz] = 0

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

            # ── Ocean underwater plants ───────────────────────────────────────
            if biome == 4:
                floor_y = _ocean_floor_height(wx, wz, seed)
                above   = floor_y + 1
                if above > SEA_LEVEL or above >= H:
                    continue

                rv = lcg & 0xFF
                if rv < 40:
                    # Kelp column — grows 2-6 blocks upward through water
                    kelp_h = 2 + int(lcg & 0x7) % 5
                    for ky in range(above, min(above + kelp_h, SEA_LEVEL)):
                        if blk[dx, ky, dz] == ID_WATER:
                            blk[dx, ky, dz] = ID_KELP
                elif rv < 70:
                    # Seagrass on the floor
                    if blk[dx, above, dz] == ID_WATER:
                        blk[dx, above, dz] = ID_SEAGRASS
                continue

            # ── Land surface features ────────────────────────────────────────
            h  = _terrain_height(wx, wz, seed)
            ty = h + 1
            if ty + 7 >= H: continue

            # Desert cactus
            if biome == 2:
                if blk[dx, h, dz] != ID_SAND: continue
                if (lcg & 0xFF) >= 30: continue
                cact_h = 2 + (lcg & 0x3)
                for y in range(ty, min(ty + cact_h, H)):
                    blk[dx, y, dz] = ID_CACTUS
                continue

            # Oak tree (plains / forest)
            if biome in (0, 1):
                thresh = 4 if biome == 1 else 2
                if (lcg & 0xFF) >= thresh: continue
                for y in range(ty, ty + 5):
                    for tx in (dx, dx + 1):
                        for tz in (dz, dz + 1):
                            if 0 <= tx < S and 0 <= tz < S:
                                blk[tx, y, tz] = ID_LOG
                for ldy in range(4, 8):
                    for ldx in range(-3, 4):
                        for ldz in range(-3, 4):
                            if abs(ldx) + abs(ldz) <= 4:
                                lx_ = dx + ldx; lz_ = dz + ldz; ly_ = ty + ldy
                                if 0 <= lx_ < S and 0 <= lz_ < S and ly_ < H:
                                    if blk[lx_, ly_, lz_] == 0:
                                        blk[lx_, ly_, lz_] = ID_LEAVES
                continue

            # Spruce tree (snowy taiga)
            if biome == 3:
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
               render, occlude, rot, colors, face_uv,
               cx, cz, greedy, show_all, water_id):
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

                    if show_all == 0:
                        if 0 <= ni < d_i:
                            nidx = [0, 0, 0]
                            nidx[axis] = ni; nidx[a_ax] = a; nidx[b_ax] = b
                            nbt = blk[nidx[0], nidx[1], nidx[2]]
                            if occlude[nbt] == 1:
                                continue
                            if bt == water_id and nbt == water_id:
                                continue
                        else:
                            if axis == 0:
                                nbt = nb_x_neg[a, b] if ndir < 0 else nb_x_pos[a, b]
                            elif axis == 2:
                                nbt = nb_z_neg[a, b] if ndir < 0 else nb_z_pos[a, b]
                            else:
                                nbt = 0

                            if occlude[nbt] == 1:
                                continue
                            if bt == water_id and nbt == water_id:
                                continue

                    mask[a, b] = bt

            done = np.zeros((d_a, d_b), dtype=np.uint8)
            for a in range(d_a):
                for b in range(d_b):
                    bt = mask[a, b]
                    if bt == 0 or done[a, b]:
                        continue
                    bw = 1; ah = 1

                    if greedy != 0:
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