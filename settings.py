"""
settings.py  –  Global constants, block-type tables, texture atlas,
                 per-face UV and colour arrays, matrix helpers.

Import:  from settings import *
"""

import math
import numpy as np
from PIL import Image

from block_textures import BlockType, BLOCK_DATA

# ═══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
CHUNK_SIZE       = 16
CHUNK_HEIGHT     = 64
SEA_LEVEL        = 32
OCEAN_FLOOR      = 24          # deepest ocean floor level
BEDROCK_LEVEL    = 1
VIEW_DISTANCE    = 2
MESH_PER_FRAME   = 2
DAY_TICK_SPEED   = 40.0
GRAVITY          = 20.0
JUMP_VEL         = 8.0
PLAYER_SPEED     = 6.0
PLAYER_HEIGHT    = 1.8
EYE_OFFSET       = 1.6
MOUSE_SENS       = 0.15
FOV              = 90.0
WIN_W, WIN_H     = 1280, 720
ENABLE_WIREFRAME = False
EDGE_COLOR       = (0.08, 0.12, 0.08)
EDGE_WIDTH       = 2.5
TILE_SIZE        = 32
USE_TEXTURES     = True
USE_GREEDY_MESH  = False
SHOW_ALL_FACES   = True
USE_WATER        = True          # ← ocean enabled
ENABLE_MOBS      = False         # temporary: disable mobs

# ── Water physics ─────────────────────────────────────────────────────────────
WATER_GRAVITY_SCALE  = 0.18   # fraction of normal gravity while submerged
WATER_HORIZ_DRAG     = 0.82   # per-frame velocity multiplier
WATER_VERT_DRAG      = 0.78   # tighter vertical drag than horizontal
WATER_SINK_CAP       = -2.5   # terminal velocity while sinking in water
WATER_RISE_CAP       =  2.5   # max upward speed while swimming
WATER_SWIM_FORCE     =  6.0   # upward impulse when pressing SPACE underwater
WATER_SURFACE_JUMP   =  7.5   # jump velocity when exiting water from surface
AIR_MAX              = 10.0   # seconds of air (= 10 "bubbles")
DROWN_DAMAGE_RATE    =  1.0   # HP per second when air is empty
BUOYANCY_STRENGTH    =  9.0   # upward force when fully submerged (blocks/sÂ²)

# Water flow simulation (block spread)
WATER_MAX_OPS        = 80     # max flow steps per tick (higher = faster)
WATER_TICK_INTERVAL  = 0.04   # seconds between flow steps

# ═══════════════════════════════════════════════════════════════════════
#  BLOCK-TYPE REGISTRY  (index == integer stored in chunk arrays)
# ═══════════════════════════════════════════════════════════════════════
AIR = 0
_BT_LIST = [
    BlockType.AIR,
    BlockType.GRASS,
    BlockType.DIRT,
    BlockType.STONE,
    BlockType.COBBLESTONE,
    BlockType.SAND,
    BlockType.GRAVEL,
    BlockType.BEDROCK,
    BlockType.SNOW,
    BlockType.OAK_LOG,
    BlockType.OAK_LEAVES,
    BlockType.SPRUCE_LOG,
    BlockType.SPRUCE_LEAVES,
    BlockType.OAK_PLANKS,
    BlockType.COAL_ORE,
    BlockType.IRON_ORE,
    BlockType.GOLD_ORE,
    BlockType.DIAMOND_ORE,
    BlockType.REDSTONE_ORE,
    BlockType.LAPIS_ORE,
    BlockType.WATER,
    BlockType.TORCH,
    BlockType.BIRCH_LOG,
    BlockType.BIRCH_PLANKS,
    BlockType.SPRUCE_PLANKS,
    BlockType.CACTUS,
    BlockType.CRAFTING_TABLE,
    BlockType.FURNACE,
    # Ocean biome blocks
    BlockType.SAND_OCEAN,
    BlockType.KELP,
    BlockType.SEAGRASS,
    BlockType.PRISMARINE,
]
BT: dict = {bt: i for i, bt in enumerate(_BT_LIST)}
N_BLOCK_TYPES = len(_BT_LIST)

# ── Per-type property arrays (uint8) ────────────────────────────────────────
BT_SOLID  = np.zeros(N_BLOCK_TYPES, dtype=np.uint8)
BT_TRANS  = np.zeros(N_BLOCK_TYPES, dtype=np.uint8)
BT_LIQUID = np.zeros(N_BLOCK_TYPES, dtype=np.uint8)  # only water-type blocks
BT_RENDER = np.zeros(N_BLOCK_TYPES, dtype=np.uint8)
BT_CROSS  = np.zeros(N_BLOCK_TYPES, dtype=np.uint8)
BT_ROT    = np.zeros((N_BLOCK_TYPES, 6), dtype=np.int8)

for _i, _bt in enumerate(_BT_LIST):
    _props = BLOCK_DATA.get(_bt, {})
    BT_SOLID[_i]  = 1 if (_props.get("solid", True) and _bt != BlockType.AIR) else 0
    BT_TRANS[_i]  = 1 if _props.get("transparent", False) else 0
    BT_LIQUID[_i] = 1 if _bt == BlockType.WATER else 0
    BT_RENDER[_i] = 1 if (_bt != BlockType.AIR and (BT_SOLID[_i] or BT_TRANS[_i])) else 0
    BT_CROSS[_i]  = 1 if _props.get("cross", False) else 0
    _faces = _props.get("rotate_side_faces")
    if _faces:
        _dir = int(_props.get("rotate_side_dir", 1))
        for _f in _faces:
            if 0 <= _f < 6:
                BT_ROT[_i, _f] = _dir

# ── Occlusion mask ───────────────────────────────────────────────────────────
# Only truly opaque solid blocks should hide their neighbour's faces.
# Transparent blocks (leaves, water, kelp, seagrass, cactus) must NOT occlude.
BT_OCCLUDE = (BT_SOLID & np.where(BT_TRANS, np.uint8(0), np.uint8(1))).astype(np.uint8)

# Blocks that are hittable by raycast (even if non-solid).
BT_HIT = BT_SOLID.copy()
BT_HIT[BT_CROSS == 1] = 1

# ═══════════════════════════════════════════════════════════════════════
#  TEXTURE ATLAS
# ═══════════════════════════════════════════════════════════════════════

def _load_image_rgba(path: str):
    try:
        img = Image.open(path).convert("RGBA")
        w, h = img.size
        # If this is an animated strip (e.g., water/lava), keep only the first frame.
        if h != w and h % w == 0:
            img = img.crop((0, 0, w, w))
            w = h = w
        if (w, h) != (TILE_SIZE, TILE_SIZE):
            img = img.resize((TILE_SIZE, TILE_SIZE), resample=Image.NEAREST)
        return img
    except Exception:
        return None


def build_texture_atlas():
    tex_paths = []
    if USE_TEXTURES:
        for props in BLOCK_DATA.values():
            for k in ("texture", "texture_top", "texture_side", "texture_bottom"):
                p = props.get(k)
                if p and p not in tex_paths:
                    tex_paths.append(p)

    virtual_tiles = {}
    if USE_TEXTURES:
        for props in BLOCK_DATA.values():
            side = props.get("texture_side")
            if not side:
                continue
            under_tex = props.get("side_underlay")
            under_col = props.get("side_underlay_color")
            if not under_tex and not under_col:
                continue
            overlay = _load_image_rgba(side)
            if overlay is None:
                continue
            if under_tex:
                base = _load_image_rgba(under_tex) or Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (255,255,255,255))
            else:
                try:    r, g, b = int(under_col[0]), int(under_col[1]), int(under_col[2])
                except: r, g, b = 255, 255, 255
                base = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (r, g, b, 255))
            comp = base.copy()
            comp.alpha_composite(overlay)
            virtual_tiles[f"__overlay__:{side}"] = comp

    color_tiles = {}
    for bt, props in BLOCK_DATA.items():
        has_tex = (props.get("texture") or props.get("texture_top") or
                   props.get("texture_side") or props.get("texture_bottom"))
        if has_tex:
            continue
        c = props.get("color")
        try:    r, g, b = int(c[0]), int(c[1]), int(c[2])
        except: r, g, b = 255, 255, 255
        color_tiles[f"__color__:{bt}"] = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (r, g, b, 255))

    white_key = "__white__"
    all_keys  = [white_key] + tex_paths + list(virtual_tiles.keys()) + list(color_tiles.keys())
    n    = len(all_keys)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    atlas   = Image.new("RGBA", (cols * TILE_SIZE, rows * TILE_SIZE), (255,255,255,255))
    uv_map  = {}

    for i, key in enumerate(all_keys):
        cx2 = i % cols;  cy2 = i // cols
        x   = cx2 * TILE_SIZE;  y = cy2 * TILE_SIZE
        if key == white_key:
            tile = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (255,255,255,255))
        elif key in virtual_tiles:
            tile = virtual_tiles[key]
        elif key in color_tiles:
            tile = color_tiles[key]
        else:
            tile = _load_image_rgba(key)
            if tile is None:
                tile = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (255,255,255,255))
        atlas.paste(tile, (x, y))
        aw, ah = atlas.size
        uv_map[key] = (x / aw, y / ah, (x + TILE_SIZE) / aw, (y + TILE_SIZE) / ah)

    return atlas, uv_map, white_key


_atlas_img, _uv_map, _white_key = build_texture_atlas()

# ═══════════════════════════════════════════════════════════════════════
#  PER-FACE UV  (N_BLOCK_TYPES × 6 × 4)
# ═══════════════════════════════════════════════════════════════════════
FACE_UV = np.zeros((N_BLOCK_TYPES, 6, 4), dtype=np.float32)

for _i, _bt in enumerate(_BT_LIST):
    _props = BLOCK_DATA.get(_bt, {})
    if not USE_TEXTURES:
        _top = _side = _bot = f"__color__:{_bt}"
    else:
        _top  = _props.get("texture_top")
        _side = _props.get("texture_side")
        _bot  = _props.get("texture_bottom")
        _allf = _props.get("texture")
        if _side and (_props.get("side_underlay") or _props.get("side_underlay_color")):
            _ovk = f"__overlay__:{_side}"
            if _ovk in _uv_map:
                _side = _ovk
        if _top is None and _side is None and _bot is None and _allf is None:
            _top = _side = _bot = f"__color__:{_bt}"
        if _allf:
            _top = _side = _bot = _allf
        if _top  and not _side: _side = _top
        if _top  and not _bot:  _bot  = _top
        _top  = _top  or _white_key
        _side = _side or _white_key
        _bot  = _bot  or _white_key

    FACE_UV[_i, 0] = _uv_map.get(_top,  _uv_map[_white_key])
    FACE_UV[_i, 1] = _uv_map.get(_bot,  _uv_map[_white_key])
    FACE_UV[_i, 2] = _uv_map.get(_side, _uv_map[_white_key])
    FACE_UV[_i, 3] = _uv_map.get(_side, _uv_map[_white_key])
    FACE_UV[_i, 4] = _uv_map.get(_side, _uv_map[_white_key])
    FACE_UV[_i, 5] = _uv_map.get(_side, _uv_map[_white_key])

# ═══════════════════════════════════════════════════════════════════════
#  PER-FACE COLOURS  (N_BLOCK_TYPES × 6 × 4)
# ═══════════════════════════════════════════════════════════════════════
_FACE_SHADE = [1.0, 0.55, 0.80, 0.80, 0.90, 0.90]


def _make_face_colors():
    out = np.ones((N_BLOCK_TYPES, 6, 4), dtype=np.float32)
    for i, bt in enumerate(_BT_LIST):
        props      = BLOCK_DATA.get(bt, {"color": (1.0, 1.0, 1.0, 1.0)})
        has_tex    = (props.get("texture") or props.get("texture_top") or
                      props.get("texture_side") or props.get("texture_bottom"))
        is_trans   = bool(props.get("transparent", False))
        tint       = bool(props.get("tint", False))
        tint_faces = props.get("tint_faces")

        if has_tex and not is_trans and not tint:
            c = (1.0, 1.0, 1.0, 1.0)
        else:
            c = props.get("color", (1.0, 1.0, 1.0, 1.0))

        try:    r, g, b = float(c[0]), float(c[1]), float(c[2])
        except: r, g, b = 1.0, 1.0, 1.0
        if max(r, g, b) > 1.0:
            r /= 255.0; g /= 255.0; b /= 255.0

        if is_trans or tint:
            a = float(c[3]) / 255.0 if len(c) >= 4 else (0.5 if is_trans else 1.0)
        else:
            a = 1.0

        # Cutout blocks (leaves, cactus) have 3-component colors — force a=1.0
        # so texture alpha channel drives the cutout. Liquid blocks (water) have
        # a 4-component color with explicit alpha — respect it for transparency.
        if is_trans and has_tex and len(c) < 4:
            a = 1.0

        for f, shade in enumerate(_FACE_SHADE):
            if tint and tint_faces is not None and f not in tint_faces:
                out[i, f] = [1.0 * shade, 1.0 * shade, 1.0 * shade, 1.0]
            else:
                out[i, f] = [r * shade, g * shade, b * shade, a]
    return out


FACE_COLORS = _make_face_colors()

# ═══════════════════════════════════════════════════════════════════════
#  MATRIX HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _persp(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    return np.array([
        [f / aspect, 0,  0,                             0                        ],
        [0,          f,  0,                             0                        ],
        [0,          0, -(far + near) / (far - near),  -2*far*near / (far - near)],
        [0,          0, -1,                             0                        ],
    ], dtype=np.float32)


def _lookat(eye: np.ndarray, forward: np.ndarray) -> np.ndarray:
    right = np.cross(forward, np.array([0, 1, 0], np.float32))
    rl    = np.linalg.norm(right)
    right = right / rl if rl > 1e-8 else np.array([1, 0, 0], np.float32)
    up    = np.cross(right, forward)
    m     = np.eye(4, dtype=np.float32)
    m[0, :3] =  right;   m[0, 3] = -np.dot(right,   eye)
    m[1, :3] =  up;      m[1, 3] = -np.dot(up,       eye)
    m[2, :3] = -forward; m[2, 3] =  np.dot(forward,  eye)
    return m
