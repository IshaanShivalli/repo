"""
BLOCK TEXTURES MODULE  –  v3  (ocean biome edition)
Defines all block types and their visual / gameplay properties.
"""

TEX_BASE = "textures/assets/minecraft/textures/block"

class _Color:
    @staticmethod
    def rgb(r, g, b):       return (r, g, b)
    @staticmethod
    def rgba(r, g, b, a):   return (r, g, b, a)
    pink  = (255, 182, 193)
    white = (255, 255, 255)
    gray  = (128, 128, 128)
    green = (0,   128,   0)

color = _Color()

class BlockType:
    AIR            = "air"
    GRASS          = "grass"
    DIRT           = "dirt"
    STONE          = "stone"
    SAND           = "sand"
    GRAVEL         = "gravel"
    BEDROCK        = "bedrock"
    SNOW           = "snow"
    SNOW_DIRT      = "snow_dirt"
    OAK_LOG        = "oak_log"
    BIRCH_LOG      = "birch_log"
    SPRUCE_LOG     = "spruce_log"
    OAK_LEAVES     = "oak_leaves"
    BIRCH_LEAVES   = "birch_leaves"
    SPRUCE_LEAVES  = "spruce_leaves"
    OAK_PLANKS     = "oak_planks"
    BIRCH_PLANKS   = "birch_planks"
    SPRUCE_PLANKS  = "spruce_planks"
    COBBLESTONE    = "cobblestone"
    STONE_BRICKS   = "stone_bricks"
    MOSSY_COBBLE   = "mossy_cobblestone"
    COAL_ORE       = "coal_ore"
    IRON_ORE       = "iron_ore"
    GOLD_ORE       = "gold_ore"
    DIAMOND_ORE    = "diamond_ore"
    REDSTONE_ORE   = "redstone_ore"
    LAPIS_ORE      = "lapis_ore"
    WATER          = "water"
    GLASS          = "glass"
    CRAFTING_TABLE = "crafting_table"
    FURNACE        = "furnace"
    CHEST          = "chest"
    BOOKSHELF      = "bookshelf"
    CACTUS         = "cactus"
    RED_SAND       = "red_sand"
    # Ocean biome
    SAND_OCEAN     = "sand_ocean"
    KELP           = "kelp"
    SEAGRASS       = "seagrass"
    PRISMARINE     = "prismarine"


BLOCK_DATA = {

    BlockType.AIR: {
        "color": color.rgba(0, 0, 0, 0),
        "solid": False,
        "drops": None,
    },

    BlockType.GRASS: {
        "color":             color.rgb(110, 170, 120),
        "texture_top":       f"{TEX_BASE}/grass_block_top.png",
        "texture_side":      f"{TEX_BASE}/grass_block_side.png",
        "texture_bottom":    f"{TEX_BASE}/dirt.png",
        "tint":              True,
        "tint_faces":        [0],
        "rotate_side_faces": [2, 3, 4, 5],
        "rotate_side_dir":   1,
        "solid": True, "drops": BlockType.DIRT,
    },
    BlockType.DIRT: {
        "color":   color.rgb(120, 95, 65),
        "texture": f"{TEX_BASE}/dirt.png",
        "solid":   True, "drops": BlockType.DIRT,
    },
    BlockType.STONE: {
        "color":   color.rgb(120, 125, 130),
        "texture": f"{TEX_BASE}/stone.png",
        "solid":   True, "drops": BlockType.COBBLESTONE,
    },
    BlockType.COBBLESTONE: {
        "color":   color.rgb(105, 105, 105),
        "texture": f"{TEX_BASE}/cobblestone.png",
        "solid":   True, "drops": BlockType.COBBLESTONE,
    },
    BlockType.STONE_BRICKS: {
        "color":   color.rgb(110, 110, 110),
        "texture": f"{TEX_BASE}/stone_bricks.png",
        "solid":   True, "drops": BlockType.STONE_BRICKS,
    },
    BlockType.MOSSY_COBBLE: {
        "color":   color.rgb(80, 110, 70),
        "texture": f"{TEX_BASE}/mossy_cobblestone.png",
        "solid":   True, "drops": BlockType.MOSSY_COBBLE,
    },
    BlockType.SAND: {
        "color":   color.rgb(210, 200, 170),
        "texture": f"{TEX_BASE}/sand.png",
        "solid":   True, "drops": BlockType.SAND,
    },
    BlockType.RED_SAND: {
        "color":   color.rgb(200, 130, 60),
        "texture": f"{TEX_BASE}/red_sand.png",
        "solid":   True, "drops": BlockType.RED_SAND,
    },
    BlockType.GRAVEL: {
        "color":   color.rgb(150, 140, 130),
        "texture": f"{TEX_BASE}/gravel.png",
        "solid":   True, "drops": BlockType.GRAVEL,
    },
    BlockType.BEDROCK: {
        "color":       color.rgb(40, 40, 40),
        "texture":     f"{TEX_BASE}/bedrock.png",
        "solid":       True, "unbreakable": True, "drops": None,
    },
    BlockType.SNOW: {
        "color":          color.rgb(240, 248, 255),
        "texture_top":    f"{TEX_BASE}/snow.png",
        "texture_side":   f"{TEX_BASE}/snow.png",
        "texture_bottom": f"{TEX_BASE}/snow.png",
        "solid": True, "drops": BlockType.SNOW,
    },
    BlockType.SNOW_DIRT: {
        "color":   color.rgb(180, 160, 130),
        "texture": f"{TEX_BASE}/dirt.png",
        "solid":   True, "drops": BlockType.DIRT,
    },

    BlockType.OAK_LOG: {
        "color":          color.rgb(120, 95, 70),
        "texture_top":    f"{TEX_BASE}/oak_log_top.png",
        "texture_side":   f"{TEX_BASE}/oak_log.png",
        "texture_bottom": f"{TEX_BASE}/oak_log_top.png",
        "solid": True, "drops": BlockType.OAK_LOG,
    },
    BlockType.BIRCH_LOG: {
        "color":          color.rgb(220, 215, 195),
        "texture_top":    f"{TEX_BASE}/birch_log_top.png",
        "texture_side":   f"{TEX_BASE}/birch_log.png",
        "texture_bottom": f"{TEX_BASE}/birch_log_top.png",
        "solid": True, "drops": BlockType.BIRCH_LOG,
    },
    BlockType.SPRUCE_LOG: {
        "color":          color.rgb(80, 55, 25),
        "texture_top":    f"{TEX_BASE}/spruce_log_top.png",
        "texture_side":   f"{TEX_BASE}/spruce_log.png",
        "texture_bottom": f"{TEX_BASE}/spruce_log_top.png",
        "solid": True, "drops": BlockType.SPRUCE_LOG,
    },

    BlockType.OAK_LEAVES: {
        "color":       color.rgb(95, 145, 110),
        "texture":     f"{TEX_BASE}/oak_leaves.png",
        "solid":       True, "transparent": True, "drops": None,
    },
    BlockType.BIRCH_LEAVES: {
        "color":       color.rgb(120, 175, 80),
        "texture":     f"{TEX_BASE}/birch_leaves.png",
        "solid":       True, "transparent": True, "drops": None,
    },
    BlockType.SPRUCE_LEAVES: {
        "color":       color.rgb(80, 120, 90),
        "texture":     f"{TEX_BASE}/spruce_leaves.png",
        "solid":       True, "transparent": True, "drops": None,
    },

    BlockType.OAK_PLANKS: {
        "color":   color.rgb(180, 145, 80),
        "texture": f"{TEX_BASE}/oak_planks.png",
        "solid":   True, "drops": BlockType.OAK_PLANKS,
    },
    BlockType.BIRCH_PLANKS: {
        "color":   color.rgb(220, 200, 155),
        "texture": f"{TEX_BASE}/birch_planks.png",
        "solid":   True, "drops": BlockType.BIRCH_PLANKS,
    },
    BlockType.SPRUCE_PLANKS: {
        "color":   color.rgb(105, 75, 40),
        "texture": f"{TEX_BASE}/spruce_planks.png",
        "solid":   True, "drops": BlockType.SPRUCE_PLANKS,
    },

    BlockType.COAL_ORE:     {"color": color.rgb(70,70,70),   "texture": f"{TEX_BASE}/coal_ore.png",     "solid": True, "drops": "coal"},
    BlockType.IRON_ORE:     {"color": color.rgb(140,105,85), "texture": f"{TEX_BASE}/iron_ore.png",     "solid": True, "drops": "iron_ore"},
    BlockType.GOLD_ORE:     {"color": color.rgb(200,175,60), "texture": f"{TEX_BASE}/gold_ore.png",     "solid": True, "drops": "gold_ore"},
    BlockType.DIAMOND_ORE:  {"color": color.rgb(75,205,210), "texture": f"{TEX_BASE}/diamond_ore.png",  "solid": True, "drops": "diamond"},
    BlockType.REDSTONE_ORE: {"color": color.rgb(180,50,50),  "texture": f"{TEX_BASE}/redstone_ore.png", "solid": True, "drops": "redstone"},
    BlockType.LAPIS_ORE:    {"color": color.rgb(60,80,180),  "texture": f"{TEX_BASE}/lapis_ore.png",    "solid": True, "drops": "lapis"},

    BlockType.WATER: {
        "color":       color.rgba(55, 100, 200, 160),
        "texture":     f"{TEX_BASE}/water_still.png",
        "solid":       False, "transparent": True, "drops": None,
    },
    BlockType.GLASS: {
        "color":       color.rgba(180, 220, 240, 120),
        "texture":     f"{TEX_BASE}/glass.png",
        "solid":       True, "transparent": True, "drops": None,
    },

    BlockType.CRAFTING_TABLE: {
        "color":          color.rgb(170, 110, 50),
        "texture_top":    f"{TEX_BASE}/crafting_table_top.png",
        "texture_side":   f"{TEX_BASE}/crafting_table_side.png",
        "texture_bottom": f"{TEX_BASE}/oak_planks.png",
        "solid": True, "drops": BlockType.CRAFTING_TABLE,
    },
    BlockType.FURNACE: {
        "color":          color.rgb(110, 110, 110),
        "texture_top":    f"{TEX_BASE}/furnace_top.png",
        "texture_side":   f"{TEX_BASE}/furnace_side.png",
        "texture_bottom": f"{TEX_BASE}/furnace_top.png",
        "solid": True, "drops": BlockType.FURNACE,
    },
    BlockType.CHEST: {
        "color":   color.rgb(160, 110, 50),
        "texture": f"{TEX_BASE}/oak_planks.png",
        "solid":   True, "drops": BlockType.CHEST,
    },
    BlockType.BOOKSHELF: {
        "color":   color.rgb(170, 130, 70),
        "texture": f"{TEX_BASE}/bookshelf.png",
        "solid":   True, "drops": BlockType.OAK_PLANKS,
    },

    BlockType.CACTUS: {
        "color":          color.rgb(50, 140, 50),
        "texture_top":    f"{TEX_BASE}/cactus_top.png",
        "texture_side":   f"{TEX_BASE}/cactus_side.png",
        "texture_bottom": f"{TEX_BASE}/cactus_bottom.png",
        "solid":       True, "transparent": True, "drops": BlockType.CACTUS,
    },

    # ── Ocean ─────────────────────────────────────────────────────────────────
    BlockType.SAND_OCEAN: {
        "color":   color.rgb(175, 168, 135),
        "texture": f"{TEX_BASE}/sand.png",
        "solid":   True, "drops": BlockType.SAND,
    },
    BlockType.KELP: {
        "color":       color.rgba(50, 145, 65, 220),
        "texture":     f"{TEX_BASE}/kelp.png",
        "solid":       False, "transparent": True, "drops": None,
    },
    BlockType.SEAGRASS: {
        "color":       color.rgba(60, 160, 85, 200),
        "texture":     f"{TEX_BASE}/seagrass.png",
        "solid":       False, "transparent": True, "drops": None,
    },
    BlockType.PRISMARINE: {
        "color":   color.rgb(100, 170, 160),
        "texture": f"{TEX_BASE}/prismarine.png",
        "solid":   True, "drops": BlockType.PRISMARINE,
    },
}