"""
inventory_system.py  –  Items, recipes (all wood & plank tools/blocks),
                         food data, furnace recipes, fuel values.

Recipes use a flat ingredient-count dict so the crafting UI can match them
without worrying about exact grid arrangement.
"""

from block_textures import BlockType


# ═══════════════════════════════════════════════════════════════════════
#  ITEM TYPES  (tools, food, misc — not block-placed items)
# ═══════════════════════════════════════════════════════════════════════
class ItemType:
    # ── Wooden tools ─────────────────────────────────────────────────────────
    WOODEN_PICKAXE  = "wooden_pickaxe"
    WOODEN_AXE      = "wooden_axe"
    WOODEN_SWORD    = "wooden_sword"
    WOODEN_SHOVEL   = "wooden_shovel"
    WOODEN_HOE      = "wooden_hoe"

    # ── Stone tools ───────────────────────────────────────────────────────────
    STONE_PICKAXE   = "stone_pickaxe"
    STONE_AXE       = "stone_axe"
    STONE_SWORD     = "stone_sword"
    STONE_SHOVEL    = "stone_shovel"
    STONE_HOE       = "stone_hoe"

    # ── Iron tools ────────────────────────────────────────────────────────────
    IRON_PICKAXE    = "iron_pickaxe"
    IRON_AXE        = "iron_axe"
    IRON_SWORD      = "iron_sword"
    IRON_SHOVEL     = "iron_shovel"
    IRON_HOE        = "iron_hoe"

    # ── Wood / Plank sub-products ─────────────────────────────────────────────
    STICK           = "stick"
    TORCH           = "torch"
    BOWL            = "bowl"
    BOAT            = "oak_boat"
    SIGN            = "oak_sign"
    LADDER          = "ladder"
    CHEST_ITEM      = "chest"       # same key used for the block too
    CRAFTING_TABLE  = "crafting_table"
    BOOKSHELF       = "bookshelf"
    WOODEN_DOOR     = "oak_door"
    TRAPDOOR        = "oak_trapdoor"
    FENCE           = "oak_fence"
    FENCE_GATE      = "oak_fence_gate"
    PRESSURE_PLATE  = "oak_pressure_plate"
    BUTTON          = "oak_button"
    SLAB            = "oak_slab"
        # Add to ItemType class:
    OAK_SAPLING = "oak_sapling"
    BIRCH_SAPLING = "birch_sapling"
    SPRUCE_SAPLING = "spruce_sapling"

    # Add to recipes (optional - for crafting):
    # Sapling can be used as fuel or compost


    # ── Food ─────────────────────────────────────────────────────────────────
    APPLE           = "apple"
    BREAD           = "bread"
    COOKIE          = "cookie"
    COOKED_PORK     = "cooked_porkchop"
    COOKED_BEEF     = "cooked_beef"
    COOKED_CHICKEN  = "cooked_chicken"
    COOKED_MUTTON   = "cooked_mutton"

    # ── Misc ─────────────────────────────────────────────────────────────────
    BOOK            = "book"
    PAPER           = "paper"
    COAL            = "coal"
    CHARCOAL        = "charcoal"
    IRON_INGOT      = "iron_ingot"
    GOLD_INGOT      = "gold_ingot"
    DIAMOND         = "diamond"
    BOW             = "bow"
    ARROW           = "arrow"
    STRING          = "string"
    FEATHER         = "feather"
    FLINT           = "flint"
    FLINT_AND_STEEL = "flint_and_steel"
    LEATHER         = "leather"
    BONE            = "bone"
    BONE_MEAL       = "bone_meal"
    BLAZE_ROD       = "blaze_rod"
    GLASS_BOTTLE    = "glass_bottle"
    FISHING_ROD     = "fishing_rod"


# ═══════════════════════════════════════════════════════════════════════
#  FOOD DATA  {item_id: {"hunger": int, "saturation": float}}
# ═══════════════════════════════════════════════════════════════════════
FOOD_DATA = {
    ItemType.APPLE:          {"hunger": 4,  "saturation": 2.4},
    ItemType.BREAD:          {"hunger": 5,  "saturation": 6.0},
    ItemType.COOKIE:         {"hunger": 2,  "saturation": 0.4},
    ItemType.COOKED_PORK:    {"hunger": 8,  "saturation": 12.8},
    ItemType.COOKED_BEEF:    {"hunger": 8,  "saturation": 12.8},
    ItemType.COOKED_CHICKEN: {"hunger": 6,  "saturation": 7.2},
    ItemType.COOKED_MUTTON:  {"hunger": 6,  "saturation": 9.6},
    "raw_porkchop":          {"hunger": 3,  "saturation": 1.8},
    "raw_beef":              {"hunger": 3,  "saturation": 1.8},
    "raw_chicken":           {"hunger": 2,  "saturation": 1.2},
    "raw_mutton":            {"hunger": 2,  "saturation": 1.2},
    "rotten_flesh":          {"hunger": 4,  "saturation": 0.8},
}

# ═══════════════════════════════════════════════════════════════════════
#  FURNACE RECIPES  {input_item: output_item}
# ═══════════════════════════════════════════════════════════════════════
FURNACE_RECIPES = {
    BlockType.STONE:         BlockType.STONE,
    BlockType.SAND:          BlockType.GLASS,
    "iron_ore":              ItemType.IRON_INGOT,
    "gold_ore":              ItemType.GOLD_INGOT,
    "raw_porkchop":          ItemType.COOKED_PORK,
    "raw_beef":              ItemType.COOKED_BEEF,
    "raw_chicken":           ItemType.COOKED_CHICKEN,
    "raw_mutton":            ItemType.COOKED_MUTTON,
    BlockType.OAK_LOG:       ItemType.CHARCOAL,
    BlockType.BIRCH_LOG:     ItemType.CHARCOAL,
    BlockType.SPRUCE_LOG:    ItemType.CHARCOAL,
    BlockType.COBBLESTONE:   BlockType.STONE,
}

# ═══════════════════════════════════════════════════════════════════════
#  FUEL DATA  {item_id: burn_seconds}
# ═══════════════════════════════════════════════════════════════════════
FUEL_DATA = {
    # Logs burn 15 s each
    BlockType.OAK_LOG:      15.0,
    BlockType.BIRCH_LOG:    15.0,
    BlockType.SPRUCE_LOG:   15.0,
    # Planks burn 7.5 s each
    BlockType.OAK_PLANKS:   7.5,
    BlockType.BIRCH_PLANKS: 7.5,
    BlockType.SPRUCE_PLANKS:7.5,
    # Coal / charcoal burn 80 s each
    ItemType.COAL:          80.0,
    ItemType.CHARCOAL:      80.0,
    # Sticks burn 2.5 s each
    ItemType.STICK:          2.5,
    # Wooden tools burn 10 s each
    ItemType.WOODEN_PICKAXE:10.0,
    ItemType.WOODEN_AXE:    10.0,
    ItemType.WOODEN_SWORD:  10.0,
    ItemType.WOODEN_SHOVEL: 10.0,
    ItemType.WOODEN_HOE:    10.0,
    # Blaze rod
    ItemType.BLAZE_ROD:    120.0,
}

# ═══════════════════════════════════════════════════════════════════════
# -----------------------------------------------------------------------------
# CRAFTING RECIPES (Shaped, Minecraft-style)
# -----------------------------------------------------------------------------

PLANKS = (BlockType.OAK_PLANKS, BlockType.BIRCH_PLANKS, BlockType.SPRUCE_PLANKS)
LOGS   = (BlockType.OAK_LOG, BlockType.BIRCH_LOG, BlockType.SPRUCE_LOG)

SHAPED_RECIPES = [
    # Basic conversion
    {"out": BlockType.OAK_PLANKS,    "count": 4, "pattern": ["L"], "key": {"L": BlockType.OAK_LOG}},
    {"out": BlockType.BIRCH_PLANKS,  "count": 4, "pattern": ["L"], "key": {"L": BlockType.BIRCH_LOG}},
    {"out": BlockType.SPRUCE_PLANKS, "count": 4, "pattern": ["L"], "key": {"L": BlockType.SPRUCE_LOG}},
    {"out": ItemType.STICK,         "count": 4, "pattern": ["P", "P"], "key": {"P": PLANKS}},

    # Wooden tools
    {"out": ItemType.WOODEN_PICKAXE, "count": 1, "pattern": ["PPP", " S ", " S "], "key": {"P": PLANKS, "S": ItemType.STICK}},
    {"out": ItemType.WOODEN_AXE,     "count": 1, "pattern": ["PP ", "PS ", " S "], "key": {"P": PLANKS, "S": ItemType.STICK}},
    {"out": ItemType.WOODEN_AXE,     "count": 1, "pattern": [" PP", " SP", " S "], "key": {"P": PLANKS, "S": ItemType.STICK}},
    {"out": ItemType.WOODEN_SWORD,   "count": 1, "pattern": [" P ", " P ", " S "], "key": {"P": PLANKS, "S": ItemType.STICK}},
    {"out": ItemType.WOODEN_SHOVEL,  "count": 1, "pattern": [" P ", " S ", " S "], "key": {"P": PLANKS, "S": ItemType.STICK}},
    {"out": ItemType.WOODEN_HOE,     "count": 1, "pattern": ["PP ", " S ", " S "], "key": {"P": PLANKS, "S": ItemType.STICK}},
    {"out": ItemType.WOODEN_HOE,     "count": 1, "pattern": [" PP", " S ", " S "], "key": {"P": PLANKS, "S": ItemType.STICK}},

    # Stone tools
    {"out": ItemType.STONE_PICKAXE,  "count": 1, "pattern": ["CCC", " S ", " S "], "key": {"C": BlockType.COBBLESTONE, "S": ItemType.STICK}},
    {"out": ItemType.STONE_AXE,      "count": 1, "pattern": ["CC ", "CS ", " S "], "key": {"C": BlockType.COBBLESTONE, "S": ItemType.STICK}},
    {"out": ItemType.STONE_AXE,      "count": 1, "pattern": [" CC", " SC", " S "], "key": {"C": BlockType.COBBLESTONE, "S": ItemType.STICK}},
    {"out": ItemType.STONE_SWORD,    "count": 1, "pattern": [" C ", " C ", " S "], "key": {"C": BlockType.COBBLESTONE, "S": ItemType.STICK}},
    {"out": ItemType.STONE_SHOVEL,   "count": 1, "pattern": [" C ", " S ", " S "], "key": {"C": BlockType.COBBLESTONE, "S": ItemType.STICK}},
    {"out": ItemType.STONE_HOE,      "count": 1, "pattern": ["CC ", " S ", " S "], "key": {"C": BlockType.COBBLESTONE, "S": ItemType.STICK}},
    {"out": ItemType.STONE_HOE,      "count": 1, "pattern": [" CC", " S ", " S "], "key": {"C": BlockType.COBBLESTONE, "S": ItemType.STICK}},

    # Iron tools
    {"out": ItemType.IRON_PICKAXE,   "count": 1, "pattern": ["III", " S ", " S "], "key": {"I": ItemType.IRON_INGOT, "S": ItemType.STICK}},
    {"out": ItemType.IRON_AXE,       "count": 1, "pattern": ["II ", "IS ", " S "], "key": {"I": ItemType.IRON_INGOT, "S": ItemType.STICK}},
    {"out": ItemType.IRON_AXE,       "count": 1, "pattern": [" II", " SI", " S "], "key": {"I": ItemType.IRON_INGOT, "S": ItemType.STICK}},
    {"out": ItemType.IRON_SWORD,     "count": 1, "pattern": [" I ", " I ", " S "], "key": {"I": ItemType.IRON_INGOT, "S": ItemType.STICK}},
    {"out": ItemType.IRON_SHOVEL,    "count": 1, "pattern": [" I ", " S ", " S "], "key": {"I": ItemType.IRON_INGOT, "S": ItemType.STICK}},
    {"out": ItemType.IRON_HOE,       "count": 1, "pattern": ["II ", " S ", " S "], "key": {"I": ItemType.IRON_INGOT, "S": ItemType.STICK}},
    {"out": ItemType.IRON_HOE,       "count": 1, "pattern": [" II", " S ", " S "], "key": {"I": ItemType.IRON_INGOT, "S": ItemType.STICK}},

    # Torches
    {"out": BlockType.TORCH,         "count": 4, "pattern": ["C", "S"], "key": {"C": (ItemType.COAL, ItemType.CHARCOAL), "S": ItemType.STICK}},

    # Wooden building blocks
    {"out": BlockType.CRAFTING_TABLE,"count": 1, "pattern": ["PP", "PP"], "key": {"P": PLANKS}},
    {"out": BlockType.CHEST,         "count": 1, "pattern": ["PPP", "P P", "PPP"], "key": {"P": PLANKS}},
    {"out": BlockType.BOOKSHELF,     "count": 1, "pattern": ["PPP", "BBB", "PPP"], "key": {"P": PLANKS, "B": ItemType.BOOK}},

    {"out": ItemType.WOODEN_DOOR,    "count": 3, "pattern": ["PP", "PP", "PP"], "key": {"P": PLANKS}},
    {"out": ItemType.TRAPDOOR,       "count": 2, "pattern": ["PPP", "PPP"], "key": {"P": PLANKS}},
    {"out": ItemType.FENCE,          "count": 3, "pattern": ["PSP", "PSP"], "key": {"P": PLANKS, "S": ItemType.STICK}},
    {"out": ItemType.FENCE_GATE,     "count": 1, "pattern": ["SPS", "SPS"], "key": {"P": PLANKS, "S": ItemType.STICK}},
    {"out": ItemType.PRESSURE_PLATE, "count": 1, "pattern": ["PP"], "key": {"P": PLANKS}},
    {"out": ItemType.BUTTON,         "count": 1, "pattern": ["P"], "key": {"P": PLANKS}},
    {"out": ItemType.SLAB,           "count": 6, "pattern": ["PPP"], "key": {"P": PLANKS}},
    {"out": ItemType.LADDER,         "count": 3, "pattern": ["S S", "SSS", "S S"], "key": {"S": ItemType.STICK}},
    {"out": ItemType.BOWL,           "count": 4, "pattern": ["P P", " P "], "key": {"P": PLANKS}},
    {"out": ItemType.BOAT,           "count": 1, "pattern": ["P P", "PPP"], "key": {"P": PLANKS}},
    {"out": ItemType.SIGN,           "count": 3, "pattern": ["PPP", "PPP", " S "], "key": {"P": PLANKS, "S": ItemType.STICK}},

    # Glass bottle
    {"out": ItemType.GLASS_BOTTLE,   "count": 3, "pattern": ["G G", " G "], "key": {"G": BlockType.GLASS}},

    # Flint and steel
    {"out": ItemType.FLINT_AND_STEEL,"count": 1, "pattern": ["I ", " F"], "key": {"I": ItemType.IRON_INGOT, "F": ItemType.FLINT}},

    # Cookie / Bread / Paper
    {"out": ItemType.COOKIE,         "count": 8, "pattern": ["W C W"], "key": {"W": "wheat", "C": "cocoa_beans"}},
    {"out": ItemType.BREAD,          "count": 1, "pattern": ["WWW"], "key": {"W": "wheat"}},
    {"out": ItemType.PAPER,          "count": 3, "pattern": ["CCC"], "key": {"C": "sugar_cane"}},

    # Arrow
    {"out": ItemType.ARROW,          "count": 4, "pattern": ["F", "S", "A"], "key": {"F": ItemType.FLINT, "S": ItemType.STICK, "A": ItemType.FEATHER}},
]


#  LEGACY CLASS SHIM  (main.py imports PlayerInventorySystem from here)
# ═══════════════════════════════════════════════════════════════════════
class PlayerInventorySystem:
    pass
