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
#  CRAFTING RECIPES
#  Format: {output_item: [(ingredient, count), ...]}
#  The UI matches by summing ingredient counts — no grid layout needed.
# ═══════════════════════════════════════════════════════════════════════

_P   = BlockType.OAK_PLANKS    # shorthand for oak planks
_S   = ItemType.STICK           # shorthand for stick
_LOG = BlockType.OAK_LOG        # shorthand for oak log
_COB = BlockType.COBBLESTONE
_STR = ItemType.STRING

RECIPES = {

    # ── Basic conversion ──────────────────────────────────────────────────────
    # 1 log → 4 planks
    BlockType.OAK_PLANKS:    [(_LOG, 1)],
    BlockType.BIRCH_PLANKS:  [(BlockType.BIRCH_LOG,   1)],
    BlockType.SPRUCE_PLANKS: [(BlockType.SPRUCE_LOG,  1)],
    # 2 planks (stacked vertically) → 4 sticks
    ItemType.STICK:          [(_P, 2)],

    # ── Basic tools (wooden) ─────────────────────────────────────────────────
    # Pickaxe: 3 planks + 2 sticks
    ItemType.WOODEN_PICKAXE:  [(_P, 3), (_S, 2)],
    # Axe: 3 planks + 2 sticks
    ItemType.WOODEN_AXE:      [(_P, 3), (_S, 2)],
    # Sword: 2 planks + 1 stick
    ItemType.WOODEN_SWORD:    [(_P, 2), (_S, 1)],
    # Shovel: 1 plank + 2 sticks
    ItemType.WOODEN_SHOVEL:   [(_P, 1), (_S, 2)],
    # Hoe: 2 planks + 2 sticks
    ItemType.WOODEN_HOE:      [(_P, 2), (_S, 2)],

    # ── Stone tools ───────────────────────────────────────────────────────────
    ItemType.STONE_PICKAXE:   [(_COB, 3), (_S, 2)],
    ItemType.STONE_AXE:       [(_COB, 3), (_S, 2)],
    ItemType.STONE_SWORD:     [(_COB, 2), (_S, 1)],
    ItemType.STONE_SHOVEL:    [(_COB, 1), (_S, 2)],
    ItemType.STONE_HOE:       [(_COB, 2), (_S, 2)],

    # ── Iron tools ────────────────────────────────────────────────────────────
    ItemType.IRON_PICKAXE:    [(ItemType.IRON_INGOT, 3), (_S, 2)],
    ItemType.IRON_AXE:        [(ItemType.IRON_INGOT, 3), (_S, 2)],
    ItemType.IRON_SWORD:      [(ItemType.IRON_INGOT, 2), (_S, 1)],
    ItemType.IRON_SHOVEL:     [(ItemType.IRON_INGOT, 1), (_S, 2)],
    ItemType.IRON_HOE:        [(ItemType.IRON_INGOT, 2), (_S, 2)],

    # ── Torches: 1 coal/charcoal + 1 stick → 4 torches ──────────────────────
    ItemType.TORCH:           [(ItemType.COAL, 1), (_S, 1)],

    # ── Wooden building blocks ────────────────────────────────────────────────
    # Crafting Table: 4 planks in 2×2
    BlockType.CRAFTING_TABLE: [(_P, 4)],
    # Chest: 8 planks around a 3×3 (empty centre)
    BlockType.CHEST:          [(_P, 8)],
    # Bookshelf: 6 planks + 3 books
    BlockType.BOOKSHELF:      [(_P, 6), (ItemType.BOOK, 3)],

    # Door (3-wide × 2-tall = 6 planks)
    ItemType.WOODEN_DOOR:     [(_P, 6)],
    # Trapdoor: 6 planks in a 3×2
    ItemType.TRAPDOOR:        [(_P, 6)],
    # Fence: 4 planks + 2 sticks
    ItemType.FENCE:           [(_P, 4), (_S, 2)],
    # Fence gate: 2 planks + 4 sticks
    ItemType.FENCE_GATE:      [(_P, 2), (_S, 4)],
    # Pressure plate: 2 planks
    ItemType.PRESSURE_PLATE:  [(_P, 2)],
    # Button: 1 plank
    ItemType.BUTTON:          [(_P, 1)],
    # Slab: 3 planks → 6 slabs  (we yield 6 per craft)
    ItemType.SLAB:            [(_P, 3)],

    # ── Ladder: 7 sticks ─────────────────────────────────────────────────────
    ItemType.LADDER:          [(_S, 7)],

    # ── Bowl: 3 planks ────────────────────────────────────────────────────────
    ItemType.BOWL:            [(_P, 3)],

    # ── Boat: 5 planks ────────────────────────────────────────────────────────
    ItemType.BOAT:            [(_P, 5)],

    # ── Sign: 6 planks + 1 stick ─────────────────────────────────────────────
    ItemType.SIGN:            [(_P, 6), (_S, 1)],

    # ── Bow: 3 sticks + 3 string ─────────────────────────────────────────────
    ItemType.BOW:             [(_S, 3), (_STR, 3)],
    # Arrow: 1 flint + 1 stick + 1 feather
    ItemType.ARROW:           [(ItemType.FLINT, 1), (_S, 1), (ItemType.FEATHER, 1)],

    # ── Paper & Book ──────────────────────────────────────────────────────────
    ItemType.PAPER:           [("sugar_cane", 3)],
    ItemType.BOOK:            [(ItemType.PAPER, 3), (ItemType.LEATHER, 1)],

    # ── Fishing rod ──────────────────────────────────────────────────────────
    ItemType.FISHING_ROD:     [(_S, 3), (_STR, 2)],

    # ── Glass bottle ─────────────────────────────────────────────────────────
    ItemType.GLASS_BOTTLE:    [(BlockType.GLASS, 3)],

    # ── Flint and steel ──────────────────────────────────────────────────────
    ItemType.FLINT_AND_STEEL: [(ItemType.IRON_INGOT, 1), (ItemType.FLINT, 1)],

    # ── Cookie: 2 wheat + 1 cocoa beans ──────────────────────────────────────
    ItemType.COOKIE:          [("wheat", 2), ("cocoa_beans", 1)],
    # Bread: 3 wheat
    ItemType.BREAD:           [("wheat", 3)],
}

# Crafting output yields (defaults to 1 if not listed)
CRAFT_YIELD = {
    BlockType.OAK_PLANKS:    4,
    BlockType.BIRCH_PLANKS:  4,
    BlockType.SPRUCE_PLANKS: 4,
    ItemType.STICK:          4,
    ItemType.TORCH:          4,
    ItemType.SLAB:           6,
    ItemType.WOODEN_DOOR:    2,
    ItemType.TRAPDOOR:       2,
}


# ═══════════════════════════════════════════════════════════════════════
#  LEGACY CLASS SHIM  (main.py imports PlayerInventorySystem from here)
# ═══════════════════════════════════════════════════════════════════════
class PlayerInventorySystem:
    pass
