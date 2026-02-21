"""
main.py  –  Entry point.  Wires together settings, terrain, player and all
             the rendering / UI code.

Split from the original monolithic file:
  settings.py  – constants, block arrays (incl. BT_OCCLUDE transparency fix),
                  texture atlas, face UV / colour tables, matrix helpers.
  terrain.py   – Numba JIT gen_chunk() and build_mesh().
  player.py    – Player physics, collision, raycast.
"""

import math
import os
import random
import time as _time
from collections import deque

import numpy as np
import pyglet
import pyglet.gl as gl
from pyglet.window import key, mouse
import moderngl

# ── project modules ───────────────────────────────────────────────────────────
from block_textures import BlockType, BLOCK_DATA, HEART_TEX, BUBBLE_FRAMES
from inventory_system import PlayerInventorySystem, ItemType, FOOD_DATA, RECIPES, FURNACE_RECIPES, FUEL_DATA
from mobs import MobManager, MobType, Vec3 as MobVec3
import sun_moon

from settings import (
    CHUNK_SIZE, CHUNK_HEIGHT, SEA_LEVEL, BEDROCK_LEVEL, OCEAN_FLOOR,
    VIEW_DISTANCE, MESH_PER_FRAME, DAY_TICK_SPEED,
    GRAVITY, JUMP_VEL, PLAYER_SPEED, PLAYER_HEIGHT, EYE_OFFSET,
    MOUSE_SENS, FOV, WIN_W, WIN_H,
    ENABLE_WIREFRAME, EDGE_COLOR, EDGE_WIDTH,
    TILE_SIZE, USE_TEXTURES, USE_GREEDY_MESH, SHOW_ALL_FACES, USE_WATER,
    AIR, _BT_LIST, BT, N_BLOCK_TYPES,
    BT_SOLID, BT_TRANS, BT_LIQUID, BT_RENDER, BT_ROT,
    BT_OCCLUDE,          # ← the fixed occlusion mask (solid AND opaque)
    FACE_UV, FACE_COLORS,
    _atlas_img, _uv_map, _white_key,
    _persp, _lookat, _load_image_rgba,
)
from terrain import gen_chunk, build_mesh, build_mesh_split, _terrain_height, _biome_at
from player import Player

# ── world seed (must be module-level so terrain/player helpers can see it) ──
SEED = random.randint(1, 999_999_999_999)
print(f"World seed : {SEED}")
print(f"View dist  : {VIEW_DISTANCE}")


# ═══════════════════════════════════════════════════════════════════════
#  GLSL
# ═══════════════════════════════════════════════════════════════════════
VERT_GLSL = """
#version 330 core
in vec3 in_pos; in vec3 in_normal; in vec4 in_color; in vec2 in_uv;
uniform mat4 u_mvp;
out vec4 v_color; out vec3 v_normal; out vec2 v_uv;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_color = in_color; v_normal = in_normal; v_uv = in_uv;
}"""

FRAG_GLSL = """
#version 330 core
in vec4 v_color; in vec3 v_normal; in vec2 v_uv;
uniform vec3 u_sun; uniform float u_ambient; uniform vec3 u_tint;
uniform sampler2D u_tex;
out vec4 frag_color;
void main() {
    float diff  = max(dot(normalize(v_normal), normalize(u_sun)), 0.0);
    float light = u_ambient + (1.0 - u_ambient) * diff;
    vec4 texel  = texture(u_tex, v_uv);
    if (texel.a < 0.05) discard;        // cutout for leaves / cactus
    vec3 rgb    = texel.rgb * v_color.rgb;
    frag_color  = vec4(rgb * u_tint * light, v_color.a * texel.a);
}"""

SKY_VERT = """
#version 330 core
in vec2 in_pos; out float v_y; out vec2 v_pos;
void main() { gl_Position=vec4(in_pos,0.9999,1.0); v_y=in_pos.y*0.5+0.5; v_pos=in_pos; }"""

SKY_FRAG = """
#version 330 core
in float v_y; in vec2 v_pos; out vec4 frag_color;
uniform vec3 u_top; uniform vec3 u_bot;
void main() { frag_color=vec4(mix(u_bot,u_top,v_y),1.0); }"""

SUN_VERT = """
#version 330 core
in vec3 in_pos; in vec2 in_uv; uniform mat4 u_mvp; out vec2 v_uv;
void main() { gl_Position=u_mvp*vec4(in_pos,1.0); v_uv=in_uv; }"""

SUN_FRAG = """
#version 330 core
in vec2 v_uv; uniform sampler2D u_tex; out vec4 frag_color;
void main() { vec4 t=texture(u_tex,v_uv); if(t.a<0.01)discard; frag_color=t; }"""


# ═══════════════════════════════════════════════════════════════════════
#  WORLD
# ═══════════════════════════════════════════════════════════════════════
class Chunk:
    __slots__ = ('cx', 'cz', 'blocks', 'dirty')
    def __init__(self, cx, cz, blocks):
        self.cx=cx; self.cz=cz; self.blocks=blocks; self.dirty=True


class World:
    def __init__(self):
        self.chunks: dict[tuple, Chunk] = {}
        self.dirty_queue: deque[Chunk]  = deque()
        self.time_of_day = 0.0

    @staticmethod
    def world_to_chunk(wx, wz):
        return (int(math.floor(wx / CHUNK_SIZE)), int(math.floor(wz / CHUNK_SIZE)))

    @staticmethod
    def local(wx, wz):
        cx = int(math.floor(wx / CHUNK_SIZE))
        cz = int(math.floor(wz / CHUNK_SIZE))
        return cx, cz, wx - cx * CHUNK_SIZE, wz - cz * CHUNK_SIZE

    def get_or_gen(self, cx, cz):
        k = (cx, cz)
        if k not in self.chunks:
            blk = gen_chunk(cx, cz, SEED,
                BT[BlockType.GRASS], BT[BlockType.SAND], BT[BlockType.SNOW],
                BT[BlockType.DIRT],  BT[BlockType.STONE], BT[BlockType.BEDROCK],
                BT[BlockType.WATER],
                BT[BlockType.OAK_LOG],     BT[BlockType.OAK_LEAVES],
                BT[BlockType.SPRUCE_LOG],  BT[BlockType.SPRUCE_LEAVES],
                BT[BlockType.CACTUS],
                BT[BlockType.COAL_ORE],    BT[BlockType.IRON_ORE],
                BT[BlockType.GOLD_ORE],    BT[BlockType.DIAMOND_ORE],
                # Ocean biome blocks
                BT[BlockType.SAND_OCEAN],  BT[BlockType.KELP],
                BT[BlockType.SEAGRASS])
            ch = Chunk(cx, cz, blk)
            self.chunks[k] = ch
            self.dirty_queue.append(ch)
            for dx, dz in ((-1,0),(1,0),(0,-1),(0,1)):
                nb = self.chunks.get((cx+dx, cz+dz))
                if nb:
                    nb.dirty = True
                    if nb not in self.dirty_queue:
                        self.dirty_queue.appendleft(nb)
        return self.chunks[k]

    def get_block(self, wx, wy, wz):
        cx, cz, lx, lz = self.local(wx, wz)
        ch = self.chunks.get((cx, cz))
        if ch is None: return 0
        if not (0 <= lx < CHUNK_SIZE and 0 <= wy < CHUNK_HEIGHT and 0 <= lz < CHUNK_SIZE): return 0
        return int(ch.blocks[lx, wy, lz])

    def set_block(self, wx, wy, wz, bt):
        cx, cz, lx, lz = self.local(wx, wz)
        ch = self.chunks.get((cx, cz))
        if ch is None: return
        if not (0 <= lx < CHUNK_SIZE and 0 <= wy < CHUNK_HEIGHT and 0 <= lz < CHUNK_SIZE): return
        ch.blocks[lx, wy, lz] = bt
        ch.dirty = True
        if ch not in self.dirty_queue:
            self.dirty_queue.appendleft(ch)
        for test_lx, nb_dx in ((0, -1), (CHUNK_SIZE-1, 1)):
            if lx == test_lx:
                nb = self.chunks.get((cx+nb_dx, cz))
                if nb:
                    nb.dirty = True
                    if nb not in self.dirty_queue: self.dirty_queue.appendleft(nb)
        for test_lz, nb_dz in ((0, -1), (CHUNK_SIZE-1, 1)):
            if lz == test_lz:
                nb = self.chunks.get((cx, cz+nb_dz))
                if nb:
                    nb.dirty = True
                    if nb not in self.dirty_queue: self.dirty_queue.appendleft(nb)
        # Trigger water flow evaluation whenever a block changes
        self.schedule_water(wx, wy, wz)

    def is_solid(self, wx, wy, wz):
        bt = self.get_block(int(math.floor(wx)), int(math.floor(wy)), int(math.floor(wz)))
        return bt > 0 and bool(BT_SOLID[bt])

    def surface_y(self, wx, wz):
        cx, cz, lx, lz = self.local(wx, wz)
        lx = max(0, min(CHUNK_SIZE-1, lx));  lz = max(0, min(CHUNK_SIZE-1, lz))
        ch = self.chunks.get((cx, cz))
        if ch is None:
            return int(_terrain_height(wx, wz, SEED))
        for y in range(CHUNK_HEIGHT-1, -1, -1):
            if BT_SOLID[ch.blocks[lx, y, lz]]:
                return y
        return 0

    def has_sky_access(self, wx, wy, wz):
        ix = int(math.floor(wx));  iz = int(math.floor(wz));  iy = int(math.floor(wy))
        for y in range(iy+1, CHUNK_HEIGHT):
            if self.is_solid(ix, y, iz):
                return False
        return True

    # ── water flow simulation ─────────────────────────────────────────────────
    # We keep a pending set of (wx,wy,wz) blocks that need to be checked for
    # flow. When a block is placed / removed we schedule its neighbours.
    # Each tick we process a bounded batch so it never stalls the frame.

    def _init_water(self):
        if not hasattr(self, '_water_queue'):
            self._water_queue: deque = deque()
            self._water_scheduled: set = set()

    def schedule_water(self, wx: int, wy: int, wz: int):
        """Mark a position (and its neighbours) for water-flow evaluation."""
        self._init_water()
        for pos in [(wx,wy,wz),(wx-1,wy,wz),(wx+1,wy,wz),
                    (wx,wy-1,wz),(wx,wy+1,wz),(wx,wy,wz-1),(wx,wy,wz+1)]:
            if pos not in self._water_scheduled:
                self._water_scheduled.add(pos)
                self._water_queue.append(pos)

    def _water_tick(self, max_ops: int = 64):
        """Process up to max_ops water-flow steps."""
        self._init_water()
        ID_WATER = BT.get(BlockType.WATER, 0)
        if not ID_WATER:
            return
        ops = 0
        while self._water_queue and ops < max_ops:
            pos = self._water_queue.popleft()
            self._water_scheduled.discard(pos)
            ops += 1
            wx, wy, wz = pos
            bt = self.get_block(wx, wy, wz)

            # Only flow from a source water block
            if bt != ID_WATER:
                continue

            # Flow downward first
            below_bt = self.get_block(wx, wy - 1, wz)
            if below_bt == 0:  # air below → fall straight down
                self.set_block(wx, wy - 1, wz, ID_WATER)
                self.schedule_water(wx, wy - 1, wz)
                continue  # downward flow takes priority – skip horizontal

            # Spread horizontally only into cells with a solid floor below
            for nx, nz in ((wx-1,wz),(wx+1,wz),(wx,wz-1),(wx,wz+1)):
                nb_bt = self.get_block(nx, wy, nz)
                if nb_bt == 0:
                    below_nb = self.get_block(nx, wy - 1, nz)
                    if below_nb != 0:
                        self.set_block(nx, wy, nz, ID_WATER)
                        self.schedule_water(nx, wy, nz)

    def tick(self, dt):
        self.time_of_day = (self.time_of_day + DAY_TICK_SPEED * dt) % 24000.0
        self._water_tick()

    def sky(self):
        t = self.time_of_day
        def L(a,b,f): return (a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f)
        D  = ((0.35,0.67,1.0),(0.55,0.78,1.0))
        SK = ((0.9,0.4,0.1), (1.0,0.6,0.3))
        N  = ((0.02,0.03,0.12),(0.04,0.05,0.15))
        SR = ((0.9,0.5,0.2), (1.0,0.7,0.4))
        if   1000 <= t <= 12000: top,bot,amb = D[0],D[1],0.85
        elif t <= 13000:
            f=max(0.,min(1.,(t-12000)/1000)); top=L(D[0],SK[0],f); bot=L(D[1],SK[1],f); amb=0.85-0.45*f
        elif t <= 14000:
            f=max(0.,min(1.,(t-13000)/1000)); top=L(SK[0],N[0],f); bot=L(SK[1],N[1],f); amb=0.40-0.25*f
        elif t <= 22000: top,bot,amb = N[0],N[1],0.25
        elif t <= 23000:
            f=max(0.,min(1.,(t-22000)/1000)); top=L(N[0],SR[0],f); bot=L(N[1],SR[1],f); amb=0.12+0.28*f
        else:
            f=max(0.,min(1.,(t-23000)/1000)); top=L(SR[0],D[0],f); bot=L(SR[1],D[1],f); amb=0.40+0.45*f
        sun_dir, moon_dir, moon_tex = sun_moon.get_celestial(t)
        return top, bot, sun_dir, amb, moon_dir, moon_tex

    def phase_str(self):
        t = self.time_of_day
        if   1000 <= t <= 12000: return "☀ Day"
        elif t <= 14000:          return "🌅 Sunset"
        elif t <= 22000:          return "🌙 Night"
        else:                     return "🌄 Sunrise"


# ═══════════════════════════════════════════════════════════════════════
#  RENDERER
# ═══════════════════════════════════════════════════════════════════════
class Renderer:
    def __init__(self, ctx):
        self.ctx  = ctx
        self.prog = ctx.program(vertex_shader=VERT_GLSL, fragment_shader=FRAG_GLSL)
        self.skyp = ctx.program(vertex_shader=SKY_VERT,  fragment_shader=SKY_FRAG)
        self.sunp = ctx.program(vertex_shader=SUN_VERT,  fragment_shader=SUN_FRAG)
        self.atlas = ctx.texture(_atlas_img.size, 4, _atlas_img.tobytes())
        self.atlas.filter    = (moderngl.NEAREST, moderngl.NEAREST)
        self.atlas.repeat_x  = False;  self.atlas.repeat_y = False

        quad    = np.array([-1,-1,1,-1,-1,1,1,1], dtype=np.float32)
        sky_vbo = ctx.buffer(quad.tobytes())
        self.sky_vao = ctx.vertex_array(self.skyp, [(sky_vbo, '2f', 'in_pos')])

        sun_img = _load_image_rgba(sun_moon.SUN_TEX)
        self.sun_tex = ctx.texture(sun_img.size, 4, sun_img.tobytes()) if sun_img else None
        if self.sun_tex:
            self.sun_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self.moon_texes: dict[str, moderngl.Texture] = {}
        for mt in sun_moon.MOON_TEXS:
            img = _load_image_rgba(mt)
            if img:
                tex = ctx.texture(img.size, 4, img.tobytes())
                tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self.moon_texes[mt] = tex

        self.sun_vbo  = ctx.buffer(reserve=6*5*4)
        self.sun_vao2 = ctx.vertex_array(self.sunp, [(self.sun_vbo,  '3f 2f', 'in_pos', 'in_uv')])
        self.moon_vbo = ctx.buffer(reserve=6*5*4)
        self.moon_vao = ctx.vertex_array(self.sunp, [(self.moon_vbo, '3f 2f', 'in_pos', 'in_uv')])

        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        if SHOW_ALL_FACES: ctx.disable(moderngl.CULL_FACE)
        else:              ctx.enable(moderngl.CULL_FACE)
        ctx.front_face = 'ccw'
        self._vaos: dict = {}

    def upload(self, ch: Chunk, world: World) -> None:
        k = (ch.cx, ch.cz)
        if k in self._vaos:
            self._vaos[k][0].release()
            self._vaos[k][1].release()
            del self._vaos[k]

        S = CHUNK_SIZE;  H = CHUNK_HEIGHT
        nb_x_neg = np.zeros((S, H), dtype=np.uint8)
        nb_x_pos = np.zeros((S, H), dtype=np.uint8)
        nb_z_neg = np.zeros((S, H), dtype=np.uint8)
        nb_z_pos = np.zeros((S, H), dtype=np.uint8)
        left  = world.chunks.get((ch.cx-1, ch.cz));  left  and nb_x_neg.__setitem__(slice(None), left.blocks[S-1,:,:].T)
        right = world.chunks.get((ch.cx+1, ch.cz));  right and nb_x_pos.__setitem__(slice(None), right.blocks[0,:,:].T)
        back  = world.chunks.get((ch.cx, ch.cz-1));  back  and nb_z_neg.__setitem__(slice(None), back.blocks[:,:,S-1])
        front = world.chunks.get((ch.cx, ch.cz+1));  front and nb_z_pos.__setitem__(slice(None), front.blocks[:,:,0])

        # ── KEY FIX: pass BT_OCCLUDE (not BT_SOLID) as the occlusion mask ────
        # BT_SOLID includes leaves and cactus (solid=True, transparent=True).
        # Using BT_SOLID caused adjacent leaf/cactus faces to be culled away,
        # making the blocks appear hollow or completely invisible.
        # BT_OCCLUDE = solid AND NOT transparent, so only truly opaque blocks
        # hide their neighbours' faces.
        # Two-pass render: split opaque and transparent into separate VAOs
        vdata_o, vdata_t = build_mesh_split(
            ch.blocks, nb_x_neg, nb_x_pos, nb_z_neg, nb_z_pos,
            BT_RENDER, BT_OCCLUDE, BT_TRANS, BT_LIQUID, BT_ROT, FACE_COLORS, FACE_UV,
            ch.cx, ch.cz,
            1 if USE_GREEDY_MESH else 0,
        )
        if vdata_o.size == 0 and vdata_t.size == 0:
            ch.dirty = False;  return

        vbo_o = self.ctx.buffer(vdata_o.tobytes() if vdata_o.size > 0 else bytes(12))
        vao_o = self.ctx.vertex_array(
            self.prog,
            [(vbo_o, '3f 3f 4f 2f', 'in_pos', 'in_normal', 'in_color', 'in_uv')],
        )
        vbo_t = self.ctx.buffer(vdata_t.tobytes() if vdata_t.size > 0 else bytes(12))
        vao_t = self.ctx.vertex_array(
            self.prog,
            [(vbo_t, '3f 3f 4f 2f', 'in_pos', 'in_normal', 'in_color', 'in_uv')],
        )
        self._vaos[k] = (vao_o, vbo_o, vdata_o.size // 12,
                         vao_t, vbo_t, vdata_t.size // 12)
        ch.dirty = False

    def remove(self, k):
        if k in self._vaos:
            entry = self._vaos[k]
            entry[0].release(); entry[1].release()
            if len(entry) > 3: entry[3].release(); entry[4].release()
            del self._vaos[k]

    def rebuild_all(self, world: World) -> None:
        for k in list(self._vaos): self.remove(k)
        world.dirty_queue.clear()
        for ch in world.chunks.values():
            ch.dirty = True;  world.dirty_queue.append(ch)

    def draw(self, player: Player, world: World, w: int, h: int) -> None:
        top, bot, sun, amb, moon_dir, moon_tex = world.sky()
        self.ctx.clear(*bot, 1.0, depth=1.0)

        # Sky
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.skyp['u_top'].value = top;  self.skyp['u_bot'].value = bot
        self.sky_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.DEPTH_TEST)

        # Sun / Moon billboards
        if self.sun_tex:
            fx, fy, fz = player.forward()
            fl = math.sqrt(fx*fx+fy*fy+fz*fz);  fx/=fl; fy/=fl; fz/=fl
            rx,ry,rz = fy*0-fz*1, fz*0-fx*0, fx*1-fy*0
            rl = math.sqrt(rx*rx+ry*ry+rz*rz)
            if rl < 1e-6: rx,ry,rz = 1.,0.,0.
            else:         rx/=rl; ry/=rl; rz/=rl
            ux,uy,uz = ry*fz-rz*fy, rz*fx-rx*fz, rx*fy-ry*fx

            proj = _persp(FOV, w/h, 0.05, 800.)
            view = player.view_mat()
            mvp  = proj @ view
            self.sunp['u_mvp'].write(mvp.T.astype(np.float32).tobytes())

            def write_bb(vbo, center, size):
                cx2,cy2,cz2 = center
                sx2=rx*size; sy2=ry*size; sz2=rz*size
                ux2=ux*size; uy2=uy*size; uz2=uz*size
                p0=(cx2-sx2-ux2,cy2-sy2-uy2,cz2-sz2-uz2)
                p1=(cx2+sx2-ux2,cy2+sy2-uy2,cz2+sz2-uz2)
                p2=(cx2+sx2+ux2,cy2+sy2+uy2,cz2+sz2+uz2)
                p3=(cx2-sx2+ux2,cy2-sy2+uy2,cz2-sz2+uz2)
                data = np.array([*p0,0.,1.,*p1,1.,1.,*p2,1.,0.,
                                  *p0,0.,1.,*p2,1.,0.,*p3,0.,0.], dtype=np.float32)
                vbo.write(data.tobytes())

            dist = 200.
            sx2,sy2,sz2 = sun
            write_bb(self.sun_vbo,
                     (player.x+sx2*dist, player.y+sy2*dist, player.z+sz2*dist), 12.)
            self.sun_tex.use(0);  self.sunp['u_tex'].value = 0
            self.sun_vao2.render(moderngl.TRIANGLES)

            mtex = self.moon_texes.get(moon_tex)
            if mtex:
                mx2,my2,mz2 = moon_dir
                write_bb(self.moon_vbo,
                         (player.x+mx2*dist, player.y+my2*dist, player.z+mz2*dist), 10.)
                mtex.use(0);  self.sunp['u_tex'].value = 0
                self.moon_vao.render(moderngl.TRIANGLES)

        # World chunks
        if SHOW_ALL_FACES: self.ctx.disable(moderngl.CULL_FACE)
        else:              self.ctx.enable(moderngl.CULL_FACE)
        proj = _persp(FOV, w/h, 0.05, 800.)
        view = player.view_mat()
        mvp  = proj @ view
        self.prog['u_mvp'].write(mvp.T.astype(np.float32).tobytes())
        self.prog['u_sun'].value     = sun
        self.prog['u_ambient'].value = amb

        # Underwater tint: if player eye is inside a water block, tint blue-green
        eye_bt = world.get_block(int(math.floor(player.x)),
                                  int(math.floor(player.y + EYE_OFFSET - 0.1)),
                                  int(math.floor(player.z)))
        if eye_bt and eye_bt < N_BLOCK_TYPES and _BT_LIST[eye_bt] == BlockType.WATER:
            self.prog['u_tint'].value = (0.25, 0.55, 0.80)
            self.ctx.clear(0.08, 0.25, 0.55, 1.0, depth=1.0)
        else:
            self.prog['u_tint'].value = (1.0, 1.0, 1.0)

        self.atlas.use(0);  self.prog['u_tex'].value = 0
        # Pass 1: Opaque geometry – depth write ON
        self.ctx.depth_func = '<'
        for k2, entry in self._vaos.items():
            nv = entry[2]
            if nv > 0: entry[0].render(moderngl.TRIANGLES, vertices=nv)
        # Pass 2: Transparent geometry (water, leaves)
        # depth_mask=False: water doesn't write depth so layers blend through each other.
        # Back-to-front sort ensures correct painter's algorithm blending.
        self.ctx.enable(moderngl.BLEND)
        self.ctx.depth_mask = False
        px, pz = player.x, player.z
        trans_chunks = [(k2, e) for k2, e in self._vaos.items()
                        if len(e) > 3 and e[5] > 0]
        trans_chunks.sort(
            key=lambda t: -(t[0][0]*CHUNK_SIZE - px)**2 - (t[0][1]*CHUNK_SIZE - pz)**2
        )
        for k2, entry in trans_chunks:
            entry[3].render(moderngl.TRIANGLES, vertices=entry[5])
        self.ctx.depth_mask = True

        if ENABLE_WIREFRAME:
            self.ctx.wireframe  = True
            self.ctx.line_width = EDGE_WIDTH
            self.prog['u_ambient'].value = 1.0
            self.prog['u_tint'].value    = EDGE_COLOR
            for k2, (vao, vbo, nv) in self._vaos.items():
                if nv > 0: vao.render(moderngl.TRIANGLES, vertices=nv)
            self.ctx.wireframe = False


# ═══════════════════════════════════════════════════════════════════════
#  UI SYSTEM  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════

_icon_cache: dict = {}


def _load_icon(item: str) -> pyglet.image.AbstractImage:
    if item in _icon_cache:
        return _icon_cache[item]
    props = BLOCK_DATA.get(item, {})
    path  = (props.get("texture") or props.get("texture_top") or
             props.get("texture_side") or props.get("texture_bottom"))
    if not path or not os.path.exists(path):
        path = f"textures/assets/minecraft/textures/item/{item}.png"
    if path and os.path.exists(path):
        try:
            img = pyglet.image.load(path)
            _icon_cache[item] = img
            return img
        except Exception:
            pass
    img = pyglet.image.SolidColorImagePattern((180, 60, 60, 255)).create_image(16, 16)
    _icon_cache[item] = img
    return img


def _blank():
    return pyglet.image.SolidColorImagePattern((0, 0, 0, 0)).create_image(1, 1)


MC_BG     = (198, 198, 198)
MC_DARK   = (85,  85,  85)
MC_DARKER = (55,  55,  55)
MC_LIGHT  = (255, 255, 255)
MC_SLOT   = (139, 139, 139)
MC_TITLE  = (64,  64,  64, 255)


def _ui_scale(win):
    return max(1.5, min(3.0, win.height / 240.0))


MAX_STACK = 64


class SlotInventory:
    MAIN_SIZE   = 27
    HOTBAR_SIZE = 9
    SIZE        = 36

    def __init__(self):
        self._slots = [None] * self.SIZE
        self.selected = 0

    def hotbar_slot(self, hb):
        return self._slots[self.MAIN_SIZE + hb]

    def set_slot(self, idx, item, count):
        self._slots[idx] = (item, count) if (item and count > 0) else None

    @property
    def selected_item(self):
        s = self.hotbar_slot(self.selected)
        return s[0] if s else None

    @property
    def selected_count(self):
        s = self.hotbar_slot(self.selected)
        return s[1] if s else 0

    def add(self, item, count=1):
        for i in range(self.SIZE):
            if count <= 0: break
            s = self._slots[i]
            if s and s[0] == item and s[1] < MAX_STACK:
                can = min(count, MAX_STACK - s[1])
                self._slots[i] = (item, s[1] + can)
                count -= can
        for i in range(self.SIZE):
            if count <= 0: break
            if self._slots[i] is None:
                take = min(count, MAX_STACK)
                self._slots[i] = (item, take)
                count -= take
        return max(0, count)

    def remove(self, item, count=1):
        total = sum(s[1] for s in self._slots if s and s[0] == item)
        if total < count: return False
        for i in range(self.SIZE - 1, -1, -1):
            if count <= 0: break
            s = self._slots[i]
            if s and s[0] == item:
                take = min(s[1], count)
                rem  = s[1] - take
                self._slots[i] = (item, rem) if rem > 0 else None
                count -= take
        return True

    def count(self, item):
        return sum(s[1] for s in self._slots if s and s[0] == item)

    def has(self, item, n=1):
        return self.count(item) >= n

    def consume_selected(self, n=1):
        idx = self.MAIN_SIZE + self.selected
        s   = self._slots[idx]
        if not s: return
        rem = s[1] - n
        self._slots[idx] = (s[0], rem) if rem > 0 else None

    @property
    def hotbar(self):
        return [s[0] if s else None
                for s in self._slots[self.MAIN_SIZE:self.MAIN_SIZE+self.HOTBAR_SIZE]]

    def get_item_count(self, item):
        return self.count(item)

    @property
    def inventory(self):
        return self


class Inventory:
    def __init__(self):
        self.slots = SlotInventory()

    @property
    def selected_slot(self): return self.slots.selected
    @property
    def hotbar(self): return self.slots.hotbar
    def get_selected_item(self): return self.slots.selected_item
    def select_slot(self, i):    self.slots.selected = max(0, min(8, i))
    def add_item(self, item, count=1):    self.slots.add(item, count)
    def remove_item(self, item, count=1): return self.slots.remove(item, count)
    def has_item(self, item, count=1):    return self.slots.has(item, count)
    def get_item_count(self, item):       return self.slots.count(item)

    def eat(self, food_item, player):
        if food_item not in FOOD_DATA or not self.slots.has(food_item): return False
        data = FOOD_DATA[food_item]
        player.hunger     = min(player.max_hunger, player.hunger + data["hunger"])
        player.saturation = min(player.hunger, player.saturation + data["saturation"])
        self.slots.remove(food_item, 1)
        return True

    @property
    def inventory(self): return self


class MCSlot:
    def __init__(self, x, y, sz, batch):
        self.x=x; self.y=y; self.sz=sz; self._batch=batch
        self.item=None; self.count=0
        b=max(2,sz//16); self._b=b
        self._sd=pyglet.shapes.Rectangle(x,  y,  sz,   sz,   color=MC_DARKER,batch=batch)
        self._sh=pyglet.shapes.Rectangle(x+b,y+b,sz-b, sz-b, color=MC_LIGHT, batch=batch)
        self._si=pyglet.shapes.Rectangle(x+b,y+b,sz-b, sz-b, color=MC_DARK,  batch=batch)
        self._sf=pyglet.shapes.Rectangle(x+b,y+b,sz-b*2,sz-b*2,color=MC_SLOT,batch=batch)
        self._icon=pyglet.sprite.Sprite(_blank(),x=x+b,y=y+b,batch=batch)
        self._icon.visible=False
        self._lbl=pyglet.text.Label('',x=x+sz-b-1,y=y+b+1,anchor_x='right',anchor_y='bottom',
                                    font_size=max(6,sz//6),color=(255,255,255,255),batch=batch)
        self._hl=None

    def set(self,item,count=0):
        self.item=item; self.count=count
        if item and count>0:
            img=_load_icon(item); self._icon.image=img
            inner=self.sz-self._b*2; sc=inner/max(img.width,img.height); self._icon.scale=sc
            self._icon.x=self.x+self._b+(inner-img.width*sc)/2
            self._icon.y=self.y+self._b+(inner-img.height*sc)/2
            self._icon.visible=True; self._lbl.text=str(count) if count>1 else ''
        else:
            self._icon.image=_blank(); self._icon.visible=False; self._lbl.text=''

    def show(self,v):
        for o in (self._sd,self._sh,self._si,self._sf): o.visible=v
        self._icon.visible=v and bool(self.item) and self.count>0
        self._lbl.visible=v

    def hit(self,mx,my):
        return self.x<=mx<=self.x+self.sz and self.y<=my<=self.y+self.sz

    def delete(self):
        for o in (self._sd,self._sh,self._si,self._sf,self._icon,self._lbl): o.delete()


class MCPanel:
    def __init__(self,x,y,w,h,batch):
        b=4
        self._s1  =pyglet.shapes.Rectangle(x,  y,  w,   h,   color=MC_DARKER,batch=batch)
        self._s2  =pyglet.shapes.Rectangle(x+b,y+b,w-b, h-b, color=MC_LIGHT, batch=batch)
        self._face=pyglet.shapes.Rectangle(x+b,y+b,w-b*2,h-b*2,color=MC_BG, batch=batch)
    def show(self,v):
        for o in (self._s1,self._s2,self._face): o.visible=v
    def delete(self):
        for o in (self._s1,self._s2,self._face): o.delete()


class DragStack:
    def __init__(self):
        self._batch=pyglet.graphics.Batch(); self._icon=None; self._lbl=None
        self.item=None; self.count=0
    def start(self,item,count,x,y,sz):
        self.item=item; self.count=count
        if self._icon: self._icon.delete()
        if self._lbl:  self._lbl.delete()
        img=_load_icon(item)
        self._icon=pyglet.sprite.Sprite(img,x=x-sz//2,y=y-sz//2,batch=self._batch)
        self._icon.scale=sz/max(img.width,img.height)
        self._lbl=pyglet.text.Label(str(count) if count>1 else '',
                                     x=x+sz//2-2,y=y-sz//2+2,anchor_x='right',anchor_y='bottom',
                                     font_size=max(6,sz//6),color=(255,255,255,255),batch=self._batch)
    def update_label(self):
        if self._lbl: self._lbl.text=str(self.count) if self.count>1 else ''
    def move(self,x,y,sz):
        if self._icon: self._icon.x=x-sz//2; self._icon.y=y-sz//2
        if self._lbl:  self._lbl.x=x+sz//2-2; self._lbl.y=y-sz//2+2
    def stop(self):
        self.item=None; self.count=0
        if self._icon: self._icon.delete(); self._icon=None
        if self._lbl:  self._lbl.delete();  self._lbl=None
    def draw(self):
        if self._icon: self._batch.draw()
    @property
    def active(self): return self.item is not None


class HotbarUI:
    N=9; SZ_BASE=40; GAP=4
    def __init__(self,win,batch):
        self._win=win; self._batch=batch; self._slots=[]; self._panel=None; self._sel=None
        self._rebuild()
    def _sz(self): return int(self.SZ_BASE*_ui_scale(self._win))
    def _rebuild(self):
        for s in self._slots: s.delete()
        self._slots.clear()
        if self._panel: self._panel.delete()
        if self._sel:   self._sel.delete()
        sz=self._sz(); gap=self.GAP; n=self.N
        tw=n*sz+(n-1)*gap; x0=(self._win.width-tw)//2; y0=8
        self._panel=MCPanel(x0-4,y0-4,tw+8,sz+8,self._batch)
        self._sel=pyglet.shapes.Rectangle(x0-2,y0-2,sz+4,sz+4,color=(255,255,255),batch=self._batch)
        self._sel.opacity=180; self._x0=x0; self._y0=y0; self._sz_val=sz
        for i in range(n):
            sx=x0+i*(sz+gap); s=MCSlot(sx,y0,sz,self._batch); s.show(True); self._slots.append(s)
    def refresh(self,inv):
        sz=self._sz(); gap=self.GAP; tw=self.N*sz+(self.N-1)*gap; x0=(self._win.width-tw)//2
        sel=inv.slots.selected; self._sel.x=x0+sel*(sz+gap)-2
        for i,s in enumerate(self._slots):
            sx=x0+i*(sz+gap); s.x=sx; s._sd.x=sx; s._sh.x=sx+s._b; s._si.x=sx+s._b; s._sf.x=sx+s._b
            slot_data=inv.slots.hotbar_slot(i)
            if slot_data: s.set(slot_data[0],slot_data[1])
            else:         s.set(None,0)
    def resize(self): self._rebuild()
    @property
    def sz(self): return self._sz_val
    @property
    def y_top(self): return self._y0+self._sz_val


def _load_pyglet_img(path):
    """Load a pyglet image, returning a 1×1 blank on failure."""
    if path and os.path.exists(path):
        try:
            return pyglet.image.load(path)
        except Exception:
            pass
    return pyglet.image.SolidColorImagePattern((0,0,0,0)).create_image(1,1)


class StatusBarsUI:
    """
    HUD bars drawn as icon rows:
      • Health  – up to 10 heart icons  (heart.png), right-aligned
      • Hunger  – up to 10 chicken-leg bars, left-aligned  (kept as plain rects)
      • XP      – thin green bar across full width
      • Air     – up to 10 animated bubble icons above hearts, visible only underwater
    """
    N_ICONS  = 10          # hearts / bubbles shown at max
    ANIM_FPS = 12.0        # bubble animation frame rate

    def __init__(self, win, batch):
        self._win   = win
        self._batch = batch
        self._shapes: list = []

        # ── load textures once ────────────────────────────────────────────────
        self._heart_img = _load_pyglet_img(HEART_TEX)
        self._bubble_imgs = [_load_pyglet_img(p) for p in BUBBLE_FRAMES]
        self._bubble_frame = 0.0   # float accumulator for animation

        # sprite lists – rebuilt on resize
        self._heart_sprites:  list[pyglet.sprite.Sprite] = []
        self._bubble_sprites: list[pyglet.sprite.Sprite] = []

        # plain-rect sub-bars (XP, hunger)
        self._xp  = None; self._xp_w  = 0
        self._hg  = None; self._hg_w  = 0

        self._rebuild()

    # ── layout helpers ────────────────────────────────────────────────────────
    def _sz(self):
        return int(HotbarUI.SZ_BASE * _ui_scale(self._win))

    def _layout(self):
        """Return (tw, x0, bar_y, icon_sz) for current window size."""
        sz  = self._sz()
        n   = HotbarUI.N
        gap = HotbarUI.GAP
        tw  = n * sz + (n - 1) * gap
        x0  = (self._win.width - tw) // 2
        xp_y  = 8 + sz + 8
        bar_y = xp_y + 9 + 5      # just above XP bar
        icon_sz = max(10, sz // 3)
        return tw, x0, bar_y, icon_sz

    def _rebuild(self):
        # delete old shapes and sprites
        for s in self._shapes:
            s.delete()
        self._shapes.clear()
        for sp in self._heart_sprites + self._bubble_sprites:
            sp.delete()
        self._heart_sprites.clear()
        self._bubble_sprites.clear()

        tw, x0, bar_y, icon_sz = self._layout()

        # XP bar (full width, thin)
        xp_y = 8 + self._sz() + 8
        xp_bg = pyglet.shapes.Rectangle(x0, xp_y, tw, 5,
                                         color=(55,55,55), batch=self._batch)
        xp_fg = pyglet.shapes.Rectangle(x0, xp_y, 0,  5,
                                         color=(100,220,40), batch=self._batch)
        self._shapes += [xp_bg, xp_fg]
        self._xp = xp_fg; self._xp_w = tw

        # Hunger bar – left half, plain orange rectangles (no icon yet)
        hw = tw // 2 - 2
        hg_bg = pyglet.shapes.Rectangle(x0, bar_y, hw, 8,
                                         color=(50,50,50), batch=self._batch)
        hg_fg = pyglet.shapes.Rectangle(x0, bar_y, 0,  8,
                                         color=(220,140,30), batch=self._batch)
        self._shapes += [hg_bg, hg_fg]
        self._hg = hg_fg; self._hg_w = hw

        # ── Heart sprites – right half ────────────────────────────────────────
        gap_i = max(1, icon_sz // 6)
        heart_x0 = x0 + tw - self.N_ICONS * (icon_sz + gap_i) + gap_i
        for i in range(self.N_ICONS):
            sx = heart_x0 + i * (icon_sz + gap_i)
            sp = pyglet.sprite.Sprite(self._heart_img,
                                       x=sx, y=bar_y, batch=self._batch)
            sp.scale = icon_sz / max(self._heart_img.width,
                                     self._heart_img.height, 1)
            sp.visible = True
            self._heart_sprites.append(sp)

        # ── Bubble sprites – same width as hearts, one row above ─────────────
        bubble_y = bar_y + icon_sz + 3
        first_img = self._bubble_imgs[0] if self._bubble_imgs else self._heart_img
        for i in range(self.N_ICONS):
            sx = heart_x0 + i * (icon_sz + gap_i)
            sp = pyglet.sprite.Sprite(first_img,
                                       x=sx, y=bubble_y, batch=self._batch)
            sp.scale = icon_sz / max(first_img.width, first_img.height, 1)
            sp.visible = False
            self._bubble_sprites.append(sp)

    # ── update every frame ────────────────────────────────────────────────────
    def update(self, player, dt: float = 0.0):
        if not player:
            return

        # XP / hunger plain bars
        self._xp.width = int(self._xp_w * (player.xp % 100) / 100.)
        self._hg.width = int(self._hg_w * player.hunger / player.max_hunger)

        # ── Hearts: show filled/empty based on health ─────────────────────────
        hp_frac   = player.health / player.max_health   # 0.0 – 1.0
        filled    = hp_frac * self.N_ICONS              # how many full hearts
        for i, sp in enumerate(self._heart_sprites):
            # alpha: full heart = 255, empty heart = 60 (still visible, just dim)
            if i < int(filled):
                sp.opacity = 255
            elif i < math.ceil(filled):
                # partial heart – fade proportionally
                sp.opacity = max(60, int((filled - int(filled)) * 255))
            else:
                sp.opacity = 60
            sp.visible = True

        # ── Bubble animation + visibility ────────────────────────────────────
        AIR_MAX  = 10.0
        show_air = getattr(player, 'head_in_water', False)
        air_frac = getattr(player, 'air', AIR_MAX) / AIR_MAX  # 0–1

        if show_air and self._bubble_imgs:
            # advance animation frame
            self._bubble_frame = (self._bubble_frame + self.ANIM_FPS * dt) % len(self._bubble_imgs)
            frame_img = self._bubble_imgs[int(self._bubble_frame)]
            filled_bubbles = air_frac * self.N_ICONS
            for i, sp in enumerate(self._bubble_sprites):
                sp.image   = frame_img
                sp.visible = True
                if i < int(filled_bubbles):
                    sp.opacity = 255
                elif i < math.ceil(filled_bubbles):
                    sp.opacity = max(40, int((filled_bubbles - int(filled_bubbles)) * 255))
                else:
                    sp.opacity = 30   # ghost bubble: nearly invisible
        else:
            for sp in self._bubble_sprites:
                sp.visible = False

    def resize(self):
        self._rebuild()


class InventoryScreen:
    COLS=9; MAIN_ROWS=3
    def __init__(self,win):
        self._win=win; self._visible=False; self._batch=pyglet.graphics.Batch()
        self._panel=None; self._title=None; self._main_slots=[]; self._hb_slots=[]
        self._craft_slots=[]; self._craft_out=None; self._craft_arrow=None; self._sep=None
        self._craft_items=[None]*4; self._craft_result=None; self._rebuild()
    def _sz(self):  return int(32*_ui_scale(self._win))
    def _gap(self): return max(2,int(4*_ui_scale(self._win)))
    def _rebuild(self):
        for s in self._main_slots+self._hb_slots+self._craft_slots: s.delete()
        if self._craft_out:   self._craft_out.delete()
        if self._panel:       self._panel.delete()
        if self._title:       self._title.delete()
        if self._craft_arrow: self._craft_arrow.delete()
        if self._sep:         self._sep.delete()
        self._main_slots=[]; self._hb_slots=[]; self._craft_slots=[]; self._craft_out=None
        sz=self._sz(); gap=self._gap(); cols=self.COLS; mr=self.MAIN_ROWS
        inv_w=cols*(sz+gap)-gap; craft_row_w=2*(sz+gap)-gap; arrow_w=max(20,sz//2)
        craft_total=craft_row_w+arrow_w+gap+sz; pw=max(inv_w,craft_total)+gap*4
        craft_area_h=2*(sz+gap)-gap; inv_area_h=mr*(sz+gap)-gap
        ph=28+craft_area_h+12+inv_area_h+10+sz+gap*4+8
        px=(self._win.width-pw)//2; py=(self._win.height-ph)//2
        self._panel=MCPanel(px,py,pw,ph,self._batch)
        self._title=pyglet.text.Label("Inventory",x=px+gap*2,y=py+ph-22,
            font_size=max(8,int(10*_ui_scale(self._win))),color=MC_TITLE,batch=self._batch)
        self._title.visible=False
        craft_x=px+pw-gap*2-craft_total; craft_top_y=py+ph-28-craft_area_h
        for row in range(2):
            for col in range(2):
                sx=craft_x+col*(sz+gap); sy=craft_top_y+(1-row)*(sz+gap)
                s=MCSlot(sx,sy,sz,self._batch); s.show(False); self._craft_slots.append(s)
        arrow_x=craft_x+craft_row_w+gap; arrow_cy=craft_top_y+sz//2+(sz+gap)//2
        self._craft_arrow=pyglet.text.Label("→",x=arrow_x,y=arrow_cy,anchor_x='left',
            anchor_y='center',font_size=max(10,sz//2),color=(80,80,80,255),batch=self._batch)
        self._craft_arrow.visible=False
        out_x=arrow_x+arrow_w+gap; out_y=craft_top_y+(sz+gap)//2
        self._craft_out=MCSlot(out_x,out_y,sz,self._batch); self._craft_out.show(False)
        inv_x=px+gap*2; inv_top=py+8+sz+gap+10+(mr-1)*(sz+gap)
        for row in range(mr):
            for col in range(cols):
                sx=inv_x+col*(sz+gap); sy=inv_top-row*(sz+gap)
                s=MCSlot(sx,sy,sz,self._batch); s.show(False); self._main_slots.append(s)
        hb_y=py+gap*2
        for col in range(cols):
            sx=inv_x+col*(sz+gap); s=MCSlot(sx,hb_y,sz,self._batch); s.show(False); self._hb_slots.append(s)
        sep_y=hb_y+sz+4
        self._sep=pyglet.shapes.Rectangle(inv_x,sep_y,inv_w,2,color=MC_DARK,batch=self._batch)
        self._sep.visible=False
        self._px=px; self._py=py; self._pw=pw; self._ph=ph; self._panel.show(False)
    def _update_craft_output(self):
        counts={}
        for cd in self._craft_items:
            if cd: counts[cd[0]]=counts.get(cd[0],0)+cd[1]
        result=None
        for out,reqs in RECIPES.items():
            needed={}
            for ing,cnt in reqs: needed[ing]=needed.get(ing,0)+cnt
            if counts==needed: result=out; break
        self._craft_result=result
        if result: self._craft_out.set(result,1)
        else:      self._craft_out.set(None,0)
    def _refresh(self,inv):
        s=inv.slots
        for i,su in enumerate(self._main_slots):
            d=s._slots[i]; su.set(d[0],d[1]) if d else su.set(None,0)
        for j,su in enumerate(self._hb_slots):
            d=s._slots[27+j]; su.set(d[0],d[1]) if d else su.set(None,0)
        for i,su in enumerate(self._craft_slots):
            cd=self._craft_items[i]; su.set(cd[0],cd[1]) if cd else su.set(None,0)
        self._update_craft_output()
    def show(self,inv):
        self._visible=True; self._panel.show(True); self._title.visible=True
        self._craft_arrow.visible=True; self._sep.visible=True
        for s in self._main_slots+self._hb_slots+self._craft_slots: s.show(True)
        self._craft_out.show(True); self._refresh(inv)
    def hide(self):
        self._visible=False; self._panel.show(False); self._title.visible=False
        if self._craft_arrow: self._craft_arrow.visible=False
        if self._sep: self._sep.visible=False
        for s in self._main_slots+self._hb_slots+self._craft_slots: s.show(False)
        if self._craft_out: self._craft_out.show(False)
    def refresh(self,inv):
        if self._visible: self._refresh(inv)
    def draw(self):
        if self._visible: self._batch.draw()
    def resize(self): self._rebuild()
    def hit_main(self,x,y):
        for i,s in enumerate(self._main_slots):
            if s.hit(x,y): return i,s.item,s.count
        return None
    def hit_hotbar(self,x,y):
        for i,s in enumerate(self._hb_slots):
            if s.hit(x,y): return i,s.item,s.count
        return None
    def hit_craft_input(self,x,y):
        for i,s in enumerate(self._craft_slots):
            if s.hit(x,y): return i
        return None
    def hit_craft_output(self,x,y):
        return bool(self._craft_out and self._craft_out.hit(x,y))
    def take_craft_output(self,inv):
        if not self._craft_result: return None
        reqs=RECIPES.get(self._craft_result,[])
        for ing,cnt in reqs:
            have=sum(cd[1] for cd in self._craft_items if cd and cd[0]==ing)
            if have<cnt: return None
        for ing,cnt_needed in reqs:
            left=cnt_needed
            for i in range(4):
                cd=self._craft_items[i]
                if cd and cd[0]==ing and left>0:
                    take=min(cd[1],left); rem=cd[1]-take
                    self._craft_items[i]=(ing,rem) if rem>0 else None; left-=take
        inv.slots.add(self._craft_result,1); self._refresh(inv); return self._craft_result
    def in_panel(self,x,y):
        return (self._visible and self._px<=x<=self._px+self._pw and self._py<=y<=self._py+self._ph)


class CraftingTableScreen:
    COLS=9; MAIN_ROWS=3
    def __init__(self,win):
        self._win=win; self._visible=False; self._batch=pyglet.graphics.Batch()
        self._panel=None; self._title=None; self._craft_slots=[]; self._craft_out=None
        self._craft_arrow=None; self._main_slots=[]; self._hb_slots=[]; self._sep=None
        self._craft_items=[None]*9; self._craft_result=None; self._rebuild()
    def _sz(self):  return int(32*_ui_scale(self._win))
    def _gap(self): return max(2,int(4*_ui_scale(self._win)))
    def _rebuild(self):
        for s in self._craft_slots+self._main_slots+self._hb_slots: s.delete()
        if self._craft_out:   self._craft_out.delete()
        if self._panel:       self._panel.delete()
        if self._title:       self._title.delete()
        if self._craft_arrow: self._craft_arrow.delete()
        if self._sep:         self._sep.delete()
        self._craft_slots=[]; self._main_slots=[]; self._hb_slots=[]; self._craft_out=None
        sz=self._sz(); gap=self._gap(); cols=self.COLS; mr=self.MAIN_ROWS
        inv_w=cols*(sz+gap)-gap; craft_row_w=3*(sz+gap)-gap; arrow_w=max(24,sz//2)
        craft_total=craft_row_w+arrow_w+gap+sz; pw=max(inv_w,craft_total)+gap*4
        craft_area_h=3*(sz+gap)-gap; inv_area_h=mr*(sz+gap)-gap
        ph=28+craft_area_h+12+inv_area_h+10+sz+gap*4+8
        px=(self._win.width-pw)//2; py=(self._win.height-ph)//2
        self._panel=MCPanel(px,py,pw,ph,self._batch)
        self._title=pyglet.text.Label("Crafting Table",x=px+gap*2,y=py+ph-22,
            font_size=max(8,int(10*_ui_scale(self._win))),color=MC_TITLE,batch=self._batch)
        self._title.visible=False
        craft_x=px+gap*2; craft_top=py+ph-28-craft_area_h
        for row in range(3):
            for col in range(3):
                sx=craft_x+col*(sz+gap); sy=craft_top+(2-row)*(sz+gap)
                s=MCSlot(sx,sy,sz,self._batch); s.show(False); self._craft_slots.append(s)
        arrow_x=craft_x+craft_row_w+gap; arrow_cy=craft_top+craft_area_h//2
        self._craft_arrow=pyglet.text.Label("→",x=arrow_x,y=arrow_cy,anchor_x='left',
            anchor_y='center',font_size=max(10,sz//2),color=(80,80,80,255),batch=self._batch)
        self._craft_arrow.visible=False
        out_x=arrow_x+arrow_w+gap; out_y=craft_top+craft_area_h//2-sz//2
        self._craft_out=MCSlot(out_x,out_y,sz,self._batch); self._craft_out.show(False)
        inv_x=px+gap*2; inv_top=py+8+sz+gap+10+(mr-1)*(sz+gap)
        for row in range(mr):
            for col in range(cols):
                sx=inv_x+col*(sz+gap); sy=inv_top-row*(sz+gap)
                s=MCSlot(sx,sy,sz,self._batch); s.show(False); self._main_slots.append(s)
        hb_y=py+gap*2
        for col in range(cols):
            sx=inv_x+col*(sz+gap); s=MCSlot(sx,hb_y,sz,self._batch); s.show(False); self._hb_slots.append(s)
        sep_y=hb_y+sz+4
        self._sep=pyglet.shapes.Rectangle(inv_x,sep_y,inv_w,2,color=MC_DARK,batch=self._batch)
        self._sep.visible=False
        self._px=px; self._py=py; self._pw=pw; self._ph=ph; self._panel.show(False)
    def _update_craft_output(self):
        counts={}
        for cd in self._craft_items:
            if cd: counts[cd[0]]=counts.get(cd[0],0)+cd[1]
        result=None
        for out,reqs in RECIPES.items():
            needed={}
            for ing,cnt in reqs: needed[ing]=needed.get(ing,0)+cnt
            if counts==needed: result=out; break
        self._craft_result=result
        if result: self._craft_out.set(result,1)
        else:      self._craft_out.set(None,0)
    def _refresh(self,inv):
        s=inv.slots
        for i,su in enumerate(self._main_slots):
            d=s._slots[i]; su.set(d[0],d[1]) if d else su.set(None,0)
        for j,su in enumerate(self._hb_slots):
            d=s._slots[27+j]; su.set(d[0],d[1]) if d else su.set(None,0)
        for i,su in enumerate(self._craft_slots):
            cd=self._craft_items[i]; su.set(cd[0],cd[1]) if cd else su.set(None,0)
        self._update_craft_output()
    def show(self,inv):
        self._visible=True; self._panel.show(True); self._title.visible=True
        if self._craft_arrow: self._craft_arrow.visible=True
        if self._sep: self._sep.visible=True
        for s in self._craft_slots+self._main_slots+self._hb_slots: s.show(True)
        if self._craft_out: self._craft_out.show(True)
        self._refresh(inv)
    def hide(self):
        self._visible=False; self._panel.show(False); self._title.visible=False
        if self._craft_arrow: self._craft_arrow.visible=False
        if self._sep: self._sep.visible=False
        for s in self._craft_slots+self._main_slots+self._hb_slots: s.show(False)
        if self._craft_out: self._craft_out.show(False)
    def refresh(self,inv):
        if self._visible: self._refresh(inv)
    def draw(self):
        if self._visible: self._batch.draw()
    def resize(self): self._rebuild()
    def hit_craft_input(self,x,y):
        for i,s in enumerate(self._craft_slots):
            if s.hit(x,y): return i
        return None
    def hit_craft_output(self,x,y):
        return bool(self._craft_out and self._craft_out.hit(x,y))
    def take_craft_output(self,inv):
        if not self._craft_result: return None
        reqs=RECIPES.get(self._craft_result,[])
        for ing,cnt in reqs:
            have=sum(cd[1] for cd in self._craft_items if cd and cd[0]==ing)
            if have<cnt: return None
        for ing,cnt_needed in reqs:
            left=cnt_needed
            for i in range(9):
                cd=self._craft_items[i]
                if cd and cd[0]==ing and left>0:
                    take=min(cd[1],left); rem=cd[1]-take
                    self._craft_items[i]=(ing,rem) if rem>0 else None; left-=take
        inv.slots.add(self._craft_result,1); self._refresh(inv); return self._craft_result
    def hit_main(self,x,y):
        for i,s in enumerate(self._main_slots):
            if s.hit(x,y): return i,s.item,s.count
        return None
    def hit_hotbar(self,x,y):
        for i,s in enumerate(self._hb_slots):
            if s.hit(x,y): return i,s.item,s.count
        return None
    def in_panel(self,x,y):
        return (self._visible and self._px<=x<=self._px+self._pw and self._py<=y<=self._py+self._ph)


class FurnaceScreen:
    COLS=9; MAIN_ROWS=3
    def __init__(self,win):
        self._win=win; self._visible=False; self._batch=pyglet.graphics.Batch()
        self._panel=None; self._title=None
        self._in_slot=None; self._fuel_slot=None; self._out_slot=None
        self._flame_bg=None; self._flame_fg=None; self._flame_h=0
        self._arrow_bg=None; self._arrow_fg=None; self._arrow_w=0; self._arrow_lbl=None
        self._main_slots=[]; self._hb_slots=[]; self._sep=None
        self.input_item=None; self.input_count=0; self.fuel_item=None; self.fuel_count=0
        self.output_item=None; self.output_count=0
        self.burn_time=0.; self.burn_max=0.; self.cook_time=0.
        self._rebuild()
    def _sz(self):  return int(32*_ui_scale(self._win))
    def _gap(self): return max(2,int(4*_ui_scale(self._win)))
    def _rebuild(self):
        for s in self._main_slots+self._hb_slots: s.delete()
        for o in (self._in_slot,self._fuel_slot,self._out_slot):
            if o: o.delete()
        if self._panel: self._panel.delete()
        if self._title: self._title.delete()
        if self._sep:   self._sep.delete()
        for o in (self._flame_bg,self._flame_fg,self._arrow_bg,self._arrow_fg,self._arrow_lbl):
            if o:
                try: o.delete()
                except: pass
        self._main_slots=[]; self._hb_slots=[]
        self._in_slot=self._fuel_slot=self._out_slot=None
        self._flame_bg=self._flame_fg=self._arrow_bg=self._arrow_fg=self._arrow_lbl=None
        sz=self._sz(); gap=self._gap(); cols=self.COLS; mr=self.MAIN_ROWS
        inv_w=cols*(sz+gap)-gap; arrow_w=max(40,sz)
        furn_w=sz+gap+sz//3+gap+arrow_w+gap+sz; pw=max(inv_w,furn_w)+gap*4
        furn_h=2*sz+gap; inv_area_h=mr*(sz+gap)-gap
        ph=28+furn_h+12+inv_area_h+10+sz+gap*4+8
        px=(self._win.width-pw)//2; py=(self._win.height-ph)//2
        self._panel=MCPanel(px,py,pw,ph,self._batch)
        self._title=pyglet.text.Label("Furnace",x=px+gap*2,y=py+ph-22,
            font_size=max(8,int(10*_ui_scale(self._win))),color=MC_TITLE,batch=self._batch)
        self._title.visible=False
        furn_x=px+(pw-furn_w)//2; furn_top=py+ph-28-furn_h
        in_x=furn_x; in_y=furn_top+sz+gap
        self._in_slot=MCSlot(in_x,in_y,sz,self._batch); self._in_slot.show(False)
        fuel_y=furn_top
        self._fuel_slot=MCSlot(in_x,fuel_y,sz,self._batch); self._fuel_slot.show(False)
        fl_x=in_x+sz//4; fl_y=fuel_y+sz+gap//2; fl_w=sz//2; fl_h=max(8,gap*2+2)
        self._flame_bg=pyglet.shapes.Rectangle(fl_x,fl_y,fl_w,fl_h,color=(60,30,10),batch=self._batch)
        self._flame_fg=pyglet.shapes.Rectangle(fl_x,fl_y,fl_w,0,   color=(255,120,0),batch=self._batch)
        self._flame_h=fl_h
        arr_x=in_x+sz+gap; arr_cy=furn_top+sz//2+gap//2+sz//2; arr_h=sz//2
        self._arrow_bg=pyglet.shapes.Rectangle(arr_x,arr_cy-arr_h//2,arrow_w,arr_h,color=(60,60,60),batch=self._batch)
        self._arrow_fg=pyglet.shapes.Rectangle(arr_x,arr_cy-arr_h//2,0,      arr_h,color=(80,200,80),batch=self._batch)
        self._arrow_lbl=pyglet.text.Label("→",x=arr_x+arrow_w//2,y=arr_cy,anchor_x='center',anchor_y='center',
            font_size=max(10,sz//2),color=(180,180,180,200),batch=self._batch)
        self._arrow_w=arrow_w
        out_x=arr_x+arrow_w+gap; out_y=arr_cy-sz//2
        self._out_slot=MCSlot(out_x,out_y,sz,self._batch); self._out_slot.show(False)
        inv_x=px+gap*2; inv_top=py+8+sz+gap+10+(mr-1)*(sz+gap)
        for row in range(mr):
            for col in range(cols):
                sx=inv_x+col*(sz+gap); sy=inv_top-row*(sz+gap)
                s=MCSlot(sx,sy,sz,self._batch); s.show(False); self._main_slots.append(s)
        hb_y=py+gap*2
        for col in range(cols):
            sx=inv_x+col*(sz+gap); s=MCSlot(sx,hb_y,sz,self._batch); s.show(False); self._hb_slots.append(s)
        sep_y=hb_y+sz+4
        self._sep=pyglet.shapes.Rectangle(inv_x,sep_y,inv_w,2,color=MC_DARK,batch=self._batch)
        self._sep.visible=False
        self._px=px; self._py=py; self._pw=pw; self._ph=ph; self._panel.show(False)
        for o in (self._flame_bg,self._flame_fg,self._arrow_bg,self._arrow_fg,self._arrow_lbl):
            if o:
                try: o.visible=False
                except: pass
    def _refresh_inv(self,inv):
        s=inv.slots
        for i,su in enumerate(self._main_slots):
            d=s._slots[i]; su.set(d[0],d[1]) if d else su.set(None,0)
        for j,su in enumerate(self._hb_slots):
            d=s._slots[27+j]; su.set(d[0],d[1]) if d else su.set(None,0)
    def _update_bars(self):
        ratio=max(0.,min(1.,self.burn_time/self.burn_max)) if self.burn_max>0 else 0.
        self._flame_fg.height=int(self._flame_h*ratio)
        cratio=max(0.,min(1.,self.cook_time/5.)); self._arrow_fg.width=int(self._arrow_w*cratio)
    def show(self,inv):
        self._visible=True; self._panel.show(True); self._title.visible=True
        self._in_slot.show(True); self._fuel_slot.show(True); self._out_slot.show(True)
        self._in_slot.set(self.input_item,self.input_count)
        self._fuel_slot.set(self.fuel_item,self.fuel_count)
        self._out_slot.set(self.output_item,self.output_count)
        for o in (self._flame_bg,self._flame_fg,self._arrow_bg,self._arrow_fg,self._arrow_lbl):
            if o:
                try: o.visible=True
                except: pass
        for s in self._main_slots+self._hb_slots: s.show(True)
        if self._sep: self._sep.visible=True
        self._refresh_inv(inv); self._update_bars()
    def hide(self):
        self._visible=False; self._panel.show(False); self._title.visible=False
        for o in (self._in_slot,self._fuel_slot,self._out_slot):
            if o: o.show(False)
        for o in (self._flame_bg,self._flame_fg,self._arrow_bg,self._arrow_fg,self._arrow_lbl):
            if o:
                try: o.visible=False
                except: pass
        for s in self._main_slots+self._hb_slots: s.show(False)
        if self._sep: self._sep.visible=False
    def tick(self,dt,inv):
        if self.burn_time<=0 and self.fuel_item:
            burn=FUEL_DATA.get(self.fuel_item,0)
            if burn>0:
                self.burn_time=burn; self.burn_max=burn; self.fuel_count-=1
                if self.fuel_count<=0: self.fuel_item=None; self.fuel_count=0
                if self._visible: self._fuel_slot.set(self.fuel_item,self.fuel_count)
        if self.burn_time>0:
            self.burn_time=max(0.,self.burn_time-dt)
            if self.input_item and self.input_item in FURNACE_RECIPES:
                self.cook_time+=dt
                if self.cook_time>=5.:
                    out=FURNACE_RECIPES[self.input_item]
                    if self.output_item is None or self.output_item==out:
                        self.output_item=out; self.output_count+=1; self.input_count-=1
                        if self.input_count<=0: self.input_item=None; self.input_count=0
                        if self._visible:
                            self._in_slot.set(self.input_item,self.input_count)
                            self._out_slot.set(self.output_item,self.output_count)
                    self.cook_time=0.
            else: self.cook_time=0.
        else: self.cook_time=0.
        if self._visible: self._update_bars()
    def refresh(self,inv):
        if self._visible: self._refresh_inv(inv)
    def draw(self):
        if self._visible: self._batch.draw()
    def resize(self): self._rebuild()
    def hit_in(self,x,y):   return bool(self._in_slot   and self._in_slot.hit(x,y))
    def hit_fuel(self,x,y): return bool(self._fuel_slot  and self._fuel_slot.hit(x,y))
    def hit_out(self,x,y):  return bool(self._out_slot   and self._out_slot.hit(x,y))
    def hit_main(self,x,y):
        for i,s in enumerate(self._main_slots):
            if s.hit(x,y): return i,s.item,s.count
        return None
    def hit_hotbar(self,x,y):
        for i,s in enumerate(self._hb_slots):
            if s.hit(x,y): return i,s.item,s.count
        return None
    def in_panel(self,x,y):
        return (self._visible and self._px<=x<=self._px+self._pw and self._py<=y<=self._py+self._ph)


class UIManager:
    NONE='none'; INV='inv'; CRAFT='craft'; FURNACE='furnace'
    def __init__(self,win):
        self._win=win; self._hb=pyglet.graphics.Batch()
        self._hotbar=HotbarUI(win,self._hb); self._bars=StatusBarsUI(win,self._hb)
        h=win.height
        self._l1=pyglet.text.Label('',x=8,y=h-18,font_size=11,color=(255,255,255,220),batch=self._hb)
        self._l2=pyglet.text.Label('',x=8,y=h-36,font_size=11,color=(255,255,255,220),batch=self._hb)
        self._l3=pyglet.text.Label('',x=8,y=h-54,font_size=11,color=(255,255,255,220),batch=self._hb)
        self._l4=pyglet.text.Label('',x=8,y=16,  font_size=10,color=(255,255,180,220),batch=self._hb)
        self._inv_scr=InventoryScreen(win); self._cft_scr=CraftingTableScreen(win)
        self._fur_scr=FurnaceScreen(win); self._drag=DragStack(); self._mode=self.NONE
    @property
    def is_open(self): return self._mode!=self.NONE
    def _hide_all(self): self._inv_scr.hide(); self._cft_scr.hide(); self._fur_scr.hide()
    def open_inv(self):
        self._hide_all(); self._mode=self.INV; self._inv_scr.show(self._win.inventory)
        self._win.set_exclusive_mouse(False)
    def open_craft(self):
        self._hide_all(); self._mode=self.CRAFT; self._cft_scr.show(self._win.inventory)
        self._win.set_exclusive_mouse(False)
    def open_furnace(self):
        self._hide_all(); self._mode=self.FURNACE; self._fur_scr.show(self._win.inventory)
        self._win.set_exclusive_mouse(False)
    def close(self):
        if self._drag.active:
            self._win.inventory.slots.add(self._drag.item,self._drag.count); self._drag.stop()
        self._hide_all(); self._mode=self.NONE; self._win.set_exclusive_mouse(True)
    def toggle_inv(self):
        if self._mode==self.NONE: self.open_inv()
        else: self.close()
    def tick(self,dt): self._fur_scr.tick(dt,self._win.inventory)
    def update_hud(self,player,world,fps_buf,dt=0.0):
        fps=sum(fps_buf)/len(fps_buf) if fps_buf else 0
        inv=self._win.inventory; sel=inv.slots.selected_item or '—'
        bid=int(_biome_at(int(player.x),int(player.z),SEED))
        bname=["Plains","Forest","Desert","Snow Taiga","Ocean"][bid] if 0<=bid<=4 else "?"
        self._l1.text=(f"FPS {fps:.0f}  |  {world.phase_str()}"
                       f"  ({int(world.time_of_day/1000):02d}:00)  Biome:{bname}")
        self._l2.text=(f"XYZ {player.x:.1f} {player.y:.1f} {player.z:.1f}"
                       f"  Chunks:{len(world.chunks)}")
        water_str = ""
        if getattr(player,'head_in_water',False):
            air_pct = int(getattr(player,'air',10.0) / 10.0 * 100)
            water_str = f"  🌊 Air:{air_pct}%"
        elif getattr(player,'in_water',False):
            water_str = "  🌊 Swimming"
        self._l3.text=(f"HP {int(player.health)}/20  Hunger {int(player.hunger)}/20"
                       f"  XP {player.xp}{water_str}")
        if self.is_open:
            hint="[ESC/E to close]"
        elif getattr(player,'in_water',False):
            hint=f"[{sel}]  WASD=swim  Space=rise  Shift=dive  LMB=mine  RMB=place"
        else:
            hint=f"[{sel}]  WASD  Space=jump  LMB=mine  RMB=place  1-9=slot  F=eat  E=inv"
        self._l4.text=hint
        self._bars.update(player, dt); self._hotbar.refresh(inv)
    def draw(self):
        self._hb.draw()
        if   self._mode==self.INV:     self._inv_scr.draw()
        elif self._mode==self.CRAFT:   self._cft_scr.draw()
        elif self._mode==self.FURNACE: self._fur_scr.draw()
        self._drag.draw()
    def _slot_sz(self): return self._hotbar.sz
    def _pick_up_slot(self,item,count,x,y):
        self._drag.start(item,count,x,y,self._slot_sz())
    def _place_into_inv_slot(self,inv,slot_idx,item,count,mx=0,my=0):
        existing=inv.slots._slots[slot_idx]
        if existing is None:
            take=min(count,MAX_STACK); inv.slots._slots[slot_idx]=(item,take); return count-take
        elif existing[0]==item:
            can=min(count,MAX_STACK-existing[1]); inv.slots._slots[slot_idx]=(item,existing[1]+can); return count-can
        else:
            old_item,old_count=existing; inv.slots._slots[slot_idx]=(item,count)
            self._drag.item=old_item; self._drag.count=old_count; self._drag.update_label()
            if self._drag._icon:
                img=_load_icon(old_item); self._drag._icon.image=img
                self._drag._icon.scale=self._slot_sz()/max(img.width,img.height)
            return -1
    def _shift_click_inv_slot(self,inv,slot_idx,scr):
        existing=inv.slots._slots[slot_idx]
        if not existing: return
        item,count=existing
        target_range=range(27,36) if slot_idx<27 else range(0,27)
        for i in target_range:
            if count<=0: break
            s=inv.slots._slots[i]
            if s and s[0]==item and s[1]<MAX_STACK:
                can=min(count,MAX_STACK-s[1]); inv.slots._slots[i]=(item,s[1]+can); count-=can
        for i in target_range:
            if count<=0: break
            if inv.slots._slots[i] is None:
                take=min(count,MAX_STACK); inv.slots._slots[i]=(item,take); count-=take
        inv.slots._slots[slot_idx]=(item,count) if count>0 else None
        scr.refresh(inv); self._hotbar.refresh(inv)
    def on_press(self,x,y,btn,mod=0):
        if self._mode==self.NONE: return False
        inv=self._win.inventory; sz=self._slot_sz()
        shift_held=bool(mod & key.MOD_SHIFT)
        if self._mode==self.INV:
            scr=self._inv_scr
            if scr.hit_craft_output(x,y):
                if not self._drag.active:
                    result=scr.take_craft_output(inv)
                    if result: self._pick_up_slot(result,1,x,y); scr.refresh(inv); self._hotbar.refresh(inv)
                return True
            ci=scr.hit_craft_input(x,y)
            if ci is not None:
                cd=scr._craft_items[ci]
                if self._drag.active:
                    if cd is None or cd[0]==self._drag.item:
                        new_cnt=(cd[1] if cd else 0)+1; scr._craft_items[ci]=(self._drag.item,new_cnt)
                        self._drag.count-=1
                        if self._drag.count<=0: self._drag.stop()
                        else: self._drag.move(x,y,sz); self._drag.update_label()
                    scr._update_craft_output(); scr.refresh(inv)
                elif cd:
                    self._pick_up_slot(cd[0],cd[1],x,y); scr._craft_items[ci]=None
                    scr._update_craft_output(); scr.refresh(inv)
                return True
            res=scr.hit_main(x,y)
            if res is not None:
                idx,item,count=res
                if shift_held and item and not self._drag.active:
                    self._shift_click_inv_slot(inv,idx,scr)
                elif self._drag.active:
                    left=self._place_into_inv_slot(inv,idx,self._drag.item,self._drag.count,x,y)
                    if left==-1: pass
                    elif left<=0: self._drag.stop()
                    else: self._drag.count=left; self._drag.move(x,y,sz); self._drag.update_label()
                    scr.refresh(inv); self._hotbar.refresh(inv)
                elif item:
                    self._pick_up_slot(item,count,x,y); inv.slots._slots[idx]=None
                    scr.refresh(inv); self._hotbar.refresh(inv)
                return True
            res=scr.hit_hotbar(x,y)
            if res is not None:
                hb_idx,item,count=res; slot_idx=27+hb_idx
                if shift_held and item and not self._drag.active:
                    self._shift_click_inv_slot(inv,slot_idx,scr)
                elif self._drag.active:
                    left=self._place_into_inv_slot(inv,slot_idx,self._drag.item,self._drag.count,x,y)
                    if left==-1: pass
                    elif left<=0: self._drag.stop()
                    else: self._drag.count=left; self._drag.move(x,y,sz); self._drag.update_label()
                    scr.refresh(inv); self._hotbar.refresh(inv)
                elif item:
                    self._pick_up_slot(item,count,x,y); inv.slots._slots[slot_idx]=None
                    scr.refresh(inv); self._hotbar.refresh(inv)
                return True
        elif self._mode==self.CRAFT:
            scr=self._cft_scr
            if scr.hit_craft_output(x,y):
                if not self._drag.active:
                    result=scr.take_craft_output(inv)
                    if result: self._pick_up_slot(result,1,x,y); scr.refresh(inv); self._hotbar.refresh(inv)
                return True
            ci=scr.hit_craft_input(x,y)
            if ci is not None:
                cd=scr._craft_items[ci]
                if self._drag.active:
                    if cd is None or cd[0]==self._drag.item:
                        new_cnt=(cd[1] if cd else 0)+1; scr._craft_items[ci]=(self._drag.item,new_cnt)
                        self._drag.count-=1
                        if self._drag.count<=0: self._drag.stop()
                        else: self._drag.move(x,y,sz); self._drag.update_label()
                    scr._update_craft_output(); scr.refresh(inv)
                elif cd:
                    self._pick_up_slot(cd[0],cd[1],x,y); scr._craft_items[ci]=None
                    scr._update_craft_output(); scr.refresh(inv)
                return True
            res=scr.hit_main(x,y)
            if res is not None:
                idx,item,count=res
                if self._drag.active:
                    left=self._place_into_inv_slot(inv,idx,self._drag.item,self._drag.count,x,y)
                    if left==-1: pass
                    elif left<=0: self._drag.stop()
                    else: self._drag.count=left; self._drag.move(x,y,sz); self._drag.update_label()
                elif item:
                    self._pick_up_slot(item,count,x,y); inv.slots._slots[idx]=None
                scr.refresh(inv); self._hotbar.refresh(inv); return True
            res=scr.hit_hotbar(x,y)
            if res is not None:
                hb_idx,item,count=res; slot_idx=27+hb_idx
                if self._drag.active:
                    left=self._place_into_inv_slot(inv,slot_idx,self._drag.item,self._drag.count,x,y)
                    if left==-1: pass
                    elif left<=0: self._drag.stop()
                    else: self._drag.count=left; self._drag.move(x,y,sz); self._drag.update_label()
                elif item:
                    self._pick_up_slot(item,count,x,y); inv.slots._slots[slot_idx]=None
                scr.refresh(inv); self._hotbar.refresh(inv); return True
        elif self._mode==self.FURNACE:
            fs=self._fur_scr
            if fs.hit_in(x,y):
                if self._drag.active:
                    if fs.input_item is None or fs.input_item==self._drag.item:
                        add=min(self._drag.count,MAX_STACK-fs.input_count)
                        fs.input_item=self._drag.item; fs.input_count+=add; self._drag.count-=add
                        if self._drag.count<=0: self._drag.stop()
                        else: self._drag.move(x,y,sz); self._drag.update_label()
                        fs._in_slot.set(fs.input_item,fs.input_count)
                elif fs.input_item:
                    self._pick_up_slot(fs.input_item,fs.input_count,x,y)
                    fs.input_item=None; fs.input_count=0; fs._in_slot.set(None,0)
                return True
            if fs.hit_fuel(x,y):
                if self._drag.active:
                    if fs.fuel_item is None or fs.fuel_item==self._drag.item:
                        add=min(self._drag.count,MAX_STACK-fs.fuel_count)
                        fs.fuel_item=self._drag.item; fs.fuel_count+=add; self._drag.count-=add
                        if self._drag.count<=0: self._drag.stop()
                        else: self._drag.move(x,y,sz); self._drag.update_label()
                        fs._fuel_slot.set(fs.fuel_item,fs.fuel_count)
                elif fs.fuel_item:
                    self._pick_up_slot(fs.fuel_item,fs.fuel_count,x,y)
                    fs.fuel_item=None; fs.fuel_count=0; fs._fuel_slot.set(None,0)
                return True
            if fs.hit_out(x,y):
                if fs.output_item and not self._drag.active:
                    self._pick_up_slot(fs.output_item,fs.output_count,x,y)
                    fs.output_item=None; fs.output_count=0; fs._out_slot.set(None,0)
                return True
            res=fs.hit_main(x,y)
            if res is not None:
                idx,item,count=res
                if self._drag.active:
                    left=self._place_into_inv_slot(inv,idx,self._drag.item,self._drag.count,x,y)
                    if left==-1: pass
                    elif left<=0: self._drag.stop()
                    else: self._drag.count=left; self._drag.move(x,y,sz); self._drag.update_label()
                elif item:
                    self._pick_up_slot(item,count,x,y); inv.slots._slots[idx]=None
                fs.refresh(inv); self._hotbar.refresh(inv); return True
            res=fs.hit_hotbar(x,y)
            if res is not None:
                hb_idx,item,count=res; slot_idx=27+hb_idx
                if self._drag.active:
                    left=self._place_into_inv_slot(inv,slot_idx,self._drag.item,self._drag.count,x,y)
                    if left==-1: pass
                    elif left<=0: self._drag.stop()
                    else: self._drag.count=left; self._drag.move(x,y,sz); self._drag.update_label()
                elif item:
                    self._pick_up_slot(item,count,x,y); inv.slots._slots[slot_idx]=None
                fs.refresh(inv); self._hotbar.refresh(inv); return True
        return False
    def on_release(self,x,y,btn): pass
    def on_motion(self,x,y):
        if self._drag.active: self._drag.move(x,y,self._slot_sz())
    def resize(self,w,h):
        self._l1.y=h-18; self._l2.y=h-36; self._l3.y=h-54
        self._hotbar.resize(); self._bars.resize()
        mode=self._mode
        self._inv_scr.resize(); self._cft_scr.resize(); self._fur_scr.resize()
        inv=self._win.inventory
        if   mode==self.INV:     self._inv_scr.show(inv)
        elif mode==self.CRAFT:   self._cft_scr.show(inv)
        elif mode==self.FURNACE: self._fur_scr.show(inv)


# ═══════════════════════════════════════════════════════════════════════
#  GAME WINDOW
# ═══════════════════════════════════════════════════════════════════════
class GameWindow(pyglet.window.Window):
    def __init__(self):
        cfg = pyglet.gl.Config(major_version=3, minor_version=3,
                               forward_compatible=True, depth_size=24, double_buffer=True)
        super().__init__(WIN_W, WIN_H,
                         caption="Minecraft-Python  [pyglet · ModernGL · Numba]",
                         config=cfg, resizable=True, vsync=False)
        self.ctx = moderngl.create_context()
        self.ren = Renderer(self.ctx)
        self.world = World()

        self.inventory = Inventory()
        self.inventory.slots.add(ItemType.WOODEN_PICKAXE, 1)
        self.inventory.slots.add(ItemType.WOODEN_AXE,     1)
        self.inventory.slots.add(ItemType.WOODEN_SWORD,   1)
        self.inventory.slots.add(BlockType.OAK_PLANKS,   10)
        self.inventory.slots.add(ItemType.APPLE,           5)
        self.inventory.slots.add(BlockType.COBBLESTONE,   20)
        self.inventory.slots.add(BlockType.CRAFTING_TABLE, 1)
        self.inventory.slots.add(BlockType.FURNACE,        1)

        self.mob_manager = MobManager()
        self.mob_manager.init_gl(self.ctx)

        self._ui = UIManager(self)

        print("⚡ Generating spawn area…")
        for ddx in range(-2, 3):
            for ddz in range(-2, 3):
                self.world.get_or_gen(ddx, ddz)
        self.ren.rebuild_all(self.world)
        while self.world.dirty_queue:
            ch = self.world.dirty_queue.popleft()
            self.ren.upload(ch, self.world)
        print(f"✅ {len(self.world.chunks)} chunks, "
              f"{sum(v[2] for v in self.ren._vaos.values())} verts")

        sx, sz = CHUNK_SIZE // 2, CHUNK_SIZE // 2
        sy = self.world.surface_y(sx, sz)
        if sy <= BEDROCK_LEVEL + 1:
            sy = int(_terrain_height(sx, sz, SEED))
        self.player = Player(sx + 0.5, float(sy + 3), sz + 0.5)
        print(f"🧍 Spawn ({sx},{sy+3:.0f},{sz})")

        spawn_pos = MobVec3(self.player.x, self.player.y, self.player.z)
        self.mob_manager.spawn_initial_passive(spawn_pos, self.world, count=6)

        self.keys      = set()
        self._captured = False
        self._t_prev   = _time.perf_counter()
        self._fps_buf  = deque(maxlen=30)
        self.set_exclusive_mouse(True);  self._captured = True
        pyglet.clock.schedule_interval(self._tick, 1.0 / 60.0)

    def _stream(self):
        cx0, cz0 = World.world_to_chunk(self.player.x, self.player.z)
        for ddx in range(-VIEW_DISTANCE, VIEW_DISTANCE+1):
            for ddz in range(-VIEW_DISTANCE, VIEW_DISTANCE+1):
                self.world.get_or_gen(cx0+ddx, cz0+ddz)
        to_del = [k for k in list(self.world.chunks)
                  if abs(k[0]-cx0)>VIEW_DISTANCE+2 or abs(k[1]-cz0)>VIEW_DISTANCE+2]
        for k in to_del:
            self.ren.remove(k);  del self.world.chunks[k]
        uploaded = 0
        while self.world.dirty_queue and uploaded < MESH_PER_FRAME:
            ch = self.world.dirty_queue.popleft()
            if ch.dirty: self.ren.upload(ch, self.world);  uploaded += 1

    def _tick(self, dt):
        now = _time.perf_counter()
        dt  = min(now - self._t_prev, 0.05);  self._t_prev = now
        self._fps_buf.append(1.0 / max(dt, 1e-4))
        self.world.tick(dt);  self._ui.tick(dt)
        mx = mz = 0.
        if not self._ui.is_open:
            if key.W in self.keys: mz += 1.
            if key.S in self.keys: mz -= 1.
            if key.A in self.keys: mx -= 1.
            if key.D in self.keys: mx += 1.
            mag = math.sqrt(mx*mx + mz*mz)
            if mag > 0: mx /= mag;  mz /= mag
        self.player.update(dt, self.world,
                           self.keys if not self._ui.is_open else set(), mx, mz)
        player_mob_pos = MobVec3(self.player.x, self.player.y, self.player.z)
        self.mob_manager.update_mobs(dt, player_mob_pos, self.world,
                                     self.player, self.world.time_of_day)
        self._stream()
        self._ui.update_hud(self.player, self.world, self._fps_buf, dt)

    def on_draw(self):
        self.ren.draw(self.player, self.world, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        cx, cy = self.width//2, self.height//2
        pyglet.shapes.Line(cx-12, cy, cx+12, cy, 2, (255,255,255,180)).draw()
        pyglet.shapes.Line(cx, cy-12, cx, cy+12, 2, (255,255,255,180)).draw()
        self._ui.draw()
        self.ctx.enable(moderngl.DEPTH_TEST)
        proj = _persp(FOV, self.width/self.height, 0.05, 800.0)
        view = self.player.view_mat()
        mvp  = proj @ view
        self.mob_manager.draw_mobs(mvp, self.player.yaw, self.player.pitch)

    def on_key_press(self, sym, mod):
        self.keys.add(sym)
        if sym == key.ESCAPE:
            if self._ui.is_open: self._ui.close()
            else: self.set_exclusive_mouse(False);  self._captured = False
        if sym == key.E: self._ui.toggle_inv()
        if sym == key.B:
            global ENABLE_WIREFRAME;  ENABLE_WIREFRAME = not ENABLE_WIREFRAME
        if sym == key.G:
            global USE_GREEDY_MESH;  USE_GREEDY_MESH = not USE_GREEDY_MESH
            print(f"Greedy mesh: {USE_GREEDY_MESH}");  self.ren.rebuild_all(self.world)
        if sym == key.H:
            global SHOW_ALL_FACES;  SHOW_ALL_FACES = not SHOW_ALL_FACES
            print(f"Show all faces: {SHOW_ALL_FACES}");  self.ren.rebuild_all(self.world)
        if sym == key.R:
            print("Rebuilding…");  self.ren.rebuild_all(self.world)
        if sym == key.F:
            sel = self.inventory.slots.selected_item
            if sel and sel in FOOD_DATA:
                self.inventory.eat(sel, self.player);  self._ui._hotbar.refresh(self.inventory)
        for i, ks in enumerate([key._1, key._2, key._3, key._4, key._5,
                                  key._6, key._7, key._8, key._9]):
            if sym == ks:
                self.inventory.slots.selected = i;  self._ui._hotbar.refresh(self.inventory)

    def on_key_release(self, sym, mod):
        self.keys.discard(sym)

    def on_mouse_motion(self, x, y, dx, dy):
        self._ui.on_motion(x, y)
        if not self._captured or self._ui.is_open: return
        self.player.yaw   = (self.player.yaw - dx * MOUSE_SENS) % 360.
        self.player.pitch = max(-89., min(89., self.player.pitch + dy * MOUSE_SENS))

    def on_mouse_drag(self, x, y, dx, dy, btn, mod):
        self._ui.on_motion(x, y)
        if not self._ui.is_open: self.on_mouse_motion(x, y, dx, dy)

    def on_mouse_press(self, x, y, btn, mod):
        if self._ui.is_open:
            self._ui.on_press(x, y, btn, mod);  return
        if not self._captured:
            self.set_exclusive_mouse(True);  self._captured = True;  return
        rc = self.player.raycast(self.world)
        if rc is None: return
        hit, norm, prev = rc
        hx, hy, hz = hit
        if btn == mouse.LEFT:
            bt = self.world.get_block(hx, hy, hz)
            if bt and bt < N_BLOCK_TYPES:
                bname = _BT_LIST[bt];  props = BLOCK_DATA.get(bname, {})
                if not props.get("unbreakable", False):
                    drop = props.get("drops")
                    if drop:
                        self.inventory.slots.add(drop, 1);  self._ui._hotbar.refresh(self.inventory)
                    self.world.set_block(hx, hy, hz, 0)
        elif btn == mouse.RIGHT:
            bt = self.world.get_block(hx, hy, hz)
            if bt and bt < N_BLOCK_TYPES:
                bname = _BT_LIST[bt]
                if bname == BlockType.CRAFTING_TABLE: self._ui.open_craft();  return
                if bname == BlockType.FURNACE:        self._ui.open_furnace(); return
            sel = self.inventory.slots.selected_item
            if sel and sel in BLOCK_DATA and self.inventory.slots.has(sel):
                px2, py2, pz2 = prev
                pdx = abs(px2 - int(math.floor(self.player.x)))
                pdz = abs(pz2 - int(math.floor(self.player.z)))
                pdy = py2 - int(math.floor(self.player.y))
                if pdx < 1 and pdz < 1 and 0 <= pdy <= 2: return
                bt_id = BT.get(sel, 0)
                if bt_id:
                    self.world.set_block(px2, py2, pz2, bt_id)
                    self.inventory.slots.consume_selected(1)
                    self._ui._hotbar.refresh(self.inventory)

    def on_mouse_release(self, x, y, btn, mod): pass

    def on_resize(self, w, h):
        self.ctx.viewport = (0, 0, w, h);  self._ui.resize(w, h)

    def on_close(self):
        pyglet.app.exit()


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    win = GameWindow()
    print("Running!")
    pyglet.app.run()