#!/usr/bin/env python3
"""Copy the pinned doomgeneric sources to DST and apply our patches. Every patch asserts it applied (an engine bump fails loudly).

  patch_engine.py SRC DST gpu|cpu

Both targets: the scripted player hooks into G_BuildTiccmd, and I_FinishUpdate stops converting to RGB (the palette index screen
I_VideoBuffer and the palette `colors` are what we hash and present). gpu only: PTX-compatibility fixes and the parallel-draw hooks.
"""
import os
import re
import shutil
import sys

src, dst, target = sys.argv[1:4]
assert target in ("gpu", "cpu")
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(src, dst)


def patch(name, fn):
    path = f"{dst}/{name}"
    s = open(path).read()
    s2 = fn(s)
    assert s2 != s, f"patch had no effect: {name}"
    open(path, "w").write(s2)


def sub1(old, new):
    def f(s):
        assert s.count(old) == 1, f"anchor not unique/found: {old[:50]!r}"
        return s.replace(old, new, 1)
    return f


def first(old, new):
    """Replace the first occurrence only (the engine has several variants of the column/span routines; the first is the compiled one)."""
    def f(s):
        assert old in s, f"anchor not found: {old[:50]!r}"
        return s.replace(old, new, 1)
    return f


def chain(*fns):
    def f(s):
        for fn in fns:
            s = fn(s)
        return s
    return f


# ---- both targets
patch("g_game.c", sub1("    // special buttons\n", "    { extern void player_ticcmd(ticcmd_t*); player_ticcmd(cmd); }\n    // special buttons\n"))
patch("i_video.c", chain(
    sub1("static struct color colors[256];", "struct color colors[256];"),
    sub1("void I_FinishUpdate (void)\n{", "void I_FinishUpdate (void)\n{\n    return;   /* no RGB conversion: the screen is I_VideoBuffer + colors */")))
if target == "cpu":
    sys.exit(0)

# ---- gpu only
# sounds are disabled: self-referential initializers (S_sfx links into itself) cannot be emitted as PTX globals
patch("sounds.c", lambda s: re.sub(r"&S_sfx\[[A-Za-z0-9_]+\]", "NULL", s))
# LLVM turns these constant string tables into self-referential relative lookup tables that PTX cannot emit; volatile blocks it
patch("m_menu.c", lambda s: re.sub(r"static char \*(detailNames|msgNames)\[2\]", r"static char * volatile \1[2]", s))
# PTX cannot statically initialise a pointer to an array element; set these at startup instead (doom_glue gpu_init)
patch("tables.c", chain(
    lambda s: re.sub(r"^const fixed_t \*finecosine = &finesine\[FINEANGLES/4\];", "const fixed_t *finecosine;", s, flags=re.M),
    lambda s: s + "\nvoid gpu_init_tables(void) { finecosine = &finesine[FINEANGLES/4]; }\n"))
patch("g_game.c", chain(
    lambda s: re.sub(r"^static boolean \*mousebuttons = &mousearray\[1\];.*", "static boolean *mousebuttons;", s, flags=re.M),
    lambda s: re.sub(r"^static boolean \*joybuttons = &joyarray\[1\];.*", "static boolean *joybuttons;", s, flags=re.M),
    lambda s: s + "\nvoid gpu_init_game_ptrs(void) { mousebuttons = &mousearray[1]; joybuttons = &joyarray[1]; }\n"))

# ---- parallel drawing hooks (glue/gpu_par.c): columns and spans are queued instead of drawn; 255 workers draw them at flush points
DECL = ("extern volatile int gpu_par_active; void gpu_flush(void); void gpu_q_col(byte*, const byte*, const byte*, int, int, int, int); "
        "void gpu_q_span(byte*, const byte*, const byte*, unsigned, unsigned, int, int);\n")


def r_draw(s):
    assert '#include "r_local.h"' in s
    s = s.replace('#include "r_local.h"', '#include "r_local.h"\n' + DECL, 1)
    s = first("    frac = dc_texturemid + (dc_yl-centery)*fracstep; \n",
             "    frac = dc_texturemid + (dc_yl-centery)*fracstep; \n"
             "    if (gpu_par_active) { gpu_q_col(dest, dc_source, dc_colormap, frac, fracstep, count, dc_x); return; }\n")(s)
    s = first("    count = ds_x2 - ds_x1;\n",
             "    count = ds_x2 - ds_x1;\n"
             "    if (gpu_par_active) { gpu_q_span(dest, ds_source, ds_colormap, position, step, count, ds_x1); return; }\n")(s)
    s, n1 = re.subn(r"(void R_DrawFuzzColumn \(void\)\s*\{)", r"\1 gpu_flush();", s, count=1)       # these read the screen they draw on
    s, n2 = re.subn(r"(void R_DrawTranslatedColumn \(void\)\s*\{)", r"\1 gpu_flush();", s, count=1)
    assert n1 == 1 and n2 == 1
    return s


patch("r_draw.c", r_draw)
patch("r_main.c", chain(
    sub1("    R_DrawPlanes ();\n", "    R_DrawPlanes ();\n    { extern void gpu_epoch(void); gpu_epoch(); }\n"),
    sub1("    R_DrawMasked ();\n", "    R_DrawMasked ();\n    { extern void gpu_epoch(void); gpu_epoch(); }\n")))

# ---- split mode (glue/gpu_par.c, gpu_glue.c gpu_doom_sim/gpu_doom_post): the view is only LOGGED by the tick task and drawn by tile tasks;
# the display work after the view (HU_Drawer, border, menu, I_FinishUpdate) runs later in a post task, by re-entering D_Display at that point.
def d_display(s):
    s = s.replace("void D_Display (void)\n{", "extern int gpu_split_resume(void); extern void gpu_view_begin(int); extern int gpu_view_end(int);\nvoid D_Display (void)\n{", 1)
    s, n = re.subn(r"(\n\s*boolean\s+redrawsbar;\n)", r"\1    if (gpu_split_resume()) { wipe = false; goto gpu_resume_point; }\n", s, count=1)
    assert n == 1
    s, n = re.subn(r"(if \(gamestate == GS_LEVEL && !automapactive && gametic\)\s*\n\s*)R_RenderPlayerView \(&players\[displayplayer\]\);",
                   r"\1{ gpu_view_begin(wipe); R_RenderPlayerView (&players[displayplayer]); if (gpu_view_end((int)gametic)) return; }", s, count=1)
    assert n == 1
    s, n = re.subn(r"(\n)(\s*if \(gamestate == GS_LEVEL && gametic\)\s*\n\s*HU_Drawer \(\);)", r"\1gpu_resume_point: ;\2", s, count=1)
    assert n == 1
    return s


patch("d_main.c", d_display)

# ---- optional phase profiler (DOOM_PROF=1): cycle counters per tick phase in doom_prof[] (glue/gpu_par.c), read by tests/doomtest.cu -DDOOM_PROF_READ
if os.environ.get("DOOM_PROF"):
    PR = "{ extern void gpu_prof(int); gpu_prof(%d); }\n"
    patch("d_main.c", chain(
        sub1("    TryRunTics (); // will run at least one tic\n", "    TryRunTics (); // will run at least one tic\n" + PR % 1),
        sub1("    // draw buffered stuff to screen\n    I_UpdateNoBlit ();\n", "    // draw buffered stuff to screen\n    I_UpdateNoBlit ();\n" + PR % 2)))
    patch("r_main.c", chain(
        sub1("    R_RenderBSPNode (numnodes-1);\n", "    R_RenderBSPNode (numnodes-1);\n" + PR % 3),
        sub1("    R_DrawPlanes ();\n    { extern void gpu_epoch(void); gpu_epoch(); }\n", "    R_DrawPlanes ();\n    { extern void gpu_epoch(void); gpu_epoch(); }\n" + PR % 4),
        sub1("    R_DrawMasked ();\n    { extern void gpu_epoch(void); gpu_epoch(); }\n", "    R_DrawMasked ();\n    { extern void gpu_epoch(void); gpu_epoch(); }\n" + PR % 5)))


# ---- split mode, flat planes: R_DrawPlanes only LOGS each flat plane (visplane, flat data, light table, height); the doom_planes tasks run
# R_MakeSpans/R_MapPlane (span discovery and the per-span arithmetic) for the planes in parallel, one thread per plane, with thread-local span state;
# the spans go into the span log (atomic append, epoch 0) and the column tiles draw their pixels.
# Planes never overlap each other or the walls, so any order gives the serial picture; the masked pass (sprites) runs after all of them.
PLANE_CODE = r"""
void gpu_plane_exec(int i);

/* ---- GPU split mode (patch_engine.py): flat planes executed by the doom_planes tasks ---- */
typedef struct { visplane_t* pl; byte* src; lighttable_t** zl; fixed_t ph; int lump; } gpu_plane_t;
static gpu_plane_t gpu_planes[MAXVISPLANES];
static int gpu_nplanes;
extern int gpu_split_logging(void);
extern byte* ylookup[]; extern int columnofs[];
void gpu_q_span_mt(byte*, const byte*, const byte*, unsigned, unsigned, int, int);
static int gpu_ran_serial;   /* 1: the planes were already executed on the game thread (a fuzz column needed the finished screen) */
void gpu_planes_reset(void) { gpu_nplanes = 0; gpu_ran_serial = 0; }
int gpu_plane_count(void) { return gpu_ran_serial ? 0 : gpu_nplanes; }
int gpu_plane_log(visplane_t* pl, byte* src, lighttable_t** zl, fixed_t ph, int lump)
{
    if (!gpu_split_logging()) return 0;
    gpu_plane_t* g = &gpu_planes[gpu_nplanes++];
    g->pl = pl; g->src = src; g->zl = zl; g->ph = ph; g->lump = lump;
    return 1;
}
void gpu_planes_release(void) { for (int i = 0; i < gpu_nplanes; i++) W_ReleaseLumpNum(gpu_planes[i].lump); gpu_nplanes = 0; gpu_ran_serial = 0; }
static void gpu_map_plane(int y, int x1, int x2, const gpu_plane_t* g)
{
    angle_t angle; fixed_t distance, length, xfrac, yfrac, xstep, ystep; unsigned index; lighttable_t* cm;
    distance = FixedMul(g->ph, yslope[y]);
    xstep = FixedMul(distance, basexscale);
    ystep = FixedMul(distance, baseyscale);
    length = FixedMul(distance, distscale[x1]);
    angle = (viewangle + xtoviewangle[x1]) >> ANGLETOFINESHIFT;
    xfrac = viewx + FixedMul(finecosine[angle], length);
    yfrac = -viewy - FixedMul(finesine[angle], length);
    if (fixedcolormap) cm = fixedcolormap;
    else { index = distance >> LIGHTZSHIFT; if (index >= MAXLIGHTZ) index = MAXLIGHTZ - 1; cm = g->zl[index]; }
    unsigned int position = ((xfrac << 10) & 0xffff0000) | ((yfrac >> 6) & 0x0000ffff);
    unsigned int step = ((xstep << 10) & 0xffff0000) | ((ystep >> 6) & 0x0000ffff);
    gpu_q_span_mt(ylookup[y] + columnofs[x1], g->src, cm, position, step, x2 - x1, x1);   /* pixels are drawn by the column tiles (clipped to their band) */
}
void gpu_plane_exec(int i)
{
    const gpu_plane_t* g = &gpu_planes[i]; visplane_t* pl = g->pl;
    int ss[SCREENHEIGHT];     /* span starts: thread-local copy of spanstart[] */
    int stop = pl->maxx + 1;
    for (int x = pl->minx; x <= stop; x++) {
        int t1 = pl->top[x-1], b1 = pl->bottom[x-1], t2 = pl->top[x], b2 = pl->bottom[x];
        while (t1 < t2 && t1 <= b1) { gpu_map_plane(t1, ss[t1], x-1, g); t1++; }
        while (b1 > b2 && b1 >= t1) { gpu_map_plane(b1, ss[b1], x-1, g); b1--; }
        while (t2 < t1 && t2 <= b2) { ss[t2] = x; t2++; }
        while (b2 > b1 && b2 >= t2) { ss[b2] = x; b2--; }
    }
}
void gpu_planes_exec_serial(void) { if (gpu_ran_serial) return; for (int i = 0; i < gpu_nplanes; i++) gpu_plane_exec(i); gpu_ran_serial = 1; }
"""


def r_plane(s):
    s = s.rstrip("\n") + "\n" + PLANE_CODE
    old = "\tpl->top[pl->maxx+1] = 0xff;\n\tpl->top[pl->minx-1] = 0xff;\n"
    assert s.count(old) == 1, "R_DrawPlanes anchor"
    s = s.replace(old, old + "\tif (gpu_plane_log(pl, ds_source, planezlight, planeheight, lumpnum)) continue;   /* split mode: drawn by the doom_planes tasks, lump released by doom_post */\n", 1)
    s = s.replace("void R_DrawPlanes (void)\n{", "int gpu_plane_log(visplane_t* pl, byte* src, lighttable_t** zl, fixed_t ph, int lump);\nvoid R_DrawPlanes (void)\n{", 1)
    return s


patch("r_plane.c", r_plane)
