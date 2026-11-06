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
