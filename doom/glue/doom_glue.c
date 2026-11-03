// Engine glue shared by the GPU and CPU builds, so both play the same game tic for tic: a virtual clock that advances exactly one 35 Hz
// tic per doom_tic() call, an input word per tic (the scripted player when absent), and a state snapshot for checks.
#include <stdint.h>
#include <string.h>
#include "doomgeneric.h"
#include "doomstat.h"
#include "d_player.h"
#include "frame_hash.h"

#define DOOM_INPUT_SCRIPTED 0xFFFFFFFFu
struct color { uint32_t b:8; uint32_t g:8; uint32_t r:8; uint32_t a:8; };      // layout of i_video.c's palette entry
extern struct color colors[256];
extern uint8_t* I_VideoBuffer;
int player_waypoint(void);

uint32_t doom_input_word = DOOM_INPUT_SCRIPTED;     // read by player.c
static uint64_t g_tic, g_slept_ms;
static int g_deaths, g_was_dead, g_finished_tic = -1;

void DG_Init(void) {}
void DG_DrawFrame(void) {}
void DG_SetWindowTitle(const char* t) { (void)t; }
int DG_GetKey(int* pressed, unsigned char* key) { (void)pressed; (void)key; return 0; }
void DG_SleepMs(uint32_t ms) { g_slept_ms += ms; }                         // the engine sleeps at start-up and level changes: let the time pass
uint32_t DG_GetTicksMs(void) { return (uint32_t)((g_tic * 1000 + 34) / 35 + g_slept_ms); }

void doom_create(const char* wad) {
  static char* argv[] = {"doom", "-iwad", 0, "-warp", "1", "1", "-skill", "3", "-nomusic", "-nosfx", "-nosound", 0};
  argv[2] = (char*)wad;                         // the engine keeps argv: static storage
  g_tic = g_slept_ms = 0; g_deaths = g_was_dead = 0; g_finished_tic = -1; doom_input_word = DOOM_INPUT_SCRIPTED;
  doomgeneric_Create(11, argv);
}

void doom_tic(uint32_t input) {
  g_tic++;
  doom_input_word = input;
  doomgeneric_Tick();
  const player_t* p = &players[consoleplayer];
  const int dead = p->playerstate == PST_DEAD;
  if (dead && !g_was_dead) g_deaths++;
  g_was_dead = dead;
  if (gamestate == GS_INTERMISSION && g_finished_tic < 0) g_finished_tic = (int)g_tic;
}

// Layout = the header of the doom_tick state buffer (include/kvm_ops_game.cuh KVM_DOOM_ST_*).
void doom_state(int32_t* o) {
  const player_t* p = &players[consoleplayer];
  o[0] = 1; o[1] = (int32_t)g_tic; o[2] = (int32_t)gamestate; o[3] = p->health; o[4] = p->armorpoints; o[5] = p->killcount;
  o[6] = totalkills; o[7] = p->itemcount; o[8] = p->secretcount; o[9] = g_deaths; o[10] = g_finished_tic;
  o[11] = p->mo ? p->mo->x >> 16 : 0; o[12] = p->mo ? p->mo->y >> 16 : 0; o[13] = player_waypoint();
}

void doom_frame_hash(uint32_t* out) {
  uint32_t s = 0, x = 0;
  for (int i = 0; i < DOOM_PIXELS; i++) { const uint32_t t = dh_term(i, I_VideoBuffer[i]); s += t; x ^= t; }
  for (int i = 0; i < 256; i++) { uint32_t w; memcpy(&w, &colors[i], 4); const uint32_t t = dh_term(DOOM_PIXELS + i, w); s += t; x ^= t; }
  out[0] = s; out[1] = x;
}
// raw frame for dumps: 64000 palette indices, then 256 palette words
void doom_frame_raw(uint8_t* out) { memcpy(out, I_VideoBuffer, DOOM_PIXELS); memcpy(out + DOOM_PIXELS, colors, 1024); }
