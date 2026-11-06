// Parallel pixel drawing for Doom on one GPU block. Thread 0 runs the game (logic, BSP walk, planes setup) and QUEUES column and span
// draws; at flush points all 256 threads of the block execute the queue. Commands for the same screen column stay in order (one thread
// owns each column), spans never overlap anything queued before them in the same flush, so the picture is identical to serial drawing.
//
// SPLIT MODE (g_split, set by gpu_doom_sim): the same queue is kept for the whole view (a LOG) and executed later, by gpu_draw_tile() in
// separate scheduler tasks (one per band of screen columns). The log is replayed per column band in recording order, epoch by epoch
// (an epoch ends where the classic mode flushes: after the planes and after the masked pass), so the picture is identical to the
// classic path. Anything that must read the screen it draws on (fuzz, translated columns) or an overfull log drains the log serially
// on thread 0 (gpu_log_drain), which is the classic order by construction.
typedef unsigned char byte;
#define SCREENWIDTH 320
#define MAXCOL 16384
#define MAXSPAN 16384
typedef struct { byte* dest; const byte* src; const byte* cm; int frac, step, count, next, epoch; } colcmd_t;
typedef struct { byte* dest; const byte* src; const byte* cm; unsigned position, step; int count, x, epoch; } spancmd_t;
static colcmd_t g_col[MAXCOL];
static spancmd_t g_span[MAXSPAN];
static int g_head[SCREENWIDTH], g_tail[SCREENWIDTH], g_ncol, g_nspan;
volatile int gpu_par_active;
static volatile int g_exit;
static int g_split;          // 1: gpu_doom_sim is running a tick: draws go to the log
static int g_epoch;          // current epoch of the log
static int g_ne;             // number of epochs that hold commands
// ONE shared barrier instruction: lane 0 (running the game) and its warp-mates (workers) must execute the identical bar.sync instruction or the warp never arrives
__attribute__((noinline)) void gpu_bar(void) { asm volatile("bar.sync 2, 256;" ::: "memory"); }
#define bar gpu_bar
// whole-CTA barrier for the draw tasks (every thread of the block calls gpu_draw_tile)
__attribute__((noinline)) static void gpu_bar_all(void) { asm volatile("bar.sync 0;" ::: "memory"); }
#ifdef DOOM_PROF
unsigned long long doom_prof[32]; static unsigned long long prof_last;
static unsigned long long clk(void) { unsigned long long t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); return t; }
void gpu_prof(int id) { unsigned long long t = clk(); if (id == 0) { prof_last = t; return; } doom_prof[id] += t - prof_last; prof_last = t; }
#endif

static void do_col(const colcmd_t* c) {
  byte* d = c->dest; int f = c->frac, n = c->count;
  do { *d = c->cm[c->src[(f >> 16) & 127]]; d += SCREENWIDTH; f += c->step; } while (n--);
}
static void do_span(const spancmd_t* s) {
  byte* d = s->dest; unsigned position = s->position; int n = s->count;
  do { unsigned ytemp = (position >> 4) & 0x0fc0, xtemp = position >> 26; *d++ = s->cm[s->src[xtemp | ytemp]]; position += s->step; } while (n--);
}
// the pixels [lo, hi] (screen columns) of span s; exact because position is linear in the pixel index
static void do_span_clip(const spancmd_t* s, int x0, int x1) {
  int lo = s->x > x0 ? s->x : x0, hi = s->x + s->count < x1 - 1 ? s->x + s->count : x1 - 1;
  if (lo > hi) return;
  const int k = lo - s->x;
  byte* d = s->dest + k; unsigned position = s->position + (unsigned)k * s->step; int n = hi - lo;
  do { unsigned ytemp = (position >> 4) & 0x0fc0, xtemp = position >> 26; *d++ = s->cm[s->src[xtemp | ytemp]]; position += s->step; } while (n--);
}
static void work(int tid) {
  for (int x = tid; x < SCREENWIDTH; x += 256) { int i = g_head[x]; while (i >= 0) { do_col(&g_col[i]); i = g_col[i].next; } g_head[x] = -1; }
  for (int i = tid; i < g_nspan; i += 256) do_span(&g_span[i]);
}
static void log_reset(void) { for (int x = 0; x < SCREENWIDTH; x++) g_head[x] = -1; g_ncol = g_nspan = 0; g_epoch = 0; g_ne = 0; }
// thread 0: execute the whole log serially, in recording order, and empty it
static void log_drain(void) {
  for (int e = 0; e < g_ne; e++) {
    for (int x = 0; x < SCREENWIDTH; x++) for (int i = g_head[x]; i >= 0; i = g_col[i].next) if (g_col[i].epoch == e) do_col(&g_col[i]);
    for (int i = 0; i < g_nspan; i++) if (g_span[i].epoch == e) do_span(&g_span[i]);
  }
  log_reset();
}
void gpu_flush(void) {                                   // thread 0 only
  if (!gpu_par_active) return;
  if (g_split) { log_drain(); return; }                  // fuzz / translated columns read the screen: everything logged so far must be on it first
  if (g_ncol == 0 && g_nspan == 0) return;
#ifdef DOOM_PROF
  unsigned long long t0 = clk(); doom_prof[8] += g_ncol; doom_prof[9] += g_nspan; doom_prof[10]++;
#endif
  bar(); work(0); bar();
#ifdef DOOM_PROF
  doom_prof[7] += clk() - t0;
#endif
  g_ncol = 0; g_nspan = 0;
}
// the two flush points of R_RenderPlayerView: classic = flush, split = start a new epoch
void gpu_epoch(void) {
  if (!g_split) { gpu_flush(); return; }
  if (gpu_par_active && (g_ncol || g_nspan)) g_epoch++;
}
void gpu_q_col(byte* dest, const byte* src, const byte* cm, int frac, int step, int count, int x) {
  if (g_ncol == MAXCOL) gpu_flush();
  const int i = g_ncol++;
  g_col[i].dest = dest; g_col[i].src = src; g_col[i].cm = cm; g_col[i].frac = frac; g_col[i].step = step; g_col[i].count = count; g_col[i].next = -1; g_col[i].epoch = g_epoch; g_ne = g_epoch + 1;
  if (g_head[x] < 0) g_head[x] = i; else g_col[g_tail[x]].next = i;
  g_tail[x] = i;
}
void gpu_q_span(byte* dest, const byte* src, const byte* cm, unsigned position, unsigned step, int count, int x) {
  if (g_nspan == MAXSPAN) gpu_flush();
  spancmd_t* s = &g_span[g_nspan++];
  s->dest = dest; s->src = src; s->cm = cm; s->position = position; s->step = step; s->count = count; s->x = x; s->epoch = g_epoch; g_ne = g_epoch + 1;
}
void gpu_par_init(void) { log_reset(); g_exit = 0; gpu_par_active = 0; g_split = 0; }
void gpu_par_begin(void) { gpu_par_active = 1; }
void gpu_par_end(void) { gpu_flush(); gpu_par_active = 0; g_exit = 1; bar(); bar(); g_exit = 0; }   // second barrier lets workers read g_exit before it is cleared
void gpu_par_worker(int tid) { for (;;) { bar(); if (g_exit) { bar(); return; } work(tid); bar(); } }

// ---- split mode (thread 0 for sim/post; every thread of the block for the tile)
static int g_pending;        // a view is logged and not yet fully drawn + completed (set by view_end, cleared by post)
static int g_log_tic;        // tic number of the logged view
void gpu_split_begin(void) { g_split = 1; }
void gpu_split_end(void) { g_split = 0; }
// D_Display, just before R_RenderPlayerView: start logging (a screen wipe in the same tick keeps the classic direct path)
void gpu_view_begin(int wipe) { if (g_split && !wipe) { log_reset(); gpu_par_active = 1; } }
// just after it: returns 1 when the tick's remaining display work is to be deferred to the post task (after the tiles)
int gpu_view_end(int tic) {
  if (!(g_split && gpu_par_active)) return 0;
  gpu_par_active = 0; g_pending = 1; g_log_tic = tic;
  return 1;
}
int gpu_split_pending(void) { return g_pending; }
int gpu_split_log_tic(void) { return g_log_tic; }
void gpu_split_done(void) { g_pending = 0; }
static int g_resume;
int gpu_split_resume(void) { const int r = g_resume; g_resume = 0; return r; }
void gpu_post_arm(void) { g_resume = 1; }
int gpu_log_stats(int* ncol, int* nspan) { *ncol = g_ncol; *nspan = g_nspan; return g_ne; }

// Block-cooperative: draw the log's columns [x0, x1) epoch by epoch. Reads only the log; writes only screen pixels of that band.
void gpu_draw_tile(int tile, int ntiles, int tid, int nthreads) {
  const int x0 = tile * SCREENWIDTH / ntiles, x1 = (tile + 1) * SCREENWIDTH / ntiles;
  const int ne = g_ne, nspan = g_nspan;
  for (int e = 0; e < ne; e++) {
    for (int x = x0 + tid; x < x1; x += nthreads) {
      for (int i = g_head[x]; i >= 0 && g_col[i].epoch <= e; i = g_col[i].next) if (g_col[i].epoch == e) do_col(&g_col[i]);
    }
    for (int i = tid; i < nspan; i += nthreads) if (g_span[i].epoch == e) do_span_clip(&g_span[i], x0, x1);
    if (e + 1 < ne) gpu_bar_all();
  }
}
