// Parallel pixel drawing for Doom on one GPU block. Thread 0 runs the game (logic, BSP walk, planes setup) and QUEUES column and span
// draws; at flush points all 256 threads of the block execute the queue. Commands for the same screen column stay in order (one thread
// owns each column), spans never overlap anything queued before them in the same flush, so the picture is identical to serial drawing.
typedef unsigned char byte;
#define SCREENWIDTH 320
#define MAXCOL 8192
#define MAXSPAN 8192
typedef struct { byte* dest; const byte* src; const byte* cm; int frac, step, count, next; } colcmd_t;
typedef struct { byte* dest; const byte* src; const byte* cm; unsigned position, step; int count, pad; } spancmd_t;
static colcmd_t g_col[MAXCOL];
static spancmd_t g_span[MAXSPAN];
static int g_head[SCREENWIDTH], g_tail[SCREENWIDTH], g_ncol, g_nspan;
volatile int gpu_par_active;
static volatile int g_exit;
// ONE shared barrier instruction: lane 0 (running the game) and its warp-mates (workers) must execute the identical bar.sync instruction or the warp never arrives
__attribute__((noinline)) void gpu_bar(void) { asm volatile("bar.sync 2, 256;" ::: "memory"); }
#define bar gpu_bar

static void do_col(const colcmd_t* c) {
  byte* d = c->dest; int f = c->frac, n = c->count;
  do { *d = c->cm[c->src[(f >> 16) & 127]]; d += SCREENWIDTH; f += c->step; } while (n--);
}
static void do_span(const spancmd_t* s) {
  byte* d = s->dest; unsigned position = s->position; int n = s->count;
  do { unsigned ytemp = (position >> 4) & 0x0fc0, xtemp = position >> 26; *d++ = s->cm[s->src[xtemp | ytemp]]; position += s->step; } while (n--);
}
static void work(int tid) {
  for (int x = tid; x < SCREENWIDTH; x += 256) { int i = g_head[x]; while (i >= 0) { do_col(&g_col[i]); i = g_col[i].next; } g_head[x] = -1; }
  for (int i = tid; i < g_nspan; i += 256) do_span(&g_span[i]);
}
void gpu_flush(void) {                                   // thread 0 only
  if (!gpu_par_active || (g_ncol == 0 && g_nspan == 0)) return;
  bar(); work(0); bar();
  g_ncol = 0; g_nspan = 0;
}
void gpu_q_col(byte* dest, const byte* src, const byte* cm, int frac, int step, int count, int x) {
  if (g_ncol == MAXCOL) gpu_flush();
  const int i = g_ncol++;
  g_col[i].dest = dest; g_col[i].src = src; g_col[i].cm = cm; g_col[i].frac = frac; g_col[i].step = step; g_col[i].count = count; g_col[i].next = -1;
  if (g_head[x] < 0) g_head[x] = i; else g_col[g_tail[x]].next = i;
  g_tail[x] = i;
}
void gpu_q_span(byte* dest, const byte* src, const byte* cm, unsigned position, unsigned step, int count) {
  if (g_nspan == MAXSPAN) gpu_flush();
  spancmd_t* s = &g_span[g_nspan++];
  s->dest = dest; s->src = src; s->cm = cm; s->position = position; s->step = step; s->count = count;
}
void gpu_par_init(void) { for (int x = 0; x < SCREENWIDTH; x++) g_head[x] = -1; g_ncol = g_nspan = 0; g_exit = 0; gpu_par_active = 0; }
void gpu_par_begin(void) { gpu_par_active = 1; }
void gpu_par_end(void) { gpu_flush(); gpu_par_active = 0; g_exit = 1; bar(); bar(); g_exit = 0; }   // second barrier lets workers read g_exit before it is cleared
void gpu_par_worker(int tid) { for (;;) { bar(); if (g_exit) { bar(); return; } work(tid); bar(); } }
