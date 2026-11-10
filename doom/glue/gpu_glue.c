// GPU-only glue: WAD in device memory, engine boot on the game thread, and one tic with the other 255 threads drawing.
#include <stdint.h>
extern const unsigned char* tl_wad; extern long tl_wad_len;
void tl_heap_reset(void);
void gpu_init_tables(void); void gpu_init_game_ptrs(void);
extern volatile int gpu_par_active; void gpu_par_init(void); void gpu_par_begin(void); void gpu_par_end(void); void gpu_par_worker(int tid);
void doom_create(const char* wad); void doom_tic(uint32_t input);

// Thread 0 of a 256-thread block. Resets the engine's heap, so it also restarts a finished game.
void gpu_doom_boot(const unsigned char* wad, long wad_len) {
  tl_heap_reset(); tl_wad = wad; tl_wad_len = wad_len;
  gpu_init_tables(); gpu_init_game_ptrs(); gpu_par_init();
  doom_create("doom1.wad");
}
// Thread 0: one game tic; threads 1..255 must be inside gpu_doom_worker() for the whole tic.
#ifdef DOOM_PROF
void gpu_prof(int id); extern unsigned long long doom_prof[32];
void gpu_doom_tic(uint32_t input) { gpu_prof(0); gpu_par_begin(); doom_tic(input); gpu_par_end(); gpu_prof(6);
  static int first = 1; if (first) { first = 0; for (int i = 0; i < 32; i++) doom_prof[i] = 0; } else doom_prof[11]++; }
#else
void gpu_doom_tic(uint32_t input) { gpu_par_begin(); doom_tic(input); gpu_par_end(); }
#endif
void gpu_doom_worker(int tid) { gpu_par_worker(tid); }
// ---- split mode: sim (thread 0) logs the view instead of drawing it, the tile tasks draw it, post (thread 0) finishes the display work.
// A tick whose display does not go through the logged view (screen wipe, intermission, finale) is complete after sim: pending stays 0.
void gpu_split_begin(void); void gpu_split_end(void); int gpu_split_pending(void); int gpu_split_log_tic(void); void gpu_split_done(void); void gpu_post_arm(void);
void gpu_draw_tile(int tile, int ntiles, int tid, int nthreads);
void D_Display(void);
void gpu_split_set_tic(int t); void doom_state(int32_t* o);
void gpu_doom_sim(uint32_t input) {
  gpu_split_begin(); doom_tic(input); gpu_split_end();
  if (gpu_split_pending()) { int32_t st[16]; doom_state(st); gpu_split_set_tic(st[1]); }   // tic count of the logged frame (the glue's, as in the state buffer)
}
int gpu_doom_pending(void) { return gpu_split_pending(); }
int gpu_doom_log_tic(void) { return gpu_split_log_tic(); }
void gpu_doom_draw(int tile, int ntiles, int tid, int nthreads) { gpu_draw_tile(tile, ntiles, tid, nthreads); }
void gpu_draw_planes(int tile, int ntiles, int tid, int nthreads); int gpu_stages_ok(void); void gpu_planes_release(void);
void gpu_doom_planes(int tile, int ntiles, int tid, int nthreads) { gpu_draw_planes(tile, ntiles, tid, nthreads); }
int gpu_doom_post(void) {                  // returns 1 when a stage of the logged view never ran
  if (!gpu_split_pending()) return 0;
  const int ok = gpu_stages_ok();
  gpu_planes_release(); gpu_post_arm(); D_Display(); gpu_split_done();
  return !ok;
}
// Thread 0: leave the tic protocol without running a tic (invalid call), so the workers still return.
void gpu_doom_abort_tic(void) { gpu_par_begin(); gpu_par_end(); }
