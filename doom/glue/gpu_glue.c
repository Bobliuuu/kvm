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
// Thread 0: leave the tic protocol without running a tic (invalid call), so the workers still return.
void gpu_doom_abort_tic(void) { gpu_par_begin(); gpu_par_end(); }
