# Doom as a parallel task graph (audit gap 6) - A100-SXM4-40GB, shared Lambda box, 2026-11-06

Status: in progress, sections are added as each stage is measured. Raw per-run JSON lines (`verify.jsonl` etc.): one `RESULT` per run with metrics, the
executor-trace phase breakdown and the comparison against the CPU oracle (raw bytes, fingerprint and game state of every frame).

## Stage 0: where the serial tick's 5.4 ms goes (`classic-phase-profile.jsonl`, `DOOM_PROF=1` build, clock64 on the game thread, 699 ticks)
| phase (per tick, us) | |
|---|---|
| G_BuildTiccmd + P_Ticker (simulation) + sounds | 275 |
| ST_Drawer / HU_Erase | 292 |
| R_SetupFrame + BSP walk + wall setup (column commands queued) | 2873 |
| R_DrawPlanes (R_MakeSpans / R_MapPlane math, span commands queued) | 1547 |
| R_DrawMasked (sprites) | 333 |
| HU_Drawer + rest | 36 |
| of which the 256-thread draw (flush: 753 column + 266 span commands, 2 flushes) | 147 |

So the existing in-CTA parallel draw already takes only 2.7% of the tick; 97% is serial work on the game thread. Splitting the pixel drawing over more
CTAs can at best recover a fraction of that 147 us (stage 1); the parallelisable serial work is the plane setup (28%, independent per plane, stage 2).
The BSP walk (52%) is a front-to-back occlusion walk with a shared clip list and was NOT parallelised.

## Stage 1: split tick, draw tiles (sim -> T draw tiles -> post), `stage1-draw-tiles/verify.jsonl`
Frame-exact against the CPU oracle for T = 1, 2, 4, 5, 8, 16, 32, 64 (300 frames) and T = 8 (700 frames), alone and beside the GEMM+AdamW tenant under the dynamic
and the global scheduler (3 chains): every compared frame raw-byte, fingerprint and game-state identical.
Result: no speed-up. Executor trace, median over frames: sim 5.8 ms, draw stage wall 142 us (T=1) -> 68 us (T=8) -> 67 us (T=16), post 0.12 ms; fps 169-172 for every T against 170 serial
(300 frames incl. the boot tick; 700-frame runs: 182.1 split T=8, 182.0 serial). Tile count above 8 does not help (the tile task is dispatch-bound, ~11 us per hop).
