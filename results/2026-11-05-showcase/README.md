# Showcase run data (2026-11-05): raw outputs behind every chart in `demo/`

What this is: the executor's own outputs from one demo run (`demo/run_demo.sh tiny`), kept so `demo/viz/make_all.sh` regenerates
`demo/assets/*` and `demo/index.html` on a machine without a GPU. Nothing in the charts is typed in or simulated, except `scaleout.svg`
(a chart of `results/2026-11-04-scaleout-sim/tables.md`, a simulation) and `scoreboard.svg` (numbers copied from the README results table and
the claims audit, cited on the figure).

| Directory | Scheduler | Source run |
|---|---|---|
| `dynamic/` | `--sched dynamic` (classes share all SMs by weight) | demo run `t3`, tree commit 89ff881 (earlier than the integration tip), A100-SXM4-40GB (108 SMs, grid 108), CUDA 12.8, driver 580.105.08 |

Each directory: `trace.csv.gz` (one row per task: `%smid`, start/end `%globaltimer` in us, queue wait, weight versions read/published, class),
`run.json`, `report.json` (the checks of `demo/check_demo.py`), `observations.jsonl` (loss and served tokens as each admission finished).

Command: `KVM_HOST=... demo/run_demo.sh tiny --tag t3 --sanitize` with `--steps 200 --window 4 --serve-every 20 --frames 700` (preset tiny, those two flags
overridden), 1xTF32-exact (3xTF32) GEMMs. 27,003 tasks in 4231 ms.

## What the run showed (report.json)

- executor 27003/27003 tasks, err_code 0; versions: 460 version-carrying tasks, 460 exact, 0 violations
- training: 200 step losses 4.146 -> 0.0042; max relative deviation from the CPU golden 3.78e-06 (tolerance 2e-03)
- serving: 44/44 served token buffers equal the golden decode of the same weights version (11 requests)
- game: 700 frames raw-identical, fingerprint-identical and state-identical to the CPU oracle
- cadence: 174 ticks/s (Doom is not paced in this demo), tick gap p50/p99/max 5.67/10.08/10.72 ms, 0 of 698 gaps over 28.57 ms
- overlap: train 20,200 tasks (19,796 overlapping Doom), serve 803 tasks, game 4,900 tasks

## Caveats (do not skip)

- In the same demo invocation the compute-sanitizer stage on a small demo (4 steps, 8 frames): memcheck and racecheck passed (0 errors, 0 hazards);
  **synccheck did not**: exit 9, "process didn't terminate successfully", 32 errors (the run aborted with `terminate called without an active exception`).
  Not root-caused here. The main run above is not affected (it ran without sanitizer) but synccheck is not clean on the demo runner.
- The GPU is mostly idle: Doom's tick is one CTA, training tasks are about 10 us. Busy SMs average a few of 108.
- The shared A100 box ran other jobs around this run; absolute times are indicative.
- Only the dynamic scheduler has data here. A static-split and a global-queue run were queued with `demo/viz/collect.sh` (the binary was built on the box) but the shared GPU lock
  was never granted within the 2 h wait (about 15 other jobs queued), so there is no static or global data and no dynamic-vs-static figure; the charts
  say so by omission and `make_charts.py` draws only the modes it has. The Doom frames (45 MB) are not committed; `demo/assets/doom_demo.mp4` was rendered from them by `demo/viz/make_video.py`.
