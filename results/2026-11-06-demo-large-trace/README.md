# demo large (gpt_162m) with the full trace, on the A100: `demo/run_demo.sh large`

The 162M-parameter GPT (12 layers, d768, 12 heads, seq 1024, batch 4), serving requests on its versioned weights, and real Doom (700 frames), all in one
persistent kernel from `MANIFEST.json`, re-run to keep the complete per-task trace (the earlier `2026-11-06-demo-large` committed none). All checks pass; the weak ones are listed.

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, 108 SMs (grid 108: 1 admitter + 1 policy + 106 workers), driver 580.105.08, nvcc 12.8.93 |
| Build | `nvcc -O3 -arch=sm_80 -std=c++17 -rdc=true -DKVM_WITH_GAME --fmad=false -Xcompiler -ffp-contract=off -lineinfo` |
| Flags | preset `large`: gpt_162m, 6 steps, window 1, request every 2 steps, 700 frames, tiling a100, 1xTF32 GEMMs, AdamW lr 3e-4 with 2 warmup steps, no CPU golden, sched dynamic |
| Command | `KVM_HOST=<alias> KVM_KEY=<key> KVM_REMOTE=<private scratch dir> demo/run_demo.sh large --tag large-trace --lock-wait 14400` |
| Code | commit d657dde (branch docs/readme-visuals; the demo code is that of `test/demo-run`) |

| Check (`check.log`, `report.json`) | Result |
|---|---|
| executor | PASS: 1,948,168 / 1,948,168 tasks in 13.66 s, err_code 0, 2,097,152 slots (no recycling, so the trace is complete) |
| version coherence | PASS: 22,935 version-carrying tasks, all exact, 0 violations |
| training | PASS but weak: 6 losses 10.979, 10.937, 10.876, 10.883, 10.892, 10.924 (first 10.979 -> last 10.924). No golden at this size; six steps cannot show learning (see `2026-11-06-demo-large`; step-1 numerics against float64 are in `2026-11-06-gpt162-vs-torch`) |
| serving | SKIP: no golden decode (`--no-golden`); coherence is covered by the version check |
| Doom bytes vs CPU | PASS: 700/700 frames raw-, fingerprint- and state-identical |
| 35 Hz cadence | PASS: 172.7 frames/s, tick gap p50/p99/max 5.58/11.32/12.87 ms, 0 of 698 gaps over 28.57 ms (7 early ticks late by construction: boot tick 175 ms) |

Overlap (from `check.log`): Doom ran 4.22 s; train 1,875,696 tasks (95.9 CTA-s, 9.3 of them while Doom ran), serve 33,488 tasks, game 4,900 tasks.

What the trace shows (`assets/timeline-162m-*.png`, `assets/demo-162m.mp4`): Doom frames fill the first 1.1 s alone, the first dense training burst starts at about 1.1 s, and step 1
finishes at 4.68 s (it overlaps Doom); steps 2-6 then complete every 1.73-1.82 s (1.8 s/step, as in
`2026-11-06-demo-large`). Between bursts of dense training the workers are idle: the step is admission/dependency bound (318k tasks per step), not compute bound.
Against PyTorch the same model is 15-18x slower, see `2026-11-06-gpt162-vs-torch`.

## Files

- `run.json`, `report.json`, `check.log`, `observations.jsonl`: as written by the run.
- `trace_compact.npz` (6.9 MB): every task of the 229 MB `trace.csv`, reduced by `demo/viz/compact_trace.py` to start_us, end_us (float32), smid, op class and
  Doom frame number, sorted by start time. Lossless for the figures; the full CSV stays out of git. `python3 demo/viz/timeline.py --trace results/2026-11-06-demo-large-trace/trace_compact.npz --run-json results/2026-11-06-demo-large-trace/run.json --prefix timeline-162m`
- The Doom frames of the video (`doom_candidate/frames.bin`, 45 MB, device output) are not committed; `assets/demo-162m.mp4` is.
