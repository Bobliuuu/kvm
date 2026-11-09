# Step overlap on the executor: window 40 (one admission, 40 steps) vs window 1 (40 admissions), A100

Goal 5a. gpt_tiny, 40 AdamW steps, exact 3xTF32 (`--precise`), tiling a100, one serve request, no Doom (`--game-frames 0`), `demo/kvm_run` (built Nov 6 from
the integrated tree, before this branch's merges), one `~/.gpu.lock` hold per rep, 3 reps interleaved, A100-SXM4-40GB. Manifests: `manifest_w1.json`, `manifest_w40.json`
(generated with `python3 -m kvmfe.demo --all --train-steps 40 --window W --serve-every 1000 --serve-policy exact --game-frames 0 --game-fb-slots 0 --tiling a100 --max-bundles 2 --precise`).
Both runs: 4193 tasks done, err_code 0, `ok` true (`run_*.json`). No loss check was run (no CPU golden here); the claim is timing only.

| window | admissions | kernel_ms (3 reps) | last-admission done, us |
|---|---|---|---|
| 1 (a step ends, the next is admitted from the host; no cross-step edges in the DAG) | 41 | 86.9, 85.5, 83.9 | 86704, 85377, 83773 |
| 40 (steps are one task graph; step s+1's tiles start as soon as their inputs from step s are ready) | 2 | 60.4, 60.7, 60.3 | 60261, 60561, 60200 |

Window 40 is 1.41x faster (85.4 -> 60.5 ms, 2.1 ms/step -> 1.5 ms/step). **Confound, stated plainly:** window 1 differs from window 40 in two ways at once: (i) the
per-window barrier (no step overlap) and (ii) one host admission round trip per step (about 0.3-0.5 ms between consecutive train admissions in `run_1_*.json` launch times).
This run cannot separate them, so it shows "fusing steps into one task graph beats step-per-admission", not "overlap alone is worth 1.41x". The kernel-level
per-column overlap experiment (`2026-11-07-a100-coopt-sh/pbt-overlap`) isolates overlap inside `kvm.cu` and found none (0.8-1.0x). A fair conventional comparison
is still a CUDA graph of 40 steps, which the executor loses to at this size (`2026-11-06-vs-pytorch`).
