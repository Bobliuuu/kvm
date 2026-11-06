# demo large (gpt_162m) on the A100: `demo/run_demo.sh large`

First run of the 162M GPT (12 layers, d768, 12 heads, seq 1024, vocab 50304, batch 4; 318,139 tasks and 6.9M edges per step) on the
executor, concurrently with serving requests on versioned weights and real Doom (700 frames), one persistent kernel, from `MANIFEST.json` only.

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, 108 SMs (grid 108: 1 admitter + 1 policy + 106 workers), driver 580.105.08, nvcc 12.8.93 |
| Build | `nvcc -O3 -arch=sm_80 -std=c++17 -rdc=true -DKVM_WITH_GAME --fmad=false -Xcompiler -ffp-contract=off -lineinfo` |
| Flags | gpt_162m, 6 steps, window 1, request every 2, 700 frames, tiling a100, 1xTF32 GEMMs, AdamW lr 3e-4 with 2 warmup steps, no CPU golden |
| Command | `KVM_HOST=<alias> KVM_KEY=<key> KVM_REMOTE=<private scratch dir> demo/run_demo.sh large --lock-wait 14400` (tag large-c) |
| Code | commit f6cd267 (branch test/demo-run) |

## What happened

1. It ran on the first try. No executor, op, memory or init-size problem exists at this size: 1,948,168 / 1,948,168 tasks, err_code 0, 53 admissions
   records, 22,935 version-carrying tasks all exact, no violations. Memory is fine (persistent 1.86 GiB + activation pool 4.58 GiB, plus Doom, on a 40 GB card);
   slots 2,097,152 hold all 6 steps (1.91M tasks) so the trace is complete. **A run longer than 6 steps recycles slots**: the version check and the
   cadence check then skip (a first 8-step attempt, `lr3e-4-8steps/`, 2.59M tasks, shows exactly that; its other numbers are valid).
2. The first run used the library default (AdamW lr 3e-3, no warmup) and the training check FAILED: losses 10.979, 10.940, 11.163, 10.680, 11.053, 11.223
   (`lr3e-3-cold-start/`). That is a hyper-parameter problem, not an executor error (version check and Doom unaffected): 3e-3 cold-start Adam is far too large
   for d768 x 12 layers. Fix: `--lr` / `--warmup` on `kvmfe.demo` and `run_demo.sh` (device-side warmup schedule), and the `large` preset uses 3e-4 / 2 steps
   (test: `tests/frontend/test_game_cli.py::test_demo_lr_and_warmup_reach_the_device_schedule`).
3. With lr 3e-4 the six losses are 10.979, 10.937, 10.876, 10.883, 10.892, 10.924. The check passes, but weakly: it compares the first with the last
   loss, 6 steps is far too short to see a bigram stream learned, and the curve is back to rising from step 3 (loss at init is ln(50304) = 10.83, observed 10.98).
   Six steps prove "runs, numerically sane, no NaN", not "learns". There is no CPU golden at this size (the Python reference VM cannot run it), so the
   gradients at scale are validated only indirectly (same ops as the verified gpt_tiny/gpt_small bundles); see the 40-step run below.

## Throughput (from `run.json`, `observations.jsonl`, `out/trace.csv` of the lr 3e-4 run)

| | |
|---|---|
| Step time (steps 1-5, 1xTF32, Doom finished) | 1.83, 1.82, 1.77, 1.86, 1.76 s: mean **1.81 s**, includes a 8,432-task serve request every second step (about 80 ms) |
| Tokens/s | 4096 tokens/step / 1.81 s = **2,260 tokens/s** |
| Task rate | about 180,000 tasks/s (318,140 tasks in one step) = 5.5 us per task |
| Workers busy | 9.1 of 106 on average during one step (96 CTA-s of work over the 6 steps: about 16 CTA-s/step; matmul tiles 48 us each) |
| Critical-path bound | 59.97 ms (cost model, `kvmfe.inspect --model gpt_162m --steps 1 --tiling a100`: 391 tasks, hop 5 us); the cost model's list schedule on 108 workers is 235 ms |
| Measured vs bound | **30x** the 59.97 ms critical path, 7.7x the list schedule |

Reading: the step is not compute bound (16 CTA-s of work would take about 150 ms on 106 workers) and not latency-of-the-critical-path bound (60 ms).
It is bound by task throughput: 318k tasks x 21.7 edges each through the admitter and the dependency tracker at about 5.5 us per task, while 91% of the workers
wait (queue wait p50 1.7 ms, p99 62 ms). Doom was unaffected: 172.7 frames/s against 35 needed, 0 of 698 tick gaps over 28.57 ms. The cure is fewer, bigger
tasks (128x128 / 256x128 matmul tiles are listed as "not built" in FRONTEND.md) or a cheaper per-task path, not a fix in this change.

## 40 steps (`lr3e-4-40steps/`, commit f6cd267)

`demo/run_demo.sh custom --model gpt_162m --steps 40 --window 1 --serve-every 10 --frames 700 --lr 3e-4 --warmup 2 --tf32 --no-golden`:
12,773,360 / 12,773,360 tasks in 68.5 s (1.63 s/step averaged over steps 1-39, 140 ring wraps, slots recycled so the version check and cadence are skipped by design),
Doom still 700/700 frames identical. Losses: 10.979, 10.937, 10.876, ..., 10.572 (step 29), 11.443 (step 32), ..., 10.592 (step 39); mean of the first 10 = 10.941,
mean of the last 10 = 10.891. So it **runs for 40 steps and stays finite, but the loss only drifts down inside +-0.3 step noise**. That is expected for this data
(a bigram t -> 5t+3 mod 50304: 4096 tokens per step touch about 8% of the vocabulary rows, so ~40 steps cannot learn the table) and does not demonstrate that
the 162M gradients are right. A real convergence/correctness claim at this size needs an independent reference (PyTorch fp32 with the same init, for the first
few steps) or a stream with a much smaller effective vocabulary; neither exists yet.
