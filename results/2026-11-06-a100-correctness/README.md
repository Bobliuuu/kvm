# Correctness and coverage gaps closed (A100, 2026-11-06/07)

Four gaps that had no owner: (a) the 162M demo's serving path had no golden decode, (b) three bf16 model-zoo cases failed the weight check, (c) "dynamic shapes / data-dependent
control flow" was only a line in "Not built", (d) goal 2 ("the loss within a stated tolerance of an independent reference, CPU fp32 or PyTorch") was tied to the repo's own CPU golden
for most model families, not to PyTorch. Branch `fix/correctness-gaps` (from `origin/main` a47a5f5; `origin/perf/bf16-a100` was **not** merged, see (b)).

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, 108 SMs, sm_80, driver 580.105.08, CUDA 12.8 (nvcc 12.8.93); shared box, every GPU job inside one `flock ~/.gpu.lock` hold |
| PyTorch (box) | 2.7.0 (CPU float64 for the serving golden; the zoo comparison needs no GPU torch) |
| Executors used | (a) `demo/kvm_run` as built for `2026-11-06-demo-large-trace` (unchanged sources: only `demo/*.py`, `demo/remote_demo.sh` changed); (d) the `kvm_exec` of the bf16 branch tree (a superset of main: f32 paths regression-tested there, `1396 / 3669` ops checks), because it was already built; (b) the same binary and the bf16 branch's own zoo bundles |
| Commits | see `git log origin/main..fix/correctness-gaps`; the numbers below come from scripts at the tip named in the last line of this file |

## (a) Serving at 162M now has a golden decode

`demo/serve_golden.py` recomputes a serve request in PyTorch float64 without a KV cache and without any kvmfe op (pre-LN, RoPE, causal SDPA, tanh-GELU, untied head). `demo/check_demo.py --no-golden --model gpt_162m`
(the new default of `remote_demo.sh` when the CPU golden is off) runs it on every request whose weights are known:

* **version 0**: the executor's initial values are rebuilt from the config alone (counter-based generator, self-checked against `runtime/initgen.py`);
* **final version**: the weights are read from `kvm_run --dump '^w\.'` (`remote_demo.sh` now adds it when the CPU golden is off), so the decode is checked against the weights the executor actually holds;
* **middle versions** (`serve_s2`, `serve_s4`): not checked and reported as such. No snapshot of them survives (the next window overwrites the weights) and a CPU re-training cannot reproduce them (AdamW amplifies 1xTF32 noise, `2026-11-04-a100-gpt-golden`).

The comparison is teacher-forced per token (the reference is fed the prompt plus the executor's own earlier tokens, one causal forward gives every step's logits). A token passes if it is the float64 argmax,
or if its float64 logit gap to the argmax is within `2 x 2 x E`, where `E` is the largest logit deviation (over the whole vocabulary, at that position) of the same model with every matmul operand rounded to TF32
(float32 accumulate): the arithmetic class of the executor's default tiles. With `--precise` (3xTF32 tiles) no slack is allowed.

Run (`serving-162m/`, same A100, model gpt_162m, 6 steps, window 1, a request every 2 steps, tiling a100, 1xTF32, lr 3e-4 + 2 warmup, **no Doom**: the Doom assets were not on the box, and serving does not need them):

| check | result |
|---|---|
| executor | PASS: 1,942,568 / 1,942,568 tasks, 11.2 s, err_code 0 |
| versions | PASS: 22,935 version-carrying tasks, all exact |
| training | PASS (weak, as in `2026-11-06-demo-large-trace`): 10.979 -> 10.924 |
| **serving** | **PASS: 2 of 4 requests checked.** `serve_s0` (version 0) tokens `[41803, 33285, 42236]` = float64 argmax, min top1-top2 margin 0.020 against a tie tolerance 0.0086; `serve_s6` (final weights) `[34196, 34196, 34196]` = float64 argmax, margin 0.042 against 0.003. `serve_s2`, `serve_s4`: unchecked (no snapshot) |
| negative control (`neg.log`) | FAIL as it must: the observed tokens of another version fed to the same check give logit gaps of 2.0-2.6 against tolerances of 0.003-0.009 |

The `serve_s0` tokens are also identical to those of the earlier full run with Doom (`2026-11-06-demo-large-trace/observations.jsonl`, 41803 33285 42236; `serve_s6` 34196 x3 there too), which the new check now certifies.
Reproduce the reference alone on a laptop (10 s, 2 GB): `python3 -I demo/serve_golden.py --model gpt_162m --prompt 1 8 43 --tokens 41803 33285 42236`.
Tests: `tests/frontend/test_serve_golden.py` (torch decode equals the independent numpy oracle for untrained and trained weights; wrong tokens, a wrong version and a near tie at 3xTF32 are caught).

## (b) bf16 zoo failures: the check was wrong, not the model

**Status: code and CPU tests done; the A100 re-run of the three failing cases is NOT yet recorded** (the lab GPU lock had ~18 waiters; the eight dump jobs are queued and the re-run is one command, below).

Root causes (from `2026-11-06-a100-bf16` `check.txt`):
* `vit_bf16` / `vit_bf16s`: the key-projection bias of attention without RoPE has an exactly zero true gradient (softmax is invariant to a per-query constant). Under bf16 the golden holds rounding noise of ~3e-8 there, above the old
  magnitude cut-off (`NULL_ABS` 1e-8 on the golden norm), so only `v.*` was skipped and `w.*`/`m.*` were compared relative to the golden: AdamW's normalised step turned noise into +-lr moves (rel L2 0.90-0.98). The emulated baseline fails identically, so it is the check.
* `san_bf16s`: weights 1.7e-2 vs 1e-2 on 2 steps with real bf16 storage. Not a zero-gradient case; **not fixed here** (no tolerance was touched): needs the dump to decide between a real defect and AdamW flip amplification.

Fix of the check, not by loosening: the model now declares the parameter (`ParamSpec.invariant`, set on ViT's key bias; `tests/frontend/test_models.py` asserts the flag equals "autodiff gradient is zero" for every preset, so it is complete and true),
`gen_bundle.py` writes `invariant_params` and the optimizer to `meta.json`, and `examples/models/check.py` judges those parameters in absolute terms instead of relative to the golden: the weight must stay in AdamW's travel envelope
(|dw| <= lr x max(1, (1-b1)/sqrt(1-b2)) per step, so |w_gpu - w_golden| <= 2T, |w_gpu| <= T with T = steps x lr), the moments at the noise level (<= 2^-6 of the live gradients' scale). Live parameters keep the 1e-2 bound.
`examples/models/test_check.py` (8 tests, generates a real vit_bf16 bundle): sign-flipped noise inside the envelope passes; a weight outside it, moments at gradient scale, and a wrong live weight each fail.
Re-run on the A100 (bf16 bundles/binary from `origin/perf/bf16-a100`, whose `meta.json` need `invariant_params`; this branch does not merge it):
`python3 examples/models/check.py WORKLOAD BUNDLE DUMP --mode lowp` for `vit_bf16`, `vit_bf16s`, `san_bf16s` (dump: `kvm_exec --quiet --dump-buffers BUNDLE`).

## (c) Dynamic shapes and data-dependent control flow

Design note: `FRAMEWORK.md`, section "Dynamic shapes and data-dependent control flow" (scope, three features ranked by size, what is not built). Prototype of the smallest honest one, **admission guards**
(manifest_version 2): an admission runs only if a scalar of a buffer passes a comparison; a `while` with a data-dependent trip count is a chain of guarded admissions, early exit is the next chunk's guard reading a `done` flag.
Built in the CPU reference runner (`frontend/kvmfe/run_manifest.py`: validation, guard evaluation, skip cascade through `after_bundles`, "a guarded admission cannot publish" so the version flow stays static) and tested
(`tests/frontend/test_guard.py`: trip counts 0, 1, 2, 4 and the unroll bound for five initial values; cascade; publishing and malformed guards rejected). It touches neither the admitter's ingest path nor `kvm_claim`,
and `demo/kvm_run.cu` is **unchanged**: it accepts only `manifest_version` 1, so a guarded manifest stops there with an error instead of running every admission unconditionally. Not built: guards in `kvm_run`
(needs a 15-minute device build and an A100 test; the change is small, described in the note), masked tiles for variable sequence length (padding with `ignore = -1` is already correct, only the padded tiles' FLOPs are wasted),
and device-side conditional admission (belongs to the admission rework).

## (d) Every zoo family tied to PyTorch

`examples/models/torch_models.py` is a float64 PyTorch (autograd) implementation of GPT, Llama / GQA / tied head, MoE (top-k, capacity, load-balance loss as a second backward root), MLP and ViT, with the bf16 rounding emulation.
It shares only the initial values and the first batch with the frontend. Two levels:

1. **CPU, exact** (`cpu_vm_vs_torch.txt`, also `tests/frontend/test_torch_models.py`, 11 tests): the frontend's program on the CPU float64 VM against torch autograd: loss and the gradient of **every** parameter tensor.
   Worst gradient relative L2 over the f32 families: **2e-15** (tolerance 1e-10); bf16 emulation 4.4e-3 (Llama, tolerance 1e-2: one bf16 ulp is 3.9e-3; torch sums unrounded branch gradients before a cast where the frontend rounds per consumer) and 0 (MLP).
2. **A100, executor** (`torch-zoo/torch_compare.{txt,md,json}`, `torch_report.py compare`, `torch_run.sh`): one-step bundles of 14 cases run through `kvm_exec`, the executor's step-0 loss(es) and the step-0 gradient of every parameter
   tensor (the `g` of the step-0 AdamW tasks; `reuse none` keeps them) against torch float64 on the same bytes. 14 of 14 PASS:

| case | mode | params | loss rel err vs torch f64 | worst gradient rel L2 (tensor) | global grad norm rel | tolerance (loss / grad) | verdict |
|---|---|---|---|---|---|---|---|
| gpt_small_precise | precise | 36 | 2.1e-08 | 2.97e-06 (gpt.h1.attn.k.w) | 6.9e-07 | 1e-05 / 3e-05 | PASS |
| gpt_small_tf32 | tf32 | 36 | 2.6e-06 | 8.58e-04 (gpt.h0.attn.q.b) | 5.1e-07 | 0.0005 / 0.003 | PASS |
| llama_precise | precise | 21 | 5.8e-08 | 2.05e-06 (llama.h1.attn.k.w) | 3.0e-07 | 1e-05 / 3e-05 | PASS |
| llama_tf32 | tf32 | 21 | 5.6e-08 | 7.01e-04 (llama.h0.n2.g) | 1.4e-05 | 0.0005 / 0.003 | PASS |
| llama_gqa_tied_precise | precise | 20 | 2.0e-08 | 2.15e-06 (llama.h0.attn.k.w) | 3.0e-07 | 1e-05 / 3e-05 | PASS |
| moe_precise | precise | 22 | 3.8e-09, 5.0e-08 | 1.96e-06 (llama.h0.attn.k.w) | 2.5e-07 | 1e-05 / 3e-05 | PASS |
| moe_tf32 | tf32 | 22 | 6.8e-07, 5.2e-07 | 6.38e-04 (llama.h0.n2.g) | 4.4e-05 | 0.0005 / 0.003 | PASS |
| vit_precise | precise | 23 | 2.6e-08 | 1.13e-06 (vit.h0.attn.k.w) | 1.1e-07 | 1e-05 / 3e-05 | PASS |
| vit_tf32 | tf32 | 23 | 9.7e-06 | 8.20e-04 (vit.h0.attn.q.w) | 1.1e-04 | 0.0005 / 0.003 | PASS |
| mlp_precise | precise | 6 | 4.8e-07 | 6.64e-07 (mlp.fc2.w) | 4.8e-07 | 1e-05 / 3e-05 | PASS |
| mlp_tf32 | tf32 | 6 | 5.0e-05 | 4.16e-04 (mlp.fc0.b) | 9.0e-05 | 0.0005 / 0.003 | PASS |
| llama_bf16 | lowp | 21 | 2.8e-08 | 6.05e-03 (llama.h0.n1.g) | 4.1e-05 | 0.0001 / 0.02 | PASS |
| mlp_bf16 | lowp | 6 | 2.5e-08 | 2.70e-08 (mlp.fc0.w) | 3.5e-09 | 0.0001 / 0.02 | PASS |
| vit_bf16 | lowp | 23 | 6.5e-10 | 6.78e-03 (vit.embed.w) | 3.0e-04 | 0.0001 / 0.02 | PASS |


Tolerances are relative L2 per tensor (loss relative), set at 3-10x the measured worst: precise (3xTF32) loss 1e-5 / gradients 3e-5; tf32 (1xTF32) 5e-4 / 3e-3 (10-bit mantissa operands, the class of `2026-11-03-a100-gptops`); lowp (bf16) 1e-4 / 2e-2.
The ViT key bias, whose true gradient is exactly zero, is judged in absolute terms (`ParamSpec.invariant`, see (b)): at most 1e-3 of the largest gradient norm. Negative control: raising one gradient tensor of the dump by 1e-4 relative fails the precise case
(9.8e-5 against 3e-5). The "GPT" rows use `gpt_small` (2 layers, d64, 4 heads); gpt_162m's step-1 tie to PyTorch (all 196 tensors) is `2026-11-06-gpt162-vs-torch`.

## Fixed on the way

`main` had **17 erroring tests in `tests/frontend`** (`train.batch_tokens` read `cfg.repeat_batch`, which only the GPT config has; every Llama / MoE / MLP / ViT compile failed). One-line fix (`getattr`); `python3 -I -m unittest discover -s tests`
(145) and `-s tests/frontend` (83, was 63 with 17 errors) pass, and `-s examples/models` (8).

## Proposed status text (not applied; README.md / AGENTS.md are not edited here)

* Known failures: add "bf16 zoo: `san_bf16s` weights 1.7e-2 vs 1e-2 (open); ViT key-bias weight check replaced by an absolute envelope check (pending A100 confirmation)".
* "Implemented + measured": serving at 162M checked against a PyTorch float64 decode for the version-0 and final-weights requests (`2026-11-06-a100-correctness`); every zoo family's step-1 loss and gradients tied to PyTorch float64 (14 cases, 3xTF32 <= 3e-6, 1xTF32 <= 8.6e-4, bf16 <= 6.8e-3).
* "Not built": dynamic shapes / data-dependent control flow -> "guards exist only in the CPU reference runner; kvm_run, masked tiles, device-side conditional admission not built (FRAMEWORK.md)".
