# GPT golden mismatch: root cause and check (A100, 2026-11-04)

- GPU: NVIDIA A100-SXM4-40GB, 108 SMs, sm_80, driver 580.105.08, nvcc 12.8; box shared (`flock`). `kvm_exec` built as in `tests/gpu/build.sh` (`--fmad=false -ffp-contract=off`), run with `--blocks 4`.
- Code: `tests/gptdiff/{gen_bundle.py,golden_check.py,run.sh}` at the commit that adds this file. Command: `KVM_HOST=... tests/gptdiff/run.sh --sanitize`. Raw output: `run.txt`.

## Verdict: not a bug in the ops, the executor, the lowering or the initial values

The "golden mismatch" of `exec-recycling/README.md` is AdamW amplifying legitimate 1xTF32 gradient noise.

1. **Everything before the optimizer agrees at TF32 level.** Step-1 activations and gradients (59 buffers): worst relative L2 7.3e-4 (tolerance table: 6e-3); loss.0 2.5e-6, loss.1 1.4e-5. clip_coef after step 1: 1e-5.
2. **The adamw op is exact.** Applying float64 AdamW to the GPU's *own* gradient reproduces the GPU weights to 6e-8 and m, v to 2.9e-7 (and the CPU golden to 5e-8 from its own gradient). So the initial weights are bit-identical on both sides and the op, step counter and clip scale are right.
3. **The gap is sign flips.** AdamW's first update is `g/(|g|+1e-8)` ~ `sign(g)`: a step function. With 1 step, exactly 2 gradient elements (of ~12k) change sign between CPU fp32 and GPU 1xTF32, both with |g| far below the TF32 noise: `attn.k.w[490]` (g_cpu -7.5e-8, g_gpu +1.8e-8, tensor max |g| 3.3e-4) and `mlp.fc2.b[17]` (g_cpu -4.0e-6, g_gpu +2.7e-5, max |g| 0.37, max |dg| 1.1e-4). Each flip moves its weight by 2*lr = 6e-3; they explain 99.5% of the squared weight error. `fc2.b` has 32 elements, so one flip is 35% relative L2 of that tensor.
4. **The same bundle with 3xTF32 GEMMs (matmul flag 16) matches the CPU golden everywhere:** worst buffer relative L2 1.6e-5 (1 step), 2.4e-5 (2), 3.7e-5 (5), against 1.3e-5 for the compiler's float32 CPU reference vs the frontend's float64 oracle. After step 1 the 1xTF32 run follows a different (equally valid) trajectory; the post-Adam gap (7.5e-2 at 2 steps, 6.2e-2 at 5) is that divergence, and losses stay within 2.3e-4 over 5 steps. clip_coef 1.7e-3 in the report is the step-2 coefficient, computed from already-diverged weights.

## The check (tests/gptdiff/golden_check.py) and what passes
`tf32`: losses, step-1 activations/gradients, (1 step) adamw-op reproduction in float64 from the GPU's own gradients, and every GPU weight inside the image of the gradient tolerance band under Adam's monotone update (a flip is accepted only if the gradient is within the documented tolerance of zero). `precise`: every f32 buffer within 2e-4 of the golden. A deliberately wrong weight (+5e-4 on one element) fails both the op and the band checks.

| bundle | 1xTF32 | 3xTF32 | kvm_exec |
|---|---|---|---|
| gpt_tiny cpu tiling, 1 step (272 tasks) | PASS | PASS | OK |
| gpt_tiny cpu tiling, 2 steps (544 tasks) | PASS | PASS | OK |
| gpt_tiny cpu tiling, 5 steps (1360 tasks) | PASS | PASS | OK |
| gpt_tiny a100 tiling, 2 steps (206 tasks) | PASS | PASS | OK |

compute-sanitizer memcheck, racecheck, synccheck on the 2- and 5-step bundles, 1xTF32 and 3xTF32: 0 errors, 0 hazards (`run.txt`).

## Not covered
- The 1xTF32 post-Adam weights are not compared to the golden (they cannot be); only the first update is modelled. Bitwise equality with a CPU run is only possible with an fp32-exact GEMM.
- `kvm_exec --golden` itself is unchanged (whole-buffer relative L2 is right for fp32 ops, wrong for TF32 + AdamW); use `tests/gptdiff/run.sh` for GPT bundles.
- The frontend has no switch for 3xTF32; `gen_bundle.py --precise` sets flag 16 on every `matmul_tile` in the emitted workload.
