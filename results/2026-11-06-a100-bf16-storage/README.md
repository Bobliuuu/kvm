# Real bf16 storage (ABI dtype 2) on the A100

Code: branch `feat/bf16-hopper`. GPU: A100 (sm_80), shared box, one `flock` hold per run. Commands:
`KVM_HOST=... tests/gpu_ops/run_extra.sh` (op tests) and `KVM_HOST=... tests/bf16/run.sh OUT` (gpt_tiny bundles through `kvm_exec`).

## What was built
`KvmBuf.dtype` 2 = bf16 (2 bytes/element); workload dtype `"bf16"`; executor allocates, initialises (fill / values / file) and reads back
(widened) bf16 buffers. `round_bf16` is the narrow/widen op; `matmul_tile_bf16` accepts bf16 a, b, out, res, pre (fp32 math, one RNE rounding per
stored value). Every other op rejects a bf16 buffer (compiler check, and device E_DTYPE). Frontend `precision='bf16s'` (GPT): `kvmfe/storage.py`
folds casts into matmul epilogues, keeps fp32 master weights + AdamW, adds bf16 weight shadows, wraps non-aware ops in widen/narrow. CPU golden
(`Buf.bf16`) rounds on store.

## Results
* CPU: 152 + 67 unit tests pass (new: `tests/test_bf16_storage.py`, `tests/frontend/test_bf16_storage.py`).
* `test_extra` on the A100: first run `test_extra.run.log`: 3668 passed, 1 failed; the failure was a wrong expectation in my new test (partial-range copy);
  fixed in the test. 752 bf16-storage matmul combinations (ta, tb, shapes, a/b/out/res/pre dtype mixes) pass, worst rel err 3.8e-3 (bf16 outputs), bf16
  narrow/widen bitwise vs an independent RNE reference, bf16 buffer given to `act_fwd` is rejected with E_DTYPE and memory untouched. Rerun: `test_extra_rerun.log`.
* gpt_tiny through `kvm_exec` (`gpt_tiny/*/check.txt`): `bf16s_s5` losses equal the CPU golden to 1e-9 relative on all 5 steps; `bf16s_s1` step-1
  activation buffers within 1.7e-5 of golden; all bf16 buffers hold only bf16-representable values.
* Loss curve, 50 steps, fp32 vs bf16 storage (GPU): step 0 4.14605 vs 4.14610, step 20 2.73614 vs 2.73699, step 35 2.04294 vs 2.04222 (full curves in `gpt_tiny/*/check.txt`).

## Memory: NOT halved overall (honest)
Each bf16 tensor is exactly half its fp32 size, but only matmul outputs are stored bf16; norm / rope / attention / gelu-bwd / loss ops are fp32-only, so their
inputs go through fp32 scratch and their outputs stay fp32. Planner totals (persistent + activation pool, CPU plan, 1 step):

| model (L, d, T, B) | fp32 | bf16s, no weight shadows | bf16s with shadows |
|---|---|---|---|
| gpt_tiny (1, 32, 32, 1) | 346,696 B | 1.04x | 1.12x |
| (2, 64, 64, 2) | 3,059,824 B | 0.98x | 1.05x |
| (8, 128, 256, 4) | 97,297,504 B | 0.88x | 0.92x |

Weight shadows (2 B/param on top of 16 B/param of master, moments, gradient) cost more than they save at these sizes. Halving needs bf16-native
norm, rope, attention, softmax and AdamW-gradient paths (not built), and fp16 is not built.
