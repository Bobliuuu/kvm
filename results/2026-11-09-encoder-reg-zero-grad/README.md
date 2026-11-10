# `encoder_reg` zero-gradient failure: noise against a relative bound (not a bug), and the manifest-stage quoting bug

GPU: A100-SXM4-40GB, driver 580.105.08, CUDA 12.8 (shared box, one `~/.gpu.lock` hold per run). Branch `fix/encoder-reg-zero-grad` off `landing/all-2026-11-09`.
Failure being explained: `tests/generic/run.sh` `encoder_reg` failed the 2e-4 golden bound on both tilings (`v.enc.k.b` rel L2 10.5, `act53` 4.18, ...).

## Verdict: rounding noise around an exactly-zero gradient

The key-bias gradient of attention is mathematically 0: softmax is shift invariant in `k`'s bias, so `sum_j dK_j = sum_i q_i (sum_j dS_ij) = 0`
(each `dS` row sums to 0). The lowering emits it as `colsum(dK)` (`act53` step 1, `act115` step 2) and `m`, `v`, `w` of `enc.k.b` inherit it.

| Check | Result (`analysis-encoder_reg.txt`) |
|---|---|
| float64 reference (`refvm`, f32=False) | `act53` max 1.6e-17, `act115` 1.0e-17 against a column mass `sum_r\|dK\|` up to 0.224 (ratio 7e-17): exactly zero to f64 rounding |
| executor / CPU golden | 1.1e-8 / 2.8e-9 (`act53`): both are noise, of different sizes; noise vs noise is O(1) relative, and `v` (a square) 1e-19 vs 1e-20 is 10x |
| exactly which tensors fail | 5 of 197 f32 buffers: `act53`, `act115`, `m/v/w.enc.k.b`. The next worst, `act69` 4.3e-5, is the real worst of everything else (5x under the bound) |
| non-zero tensors deviate? | no. `dK` itself (`act47`, max 1.55e-2) differs from f64 by 9e-9 (6e-7 relative); all 192 other buffers are within 4.3e-5 of the golden. Scan of all four generic cases for buffers that are ~0 in f64 but nonzero on device/golden: only these 5 in `encoder_reg`; none in `reg_mlp`, `res_rms_mlp`, `cnn1d` |
| determinism | executor dumps bitwise identical across 3 A100 runs (landing run, run 2, run 3) for `encoder_reg` (both tilings), `cnn1d`, `reg_mlp`, `res_rms_mlp` (buffer lines compared; the timing line differs). So the failure was not a flake: it was a fixed data set against a relative bound |
| scale | `w.enc.k.b` differs by 2e-4 only because AdamW (`eps` 1e-8) turns 1e-9-sized noise into a 1e-4 step; it says nothing about the gradient |

## Fix (`tests/gptdiff/structural_zero.py`, used by `golden_check.py --mode precise`)

- Detect structurally-zero `colsum` outputs by an independent float64 evaluation, never by the device value: float64 `|out_c| <= 1e-12 * sum_r|x_rc|` for every column
  (run only if the CPU golden already shows such a colsum, so ordinary workloads pay nothing). In the generic cases that is exactly the key-bias chain (the whole category).
- Bound per column and per run from that run's own `dK`: `|g_c| <= (R+8) * 2^-23 * sum_r|dK_rc|`. Worst measured: 0.011 of the bound for the executor, 0.004 for the golden (bound 1.5e-6 vs noise 1.1e-8).
- Inherited state: `|m| <= max bound`, `v <= max bound^2`, `w` within `n_steps * lr * U` of its start (`U` the Adam update cap). The `w` check is weak by construction (Adam normalises: a real bug also moves `w` about `lr`); the gradient bound is the one that detects bugs.
- Everything else keeps the 2e-4 relative bound; the zero-class tensors are excluded from it and checked in absolute terms for both the executor and the golden.

Negative tests (`tests/gptdiff/test_structural_zero.py`, on the real `encoder_reg` bundle, CPU only): noise at 2x the bound fails (both steps); a dropped `dK` term fails (colsum 1e-3-scale, 1e2+ over the bound); one sequence's `dK` scaled by 1.01 fails; an inflated `m` or a moved `w` fails;
a real bias gradient (`act38`) is not classified as zero; an ordinary workload (`reg_mlp`) never starts the float64 run; measured-size noise and the golden pass.

## Manifest stage lost its scratch: quoting, not scratch lifecycle

`run.sh` ran the remote commands as `bash -c '$RUN'`, but `$RUN` contains its own single quotes (`--observe '^loss\.'`, `--compile-cmd '...'`). They closed the outer quote, so the
remote shell executed a mangled line: the first demo started with truncated arguments, its output redirects went astray, and `out_cnn1d` was never created, which looked like a lost scratch
(`scp ... out_reg_mlp.log: No such file`, `rsync ... out_cnn1d failed: No such file`). Reproduced locally with a stub (the string splits after `--compile-cmd python3`). The private scratch was never removed early:
`kvm_scratch_done` runs only after success. Fix: the remote commands are written to a script file (`bash -n` checked, copied to the scratch, run under the same lock hold), and the run fails unless every
`NAME exit 0` and `man_NAME kvm_run exit 0` line is present.

## Suite results (this branch, A100)

`tests/generic/run.sh`, runs 2 and 3 (`generic-run2.txt`, `generic-run3.txt`; run 1 built `kvm_exec`/`kvm_run`, a 90+ minute nvcc on the loaded shared box, and its local script was killed during the build before any GPU work):
exit 0 both times; `reg_mlp`, `res_rms_mlp`, `cnn1d`, `encoder_reg` PASS on cpu and a100 tilings (worst relative 4.3e-5 outside the zero class); `man_reg_mlp` and `man_cnn1d` ALL CHECKS PASSED
(6 steps, loss falls, 3.9e-7 and 3.1e-9 from the CPU golden). CPU: `python3 -I -m unittest discover -s tests/gptdiff` 13 tests OK.
