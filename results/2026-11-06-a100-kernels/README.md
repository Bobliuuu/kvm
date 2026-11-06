# A100-40GB (Lambda, 1410 MHz) GPT-op kernel A/B: base fc0ced7 vs K's tile_round + Acc3 edits (2a8724c)

Raw outputs: `{head,base}.{test,mm,attn}.txt`, plus mm-only sweeps `n4` (KVM_GPT_BIG_NST=4) and `velt` (KVM_RND_VELT) on head.

Correctness: test_ops 1396 passed / 0 failed for both head and base; all bench checks PASS.

Attention (3xTF32, ms, full dispatch): the Acc3/mma3s edit REGRESSED it.
- fwd B4 T1024 H16 hd64: 0.716 (base) -> 1.230 (head)
- bwd B4 T1024 H16 hd64: 2.308 (base) -> 4.886 (head)
(other shapes noisy; hd128 equal within noise). Likely cause: three accumulator sets per mma chain raise register use / lower occupancy.
Action: the attention hunks were reverted (attention code is identical to base again).

Matmul TF32: run-to-run timings are bimodal on this box (same binary: ~3 ms vs ~6.7 ms for 4096^3, i.e. ~46-50 vs ~20 TFLOP/s;
the fast/slow mode flips between layouts/runs, see [min..max] columns), so the one-shot head-vs-base matmul numbers are NOT a valid comparison.
Examples: 256x128 NN base 50.0 vs head 20.3 TFLOP/s; 256x128 TN base 21.5 vs head 46.2. n4 (4 stages): 256x128 NN 46.9, 128tile NN 40.0 (fast mode).
No tile_round win demonstrated. Peak seen: ~50 TFLOP/s (cuBLAS 105.6). The bimodality needs explaining (clock/L2/partition camping/other tenants) before any further tuning.
