# GEMM tile variants, A100-40GB (Modal), bench_mma_bf16 MMA_SET=gpt, run r2
Variants: base (BK=32, 4 stages, restructured loop), bk32s (+SPREAD), bk64 (3 stages), bk64s (BK=64 + SPREAD). All correctness checks ok (err <= 3.6e-3 vs cuBLAS).
Absolute TFLOP/s are noisy between variants (cuBLAS on the same shape moves 124..161), so compare ratios kvm/cuBLAS.
Only clear difference: head 4096x768x50304 NT ratio 0.54 (base) -> 0.66-0.68 (SPREAD/BK64). Everything else within ~0.03 of base (0.34-0.45); qkv wgrad 768x768x4096 stays 0.17-0.21; head fwd NN 0.21.
Conclusion: BK=64 / spread copies do not close the gap to cuBLAS; loss is elsewhere (warp-specialised pipeline, split-K for skinny wgrad not tried).
Raw logs in raw/r2.
