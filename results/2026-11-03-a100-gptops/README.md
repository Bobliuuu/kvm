# GPT operator library on A100 (sm_80)

Code commit: 2d364bf (branch feat/gptops). Ops: `include/kvm_ops_gpt.def` (ids 32-63), `include/kvm_ops_gpt.cuh`.
Tests: `tests/gpu_ops/{test_ops,test_e2e,bench_ops}.cu`, CPU double references in `tests/gpu_ops/refs.hpp`.

- GPU: NVIDIA A100-SXM4-40GB, 108 SMs, 1410 MHz; driver 580.105.08; CUDA 12.8 (nvcc 12.8.93)
- Flags: `nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -I../../include -I. <test>.cu` (no fast-math)
- Command: `KVM_HOST=user@host ./run_gpu_ops.sh <test_ops|test_e2e|bench_ops> [args]`; sanitizers via
  `WRAP="compute-sanitizer --tool <tool> --error-exitcode 9" ./run_gpu_ops.sh test_ops`. Every run is under `flock ~/.gpu.lock`.
- The box is shared and serialized by the lock; clocks and thermals are not controlled, so timings are indicative.

## Results (all PASS; files in this folder)
| file | result |
|---|---|
| test_ops.txt | 507 checks PASS: every op vs CPU double, edge cases (non-tile shapes, n=1, T=1, odd prefill lengths 2..97, ignore_index everywhere, logits +-80, V=1, d=1), GQA, determinism (bitwise), error codes, grad clip, sample exact-match, Philox known answers |
| test_e2e.txt | 7 PASS: 2-layer GQA GPT (vocab 64, d 32, 2 heads/1 KV head, seq 16, batch 4, rope, tied head), 200 AdamW steps as per-op launches; max abs loss difference to the double CPU model 2.0e-4 over 200 steps (tol 2e-3); step-0 gradients of all 36 tensors within 3.0e-6 relative; loss 5.527 -> 2.271; two GPU runs bitwise identical; CPU analytic gradient == finite differences to 4e-9 |
| sanitizer_{memcheck,racecheck,synccheck,initcheck}_test_ops.txt | 0 errors / 0 hazards |
| (not run) | memcheck on test_e2e: the 200-step run exceeds the 600 s timeout under compute-sanitizer with the final code, so no result is claimed (an earlier commit's run reported 0 errors; the e2e launches the same ops the unit tests cover) |
| bench_ops.txt | 16 correctness checks PASS, then timings below |

## Tolerances (relative to max |reference|)
matmul 1xTF32 6e-3 (measured 2.5-3e-4 at K=4096); 3xTF32 `precise` 2e-6*sqrt(K) (measured 1.8e-5 at K=4096, 2e-5 bound for K<=100);
norms, gelu, rope, embeddings, adamw, cross entropy 1e-5..2e-5; attention fwd 2e-5 (measured 1e-7), bwd 3e-5 (measured 1e-6);
decode 2e-5; sample exact index.

## Performance (A100 peaks: TF32 156 TFLOP/s, FP32 19.5 TFLOP/s, HBM 1555 GB/s)
"full dispatch" = the all-ops kernel (150 registers, what a megakernel compiles); "single-op" = a kernel holding only that op.
Matmul counts 2MNK nominal flops; attention counts causal 2 matmuls (fwd) and 2.5x that (bwd, recompute included).
| op | full dispatch | single-op |
|---|---|---|
| matmul TF32 NN/TN/NT 4096^3 | 29.4 / 28.4 / 21.5 TFLOP/s (14-19%) | 35.6 / 34.7 / 29.1 (19-23%) |
| matmul 3xTF32 (fp32 accuracy) 4096^3 | 17.1 TFLOP/s nominal (33% of the 52 effective peak) | 19.8 |
| matmul TF32 2048^3 / 8192x768x768 | 27.0 / 23.4 | 32.4 / 29.5 |
| attention fwd hd64 T1024 B4 H16 | 4.1 TFLOP/s (21% of fp32 peak) | 5.1 (26%) |
| attention bwd hd64 T1024 | 3.2 (17%) | 3.7 (19%) |
| attention fwd hd64 T2048 GQA 16:4 | 4.6 | 5.3 |
| attention fwd/bwd hd128 T2048 | 3.2 / 2.6 | 3.5 / 2.7 |
| layernorm fwd 16384x1024 | 887 GB/s (57%) | |
| adamw 16M params | 1025 GB/s (66%) | |
| softmax CE fwd / bwd 8192x50304 | 915 / 884 GB/s (59% / 57%) | |

Honest assessment: the matmul is a 64x64-tile wmma kernel with cp.async double buffering, about 20% of TF32 peak (cuBLAS
reaches 60-75%); attention is fp32 SIMT flash attention at about 20% of FP32 peak and uses no tensor cores. Both are
correct and deterministic first versions, not tuned.
