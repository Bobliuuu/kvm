| case | model | steps | tasks | mode | loss (steps spread over the run) | max error vs CPU golden | exec time ms (min of runs) |
|---|---|---|---|---|---|---|---|
| gpt_fused | gpt_tiny | 5 | 1,340 | precise+fuse | 4.146 3.994 3.768 3.866 3.640 | 3.7e-05 (rel L2, all buffers) | 6.93 (n=3) |
| gqa_tied | llama_gqa_tied | 10 | 8,016 | precise | 4.228 3.670 3.752 3.299 2.913 | 2.9e-05 (rel L2, all buffers) | 31.86 (n=3) |
| llama_accum | llama_tiny | 5 | 5,222 | precise | 4.180 3.872 3.627 3.510 3.286 | 1.1e-04 (rel L2, all buffers) | 20.74 (n=3) |
| llama_bf16 | llama_bf16 | 10 | 9,636 | lowp | 4.213 3.672 3.730 3.357 2.969 | 1.1e-04 (step-1 acts/grads, max|e|/max|ref|) | 39.46 (n=3) |
| llama_f32 | llama_tiny | 10 | 8,336 | precise | 4.213 3.543 3.704 3.169 2.646 | 4.9e-04 (rel L2, all buffers) | 32.05 (n=3) |
| llama_fused | llama_tiny | 10 | 8,158 | precise+fuse | 4.213 3.543 3.704 3.169 2.646 | 4.9e-04 (rel L2, all buffers) | 31.09 (n=3) |
| llama_remat | llama_remat | 5 | 6,498 | precise | 4.215 3.877 3.558 3.666 3.393 | 3.2e-04 (rel L2, all buffers) | 25.27 (n=3) |
| llama_tf32 | llama_tiny | 5 | 4,166 | tf32 | 4.213 3.894 3.543 3.657 3.213 | 1.1e-03 (step-1 acts/grads, max|e|/max|ref|) | 15.85 (n=3) |
| mlp_bf16 | mlp_bf16 | 30 | 2,790 | lowp | 2.493 0.697 0.092 0.027 0.012 | 1.3e-07 (step-1 acts/grads, max|e|/max|ref|) | 18.50 (n=3) |
| mlp_f32 | mlp_tiny | 40 | 2,760 | precise | 2.493 0.427 0.060 0.042 0.033 | 5.0e-06 (rel L2, all buffers) | 18.50 (n=3) |
| mlp_tf32 | mlp_tiny | 5 | 340 | tf32 | 2.493 1.867 1.539 1.409 1.144 | 8.8e-04 (step-1 acts/grads, max|e|/max|ref|) | 1.97 (n=3) |
| moe_f32 | moe_tiny | 10 | 9,990 | precise | 4.170 3.645 3.101 2.723 2.453 | 7.1e-05 (rel L2, all buffers) | 38.93 (n=3) |
| moe_fused | moe_tiny | 10 | 9,590 | precise+fuse | 4.170 3.645 3.101 2.723 2.453 | 7.1e-05 (rel L2, all buffers) | 34.69 (n=3) |
| moe_tf32 | moe_tiny | 5 | 4,995 | tf32 | 4.170 3.885 3.645 3.497 3.209 | 8.5e-04 (step-1 acts/grads, max|e|/max|ref|) | 19.75 (n=3) |
| san_bf16 | llama_bf16 | 2 | 1,922 | lowp | 4.213 3.893 | 1.1e-04 (step-1 acts/grads, max|e|/max|ref|) | 8.47 (n=3) |
| san_gqa_fused | llama_gqa_tied | 2 | 1,566 | precise+fuse | 4.228 4.002 | 2.1e-05 (rel L2, all buffers) | 5.96 (n=3) |
| san_llama | llama_tiny | 2 | 1,664 | precise | 4.213 3.894 | 1.5e-04 (rel L2, all buffers) | 6.38 (n=3) |
| san_mlp | mlp_tiny | 3 | 207 | precise | 2.493 1.867 1.709 | 1.1e-06 (rel L2, all buffers) | 1.08 (n=3) |
| san_moe | moe_tiny | 2 | 1,998 | precise | 4.170 3.885 | 4.1e-05 (rel L2, all buffers) | 7.74 (n=3) |
| san_moe_fused | moe_tiny | 2 | 1,918 | precise+fuse | 4.170 3.885 | 4.1e-05 (rel L2, all buffers) | 7.24 (n=3) |
| san_remat | llama_remat | 2 | 2,356 | precise | 4.180 3.872 | 9.8e-05 (rel L2, all buffers) | 8.89 (n=3) |
| san_vit | vit_tiny | 2 | 850 | precise | 1.382 1.416 | 8.8e-06 (rel L2, all buffers) | 3.50 (n=3) |
| vit_f32 | vit_tiny | 20 | 8,500 | precise | 1.382 1.251 1.125 0.902 0.779 | 5.9e-05 (rel L2, all buffers) | 38.19 (n=3) |
| vit_tf32 | vit_tiny | 5 | 2,125 | tf32 | 1.382 1.416 1.345 1.300 1.287 | 9.4e-04 (step-1 acts/grads, max|e|/max|ref|) | 9.35 (n=3) |
