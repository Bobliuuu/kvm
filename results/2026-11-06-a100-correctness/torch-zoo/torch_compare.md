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
