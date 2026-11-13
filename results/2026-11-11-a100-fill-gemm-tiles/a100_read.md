# #41 (before its AdamW chunks were sized to the model) against #41 + this PR, both on the read's frontend, one executor (6bf2bc1b),
# A100-SXM4-80GB, Nov 11 14:45-14:50Z. The weights' largest difference is reported, not gated: this PR changes its stepped-down GEMMs' summation order.

## The two draws, 5 paired rounds
| row | #41 | this branch | speed | rounds | tasks | weights |
|---|---|---|---|---|---|---|
| ho-d384-l3-t96-b4 | 86.61 ms | 71.43 ms | x1.212 (x1.212-1.212) | 5 | 71,625 → 107,417 | not gated (largest relative 2.0e+00) |
| ho-d512-l1-t64-b2 | 27.75 ms | 25.41 ms | x1.092 (x1.091-1.093) | 5 | 30,843 → 40,669 | not gated (largest relative 2.0e+00) |

## The 12 hold-out shapes, 3 paired rounds
| row | #41 | this branch | speed | rounds | tasks | weights |
|---|---|---|---|---|---|---|
| ho-gpt-d192-h3-l1-t32-b1-v64 | 9.61 ms | 9.61 ms | x0.999 (x0.999-1.000) | 3 | 8,703 → 10,233 | not gated (largest relative 1.9e+00) |
| ho-gpt-d192-h3-l2-t64-b4-v256 | 28.44 ms | 25.39 ms | x1.121 (x1.119-1.121) | 3 | 33,195 → 41,207 | not gated (largest relative 2.0e+00) |
| ho-gpt-d192-h3-l4-t128-b2-v512 | 53.94 ms | 47.94 ms | x1.125 (x1.125-1.126) | 3 | 61,021 → 78,330 | not gated (largest relative 2.0e+00) |
| ho-gpt-d256-h4-l1-t128-b4-v512 | 36.44 ms | 28.79 ms | x1.266 (x1.265-1.266) | 3 | 21,877 → 42,460 | not gated (largest relative 2.0e+00) |
| ho-gpt-d256-h4-l2-t32-b2-v128 | 22.93 ms | 23.21 ms | x0.988 (x0.988-0.989) | 3 | 24,884 → 30,886 | not gated (largest relative 2.0e+00) |
| ho-gpt-d256-h4-l3-t96-b4-v256 | 48.91 ms | 43.37 ms | x1.128 (x1.127-1.128) | 3 | 71,606 → 94,187 | not gated (largest relative 2.0e+00) |
| ho-gpt-d384-h6-l1-t64-b1-v64 | 17.59 ms | 17.69 ms | x0.995 (x0.993-0.995) | 3 | 17,530 → 20,863 | not gated (largest relative 2.0e+00) |
| ho-gpt-d384-h6-l3-t96-b4-v256 | 86.65 ms | 71.49 ms | x1.212 (x1.211-1.213) | 3 | 71,625 → 107,417 | not gated (largest relative 2.0e+00) |
| ho-gpt-d384-h6-l4-t128-b4-v512 | 120.27 ms | 100.41 ms | x1.197 (x1.197-1.200) | 3 | 102,872 → 169,138 | not gated (largest relative 2.0e+00) |
| ho-gpt-d512-h8-l1-t64-b2-v256 | 27.75 ms | 25.43 ms | x1.090 (x1.090-1.092) | 3 | 30,843 → 40,669 | not gated (largest relative 2.0e+00) |
| ho-gpt-d512-h8-l2-t128-b4-v512 | 100.09 ms | 78.67 ms | x1.271 (x1.271-1.275) | 3 | 63,321 → 117,241 | not gated (largest relative 2.0e+00) |
| ho-gpt-d512-h8-l4-t32-b1-v128 | 62.00 ms | 62.46 ms | x0.993 (x0.992-0.993) | 3 | 88,997 → 114,157 | not gated (largest relative 2.0e+00) |
