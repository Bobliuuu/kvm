| model | params | loss CPU f64 VM vs numpy oracle (abs) | worst autodiff vs finite-difference grad error (spot check of every tensor) |
|---|---|---|---|
| mlp_tiny | 1,732 | 4.4e-16 | 3.0e-10 |
| vit_tiny | 13,956 | 2.2e-16 | 6.0e-08 |
| llama_tiny | 69,952 | 8.9e-16 | 4.6e-07 |
| llama_gqa_tied | 65,856 | 0.0e+00 | 1.1e-06 |
| moe_tiny | 74,176 | 8.9e-16 | 7.3e-07 |
| moe_drop_k1 | 49,600 | 8.9e-16 | 2.6e-07 |
