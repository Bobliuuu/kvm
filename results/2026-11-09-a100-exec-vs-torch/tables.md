### mlp_runs: hidden 72, depth 2, 50 steps, best of 7 [median .. worst], us per step (all R runs advance one step)

| R | eager | compile (no graph) | compile + graph | graphs, 1 stream | graphs, R streams | batched graph | batched compile+graph | executor |
|---|---|---|---|---|---|---|---|---|
| 1 | 803.9 [811.4 .. 817.6] | 956.6 [968.5 .. 971.1] | 62.6 [64.6 .. 76.1] | 115.2 [119.3 .. 140.3] | - | - | 60.8 [62.9 .. 74.0] | 554.1 [565.2 .. 636.7] |
| 2 | 1594.6 [1634.0 .. 1713.0] | - | - | 215.5 [215.7 .. 277.4] | 127.2 [127.4 .. 164.1] | 132.3 [132.3 .. 170.1] | 61.6 [61.7 .. 79.1] | 720.4 [727.7 .. 842.9] |
| 4 | 2995.6 [3138.7 .. 3305.2] | - | - | 430.8 [430.9 .. 554.3] | 151.9 [152.8 .. 175.4] | 137.0 [137.0 .. 176.1] | 65.2 [65.3 .. 83.7] | 1091.4 [1101.3 .. 1246.0] |
| 8 | 5644.8 [5715.9 .. 6045.4] | - | - | 861.9 [862.1 .. 1109.0] | 291.3 [296.6 .. 298.8] | 146.4 [146.5 .. 188.2] | 71.2 [71.3 .. 91.6] | 1872.9 [1884.5 .. 1906.0] |
| 16 | - | - | - | 1739.9 [1742.5 .. 2242.0] | 573.9 [587.2 .. 741.7] | 152.9 [152.9 .. 196.4] | 79.3 [79.4 .. 102.1] | 3524.1 [3544.5 .. 3556.8] |
| 32 | - | - | - | 3564.8 [3565.1 .. 4092.9] | 1127.5 [1144.5 .. 1688.9] | 197.9 [197.9 .. 198.0] | 96.3 [96.3 .. 96.6] | 6614.9 [6640.6 .. 6670.5] |

Ratio = variant best / executor best (above 1: the executor is faster; below 1: the variant is faster).

| R | eager | compile (no graph) | compile + graph | graphs, 1 stream | graphs, R streams | batched graph | batched compile+graph |
|---|---|---|---|---|---|---|---|
| 1 | 1.45x | 1.73x | 0.11x | 0.21x | - | - | 0.11x |
| 2 | 2.21x | - | - | 0.30x | 0.18x | 0.18x | 0.09x |
| 4 | 2.74x | - | - | 0.39x | 0.14x | 0.13x | 0.06x |
| 8 | 3.01x | - | - | 0.46x | 0.16x | 0.08x | 0.04x |
| 16 | - | - | - | 0.49x | 0.16x | 0.04x | 0.02x |
| 32 | - | - | - | 0.54x | 0.17x | 0.03x | 0.01x |

Gate: executor vs eager loss max rel 2.40e-04, weights max rel L2 5.07e-05; vs float64: executor 3.05e-04, eager 1.89e-04; loss 4.5369 -> 0.3433; tasks/step/run 51

### mlp_size_1024: hidden 1024, depth 2, 50 steps, best of 7 [median .. worst], us per step (all R runs advance one step)

| R | eager | compile (no graph) | compile + graph | graphs, 1 stream | graphs, R streams | batched graph | batched compile+graph | executor |
|---|---|---|---|---|---|---|---|---|
| 1 | 852.3 [877.8 .. 1252.3] | 1050.0 [1116.1 .. 1341.4] | 104.3 [104.3 .. 134.0] | 129.0 [129.1 .. 165.7] | - | - | 104.0 [104.1 .. 133.7] | 3152.7 [3180.9 .. 3607.3] |
| 8 | 6097.5 [6211.4 .. 6376.6] | - | - | 1115.8 [1116.1 .. 1369.8] | 440.4 [442.7 .. 451.4] | 408.8 [408.9 .. 416.6] | 316.2 [316.3 .. 323.0] | 14028.8 [14060.4 .. 14106.2] |

Ratio = variant best / executor best (above 1: the executor is faster; below 1: the variant is faster).

| R | eager | compile (no graph) | compile + graph | graphs, 1 stream | graphs, R streams | batched graph | batched compile+graph |
|---|---|---|---|---|---|---|---|
| 1 | 0.27x | 0.33x | 0.03x | 0.04x | - | - | 0.03x |
| 8 | 0.43x | - | - | 0.08x | 0.03x | 0.03x | 0.02x |

Gate: executor vs eager loss max rel 5.99e-04, weights max rel L2 7.47e-05; vs float64: executor 8.71e-04, eager 6.87e-04; loss 6.1037 -> 0.0070; tasks/step/run 465

### mlp_size_256: hidden 256, depth 4, 50 steps, best of 7 [median .. worst], us per step (all R runs advance one step)

| R | eager | compile (no graph) | compile + graph | graphs, 1 stream | graphs, R streams | batched graph | batched compile+graph | executor |
|---|---|---|---|---|---|---|---|---|
| 1 | 1051.2 [1058.5 .. 1207.2] | 1001.3 [1020.6 .. 1138.3] | 105.6 [105.6 .. 135.7] | 167.5 [167.6 .. 215.3] | - | - | 102.6 [102.7 .. 132.0] | 1154.9 [1160.1 .. 1363.1] |
| 8 | 8883.5 [9511.5 .. 9569.5] | - | - | 1365.1 [1365.3 .. 1742.8] | 302.8 [304.3 .. 396.3] | 245.0 [245.0 .. 310.7] | 136.7 [136.8 .. 173.1] | 5500.5 [5533.2 .. 5582.9] |

Ratio = variant best / executor best (above 1: the executor is faster; below 1: the variant is faster).

| R | eager | compile (no graph) | compile + graph | graphs, 1 stream | graphs, R streams | batched graph | batched compile+graph |
|---|---|---|---|---|---|---|---|
| 1 | 0.91x | 0.87x | 0.09x | 0.15x | - | - | 0.09x |
| 8 | 1.62x | - | - | 0.25x | 0.06x | 0.04x | 0.02x |

Gate: executor vs eager loss max rel 1.00e-03, weights max rel L2 4.59e-04; vs float64: executor 8.23e-04, eager 6.68e-04; loss 4.7986 -> 0.0100; tasks/step/run 157

### mlp_steps_10: hidden 72, depth 2, 10 steps, best of 7 [median .. worst], us per step (all R runs advance one step)

| R | eager | compile (no graph) | compile + graph | graphs, 1 stream | graphs, R streams | batched graph | batched compile+graph | executor |
|---|---|---|---|---|---|---|---|---|
| 1 | 751.4 [780.4 .. 806.5] | 846.8 [879.2 .. 900.5] | 77.2 [77.7 .. 78.1] | 141.4 [141.5 .. 142.4] | - | - | 75.2 [75.5 .. 75.5] | 690.7 [694.9 .. 701.8] |
| 8 | 5663.5 [5702.2 .. 5741.5] | - | - | 1071.3 [1111.3 .. 1121.3] | 310.0 [313.0 .. 325.0] | 184.4 [191.6 .. 193.2] | 89.9 [93.2 .. 94.1] | 2326.7 [2357.2 .. 2382.4] |

Ratio = variant best / executor best (above 1: the executor is faster; below 1: the variant is faster).

| R | eager | compile (no graph) | compile + graph | graphs, 1 stream | graphs, R streams | batched graph | batched compile+graph |
|---|---|---|---|---|---|---|---|
| 1 | 1.09x | 1.23x | 0.11x | 0.20x | - | - | 0.11x |
| 8 | 2.43x | - | - | 0.46x | 0.13x | 0.08x | 0.04x |

Gate: executor vs eager loss max rel 1.24e-05, weights max rel L2 9.87e-05; vs float64: executor 1.31e-05, eager 1.65e-05; loss 4.5369 -> 4.1404; tasks/step/run 51

### mlp_steps_1000: hidden 72, depth 2, 1000 steps, best of 7 [median .. worst], us per step (all R runs advance one step)

| R | eager | compile (no graph) | compile + graph | graphs, 1 stream | graphs, R streams | batched graph | batched compile+graph | executor |
|---|---|---|---|---|---|---|---|---|
| 1 | 770.7 [808.9 .. 828.8] | 937.1 [962.4 .. 968.8] | 75.8 [76.0 .. 76.0] | 120.7 [130.9 .. 139.9] | - | - | 57.2 [57.7 .. 64.4] | 489.2 [490.4 .. 490.8] |
| 8 | 5815.8 [5937.5 .. 6382.1] | - | - | 883.7 [886.1 .. 912.6] | 283.8 [286.7 .. 313.8] | 148.1 [148.2 .. 148.2] | 71.7 [71.7 .. 71.7] | 1486.2 [1487.2 .. 1491.6] |

Ratio = variant best / executor best (above 1: the executor is faster; below 1: the variant is faster).

| R | eager | compile (no graph) | compile + graph | graphs, 1 stream | graphs, R streams | batched graph | batched compile+graph |
|---|---|---|---|---|---|---|---|
| 1 | 1.58x | 1.92x | 0.16x | 0.25x | - | - | 0.12x |
| 8 | 3.91x | - | - | 0.59x | 0.19x | 0.10x | 0.05x |

Gate: executor vs eager loss max rel 9.46e-04, weights max rel L2 4.08e-05; vs float64: executor 9.15e-04, eager 8.31e-04; loss 4.5369 -> 0.0046; tasks/step/run 51

### mlp_steps_200: hidden 72, depth 2, 200 steps, best of 7 [median .. worst], us per step (all R runs advance one step)

| R | eager | compile (no graph) | compile + graph | graphs, 1 stream | graphs, R streams | batched graph | batched compile+graph | executor |
|---|---|---|---|---|---|---|---|---|
| 1 | 724.7 [727.9 .. 736.7] | 820.7 [824.6 .. 830.2] | 58.9 [58.9 .. 75.7] | 108.7 [108.7 .. 139.8] | - | - | 57.4 [57.4 .. 73.9] | 515.5 [518.3 .. 581.4] |
| 8 | 5905.4 [6451.0 .. 6533.1] | - | - | 862.4 [862.5 .. 1094.1] | 285.9 [286.4 .. 289.8] | 147.9 [148.0 .. 148.0] | 71.4 [71.5 .. 71.5] | 1675.1 [1683.2 .. 1692.7] |

Ratio = variant best / executor best (above 1: the executor is faster; below 1: the variant is faster).

| R | eager | compile (no graph) | compile + graph | graphs, 1 stream | graphs, R streams | batched graph | batched compile+graph |
|---|---|---|---|---|---|---|---|
| 1 | 1.41x | 1.59x | 0.11x | 0.21x | - | - | 0.11x |
| 8 | 3.53x | - | - | 0.51x | 0.17x | 0.09x | 0.04x |

Gate: executor vs eager loss max rel 3.16e-04, weights max rel L2 4.53e-05; vs float64: executor 3.51e-04, eager 3.46e-04; loss 4.5369 -> 0.1002; tasks/step/run 51

### gpt_162m: 162322944 params, 4096 tokens/step, 3 steps, us per optimizer step, best [median .. worst]

| implementation | us/step | tokens/s | executor / this |
|---|---|---|---|
| eager | 70602.4 [70634.8 .. 96244.0] | 58,015 | 17.73x |
| whole-step graph | 68245.2 [68323.3 .. 68388.2] | 60,019 | 18.34x |
| compile + graph | 59097.4 [59115.2 .. 59163.7] | 69,309 | 21.18x |
| executor | 1251526.3 [1253104.7 .. 1258958.3] | 3,273 | 1.00x |

Gate: executor vs eager loss max rel 2.32e-05; modes vs eager {'eager': 0.0, 'graph': 6.27e-05, 'compile_graph': 0.000131}

### gpt_small: 116480 params, 128 tokens/step, 50 steps, us per optimizer step, best [median .. worst]

| implementation | us/step | tokens/s | executor / this |
|---|---|---|---|
| eager | 4816.1 [5138.8 .. 5201.5] | 26,578 | 0.50x |
| whole-step graph | 646.0 [646.1 .. 646.1] | 198,130 | 3.76x |
| compile (no graph) | 3089.1 [3537.2 .. 3683.2] | 41,436 | 0.79x |
| compile + graph | 331.6 [331.6 .. 331.7] | 386,065 | 7.33x |
| executor | 2429.1 [2436.8 .. 2439.6] | 52,695 | 1.00x |

Gate: executor vs eager loss max rel 7.28e-04; modes vs eager {'eager': 0.0, 'graph': 0.0, 'compile_mat': 0.00311, 'compile_graph': 0.00311}

### gpt_tiny: 16864 params, 32 tokens/step, 50 steps, us per optimizer step, best [median .. worst]

| implementation | us/step | tokens/s | executor / this |
|---|---|---|---|
| eager | 3011.3 [3060.3 .. 3196.2] | 10,627 | 0.34x |
| whole-step graph | 363.8 [427.8 .. 428.2] | 87,969 | 2.83x |
| compile (no graph) | 2046.8 [2284.0 .. 2400.2] | 15,634 | 0.50x |
| compile + graph | 172.6 [202.8 .. 203.2] | 185,350 | 5.97x |
| executor | 1031.2 [1159.1 .. 1161.9] | 31,032 | 1.00x |

Gate: executor vs eager loss max rel 1.26e-04; modes vs eager {'eager': 0.0, 'graph': 0.0, 'compile_mat': 0.00126, 'compile_graph': 0.00126}

