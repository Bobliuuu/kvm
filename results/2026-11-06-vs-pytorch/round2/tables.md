### tiny: gpt_tiny  16,864 params, 32 tokens/step, 50 steps, tf32, 5150 tasks (103/step), depth 28 ops/step

roofline per step: compute 0.0 us, bytes 0.5 us, chain 36.5 us at a graph edge / 196.3 us at the executor hop; floor 36.5 us (graph-style) / 196.3 us (executor)

| run | us/step best | median | worst | tokens/s (best) | vs eager | vs graph | x floor (own chain) | loss diff vs eager: step 0 / worst step |
|---|---|---|---|---|---|---|---|---|
| eager | 2732.6 | 2754.5 | 2940.6 | 11,710 | 1.00x | 0.12x | 75.0 | 0.0e+00 / 0.0e+00 |
| graph | 333.7 | 429.1 | 429.3 | 95,882 | 8.19x | 1.00x | 9.2 | 0.0e+00 / 0.0e+00 |
| compile_mat | 1995.4 | 2005.0 | 2147.1 | 16,037 | 1.37x | 0.17x | 54.7 | 8.3e-06 / 1.3e-03 |
| compile_ro | 1652.5 | 1677.1 | 1735.9 | 19,364 | 1.65x | 0.20x | 45.3 | 2.3e-07 / 1.3e-05 |
| compile_graph | 201.7 | 201.8 | 202.0 | 158,646 | 13.55x | 1.65x | 5.5 | 8.3e-06 / 1.3e-03 |
| kvm | 1165.8 | 1254.7 | 1304.9 | 27,448 | 2.34x | 0.29x | 5.9 | 3.2e-06 / 1.1e-04 |
| kvm_b4 | 1301.0 | 1475.0 | 1497.8 | 24,596 | 2.10x | 0.26x | 6.6 | 3.2e-06 / 1.1e-04 |

losses (eager): 4.1460 3.7472 3.3766 2.9326 2.6319 2.2291 1.9021 1.6295 1.2841 ... 1.2453

### small: gpt_small  116,480 params, 128 tokens/step, 50 steps, tf32, 16200 tasks (324/step), depth 46 ops/step

roofline per step: compute 0.6 us, bytes 4.6 us, chain 59.9 us at a graph edge / 322.3 us at the executor hop; floor 59.9 us (graph-style) / 322.3 us (executor)

| run | us/step best | median | worst | tokens/s (best) | vs eager | vs graph | x floor (own chain) | loss diff vs eager: step 0 / worst step |
|---|---|---|---|---|---|---|---|---|
| eager | 4616.5 | 4640.0 | 5147.1 | 27,726 | 1.00x | 0.14x | 77.1 | 0.0e+00 / 0.0e+00 |
| graph | 644.1 | 780.1 | 828.5 | 198,741 | 7.17x | 1.00x | 10.8 | 0.0e+00 / 0.0e+00 |
| compile_mat | 3022.9 | 3073.5 | 3167.7 | 42,344 | 1.53x | 0.21x | 50.5 | 3.8e-06 / 3.1e-03 |
| compile_ro | 2108.2 | 2115.7 | 2167.6 | 60,716 | 2.19x | 0.31x | 35.2 | 0.0e+00 / 1.4e-04 |
| compile_graph | 424.2 | 424.3 | 424.5 | 301,714 | 10.88x | 1.52x | 7.1 | 3.8e-06 / 3.1e-03 |
| kvm | 2817.7 | 2827.7 | 2831.7 | 45,427 | 1.64x | 0.23x | 8.7 | 9.8e-08 / 9.9e-04 |
| kvm_b4 | 3738.2 | 3747.1 | 3753.5 | 34,241 | 1.23x | 0.17x | 11.6 | 9.8e-08 / 9.9e-04 |

losses (eager): 4.8782 3.6226 2.9145 2.1533 1.4141 0.8765 0.4096 0.2230 0.1055 ... 0.0957

### mid: gpt_mid  3,421,696 params, 1024 tokens/step, 20 steps, tf32, 155052 tasks (7753/step), depth 82 ops/step

roofline per step: compute 135.1 us, bytes 182.4 us, chain 106.7 us at a graph edge / 574.7 us at the executor hop; floor 182.4 us (graph-style) / 574.7 us (executor)

| run | us/step best | median | worst | tokens/s (best) | vs eager | vs graph | x floor (own chain) | loss diff vs eager: step 0 / worst step |
|---|---|---|---|---|---|---|---|---|
| eager | 8530.3 | 8569.2 | 9312.4 | 120,043 | 1.00x | 0.28x | 46.8 | 0.0e+00 / 0.0e+00 |
| graph | 2383.9 | 2512.0 | 2803.4 | 429,544 | 3.58x | 1.00x | 13.1 | 0.0e+00 / 0.0e+00 |
| compile_mat | 6140.6 | 6582.5 | 6781.4 | 166,760 | 1.39x | 0.39x | 33.7 | 1.1e-05 / 4.1e-02 |
| compile_ro | 2759.4 | 2802.0 | 3172.0 | 371,092 | 3.09x | 0.86x | 15.1 | 2.3e-07 / 6.5e-03 |
| compile_graph | 1677.4 | 1678.3 | 2114.8 | 610,482 | 5.09x | 1.42x | 9.2 | 1.1e-05 / 4.1e-02 |
| kvm | 29922.1 | 29955.4 | 30084.4 | 34,222 | 0.29x | 0.08x | 52.1 | 1.1e-06 / 5.6e-03 |
| kvm_b4 | 99131.4 | 99266.6 | 99587.4 | 10,330 | 0.09x | 0.02x | 172.5 | 1.1e-06 / 5.6e-03 |

losses (eager): 6.2788 5.4716 5.1020 5.4266 4.2375 4.6870 3.4729 3.0279 2.7084 2.0919 ... 2.0435

### tiny_p: gpt_tiny  16,864 params, 32 tokens/step, 50 steps, precise, 5150 tasks (103/step), depth 28 ops/step

roofline per step: compute 0.1 us, bytes 0.5 us, chain 36.5 us at a graph edge / 196.3 us at the executor hop; floor 36.5 us (graph-style) / 196.3 us (executor)

| run | us/step best | median | worst | tokens/s (best) | vs eager | vs graph | x floor (own chain) | loss diff vs eager: step 0 / worst step |
|---|---|---|---|---|---|---|---|---|
| eager | 2630.5 | 2673.7 | 2925.8 | 12,165 | 1.00x | 0.15x | 72.2 | 0.0e+00 / 0.0e+00 |
| graph | 384.6 | 434.7 | 435.2 | 83,196 | 6.84x | 1.00x | 10.6 | 0.0e+00 / 0.0e+00 |
| compile_mat | 2006.7 | 2013.9 | 2293.0 | 15,947 | 1.31x | 0.19x | 55.0 | 1.2e-07 / 2.5e-07 |
| compile_ro | 1499.2 | 1514.4 | 1683.9 | 21,345 | 1.75x | 0.26x | 41.1 | 1.2e-07 / 2.1e-07 |
| compile_graph | 226.5 | 226.5 | 226.7 | 141,300 | 11.62x | 1.70x | 6.2 | 1.2e-07 / 2.5e-07 |
| kvm | 1215.8 | 1301.0 | 1311.9 | 26,321 | 2.16x | 0.32x | 6.2 | 1.2e-07 / 2.1e-07 |
| kvm_b4 | 1395.2 | 1484.0 | 1522.8 | 22,936 | 1.89x | 0.28x | 7.1 | 1.2e-07 / 2.1e-07 |

losses (eager): 4.1460 3.7469 3.3763 2.9322 2.6323 2.2289 1.9015 1.6296 1.2837 ... 1.2447

### mid_p: gpt_mid  3,421,696 params, 1024 tokens/step, 20 steps, precise, 155052 tasks (7753/step), depth 82 ops/step

roofline per step: compute 405.2 us, bytes 182.4 us, chain 106.7 us at a graph edge / 574.7 us at the executor hop; floor 405.2 us (graph-style) / 574.7 us (executor)

| run | us/step best | median | worst | tokens/s (best) | vs eager | vs graph | x floor (own chain) | loss diff vs eager: step 0 / worst step |
|---|---|---|---|---|---|---|---|---|
| eager | 8954.0 | 9035.8 | 9312.1 | 114,362 | 1.00x | 0.42x | 22.1 | 0.0e+00 / 0.0e+00 |
| graph | 3748.7 | 3757.6 | 3952.1 | 273,161 | 2.39x | 1.00x | 9.3 | 0.0e+00 / 0.0e+00 |
| compile_mat | 6825.4 | 7069.3 | 7388.0 | 150,027 | 1.31x | 0.55x | 16.8 | 7.6e-08 / 2.8e-05 |
| compile_ro | 3308.2 | 3335.9 | 4107.7 | 309,530 | 2.71x | 1.13x | 8.2 | 1.5e-07 / 2.0e-05 |
| compile_graph | 2927.6 | 2933.7 | 3716.4 | 349,779 | 3.06x | 1.28x | 7.2 | 7.6e-08 / 2.9e-05 |
| kvm | 30184.8 | 30309.1 | 30763.5 | 33,924 | 0.30x | 0.12x | 52.5 | 1.5e-07 / 4.0e-04 |
| kvm_b4 | 111916.8 | 111992.0 | 112842.6 | 9,150 | 0.08x | 0.03x | 194.7 | 1.5e-07 / 4.0e-04 |

losses (eager): 6.2788 5.4710 5.1076 5.4727 4.2911 4.6924 3.4499 3.0081 2.7110 2.1357 ... 2.0703

