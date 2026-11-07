### tiny: gpt_tiny  16,864 params, 32 tokens/step, 50 steps, tf32, 5150 tasks (103/step), depth 28 ops/step

roofline per step: compute 0.0 us, bytes 0.5 us, chain 36.5 us at a graph edge / 196.3 us at the executor hop; floor 36.5 us (graph-style) / 196.3 us (executor)

| run | us/step best | median | worst | tokens/s (best) | vs eager | vs graph | x floor (own chain) | loss diff vs eager: step 0 / worst step |
|---|---|---|---|---|---|---|---|---|
| eager | 2897.5 | 2922.1 | 3001.0 | 11,044 | 1.00x | 0.15x | 79.5 | 0.0e+00 / 0.0e+00 |
| graph | 429.0 | 429.1 | 429.5 | 74,586 | 6.75x | 1.00x | 11.8 | 0.0e+00 / 0.0e+00 |
| compile_mat | 2300.8 | 2365.6 | 2589.7 | 13,908 | 1.26x | 0.19x | 63.1 | 8.3e-06 / 1.3e-03 |
| compile_ro | 1519.9 | 1531.4 | 1569.4 | 21,054 | 1.91x | 0.28x | 41.7 | 7.0e-01 / 7.0e-01 |
| compile_graph | 156.7 | 156.8 | 156.9 | 204,222 | 18.49x | 2.74x | 4.3 | 8.3e-06 / 1.3e-03 |
| kvm | 1273.9 | 1296.8 | 1314.9 | 25,120 | 2.27x | 0.34x | 6.5 | 3.2e-06 / 1.1e-04 |
| kvm_b4 | 1276.6 | 1488.9 | 1505.1 | 25,067 | 2.27x | 0.34x | 6.5 | 3.2e-06 / 1.1e-04 |

losses (eager): 4.1460 3.7472 3.3766 2.9326 2.6319 2.2291 1.9021 1.6295 1.2841 ... 1.2453

### small: gpt_small  116,480 params, 128 tokens/step, 50 steps, tf32, 16200 tasks (324/step), depth 46 ops/step

roofline per step: compute 0.6 us, bytes 4.6 us, chain 59.9 us at a graph edge / 322.3 us at the executor hop; floor 59.9 us (graph-style) / 322.3 us (executor)

| run | us/step best | median | worst | tokens/s (best) | vs eager | vs graph | x floor (own chain) | loss diff vs eager: step 0 / worst step |
|---|---|---|---|---|---|---|---|---|
| eager | 4569.6 | 4604.0 | 4836.0 | 28,011 | 1.00x | 0.14x | 76.3 | 0.0e+00 / 0.0e+00 |
| graph | 644.0 | 644.0 | 828.3 | 198,772 | 7.10x | 1.00x | 10.8 | 0.0e+00 / 0.0e+00 |
| compile_mat | 3148.5 | 3167.6 | 3296.5 | 40,655 | 1.45x | 0.20x | 52.6 | 3.8e-06 / 3.1e-03 |
| compile_ro | 2129.4 | 2157.4 | 2190.0 | 60,111 | 2.15x | 0.30x | 35.6 | 9.8e-01 / 9.8e-01 |
| compile_graph | 377.9 | 390.8 | 424.2 | 338,680 | 12.09x | 1.70x | 6.3 | 3.8e-06 / 3.1e-03 |
| kvm | 2821.6 | 2831.3 | 2835.1 | 45,364 | 1.62x | 0.23x | 8.8 | 9.8e-08 / 9.9e-04 |
| kvm_b4 | 3738.2 | 3745.8 | 3770.0 | 34,241 | 1.22x | 0.17x | 11.6 | 9.8e-08 / 9.9e-04 |

losses (eager): 4.8782 3.6226 2.9145 2.1533 1.4141 0.8765 0.4096 0.2230 0.1055 ... 0.0957

### mid: gpt_mid  3,421,696 params, 1024 tokens/step, 20 steps, tf32, 155052 tasks (7753/step), depth 82 ops/step

roofline per step: compute 135.1 us, bytes 182.4 us, chain 106.7 us at a graph edge / 574.7 us at the executor hop; floor 182.4 us (graph-style) / 574.7 us (executor)

| run | us/step best | median | worst | tokens/s (best) | vs eager | vs graph | x floor (own chain) | loss diff vs eager: step 0 / worst step |
|---|---|---|---|---|---|---|---|---|
| eager | 9325.1 | 9449.4 | 9509.1 | 109,812 | 1.00x | 0.26x | 51.1 | 0.0e+00 / 0.0e+00 |
| graph | 2380.6 | 2386.7 | 3005.6 | 430,145 | 3.92x | 1.00x | 13.0 | 0.0e+00 / 0.0e+00 |
| compile_mat | 6168.8 | 6190.1 | 6533.2 | 165,997 | 1.51x | 0.39x | 33.8 | 1.1e-05 / 4.1e-02 |
| compile_ro | 3036.4 | 3070.8 | 3527.0 | 337,240 | 3.07x | 0.78x | 16.6 | 6.8e-01 / 6.8e-01 |
| compile_graph | 1677.3 | 1678.0 | 2114.6 | 610,501 | 5.56x | 1.42x | 9.2 | 1.1e-05 / 4.1e-02 |
| kvm | 29883.9 | 29979.7 | 31138.5 | 34,266 | 0.31x | 0.08x | 52.0 | 1.1e-06 / 5.6e-03 |
| kvm_b4 | 99082.8 | 99198.7 | 100536.9 | 10,335 | 0.09x | 0.02x | 172.4 | 1.1e-06 / 5.6e-03 |

losses (eager): 6.2788 5.4716 5.1020 5.4266 4.2375 4.6870 3.4729 3.0279 2.7084 2.0919 ... 2.0435

### tiny_p: gpt_tiny  16,864 params, 32 tokens/step, 50 steps, precise, 5150 tasks (103/step), depth 28 ops/step

roofline per step: compute 0.1 us, bytes 0.5 us, chain 36.5 us at a graph edge / 196.3 us at the executor hop; floor 36.5 us (graph-style) / 196.3 us (executor)

| run | us/step best | median | worst | tokens/s (best) | vs eager | vs graph | x floor (own chain) | loss diff vs eager: step 0 / worst step |
|---|---|---|---|---|---|---|---|---|
| eager | 2887.3 | 2974.0 | 3210.2 | 11,083 | 1.00x | 0.12x | 79.2 | 0.0e+00 / 0.0e+00 |
| graph | 352.8 | 395.9 | 434.5 | 90,711 | 8.18x | 1.00x | 9.7 | 0.0e+00 / 0.0e+00 |
| compile_mat | 2270.9 | 2312.8 | 2435.1 | 14,091 | 1.27x | 0.16x | 62.3 | 1.2e-07 / 2.5e-07 |
| compile_ro | 1513.0 | 1667.0 | 1679.2 | 21,150 | 1.91x | 0.23x | 41.5 | 7.0e-01 / 7.0e-01 |
| compile_graph | 226.5 | 226.5 | 226.6 | 141,300 | 12.75x | 1.56x | 6.2 | 1.2e-07 / 2.5e-07 |
| kvm | 1313.7 | 1318.1 | 1325.2 | 24,359 | 2.20x | 0.27x | 6.7 | 1.2e-07 / 2.1e-07 |
| kvm_b4 | 1477.8 | 1488.4 | 1521.4 | 21,654 | 1.95x | 0.24x | 7.5 | 1.2e-07 / 2.1e-07 |

losses (eager): 4.1460 3.7469 3.3763 2.9322 2.6323 2.2289 1.9015 1.6296 1.2837 ... 1.2447

