# Doom (serial vs split tick) beside the GEMM+AdamW training tenant, executor scheduler modes (A100, 300 frames per run, unpaced)

Tenant: `make_workload.py --tenant train`, no spin task. Medium = 6 chains x 60 rounds (186k tasks, 87 CTA-s, 85 GB moved per run); heavy = 16 chains x 25 rounds. Every run's frames were compared with the CPU oracle (raw bytes, fingerprint, state): all identical (column `exact`).
`gaps>28.6ms` = tick-to-tick gaps longer than one 35 Hz period; `late` = ticks flagged missed against their absolute deadline (f+1)/35 s (the first ~6 are the 170-330 ms boot tick).

## Medium tenant, 3 repetitions
| config | n | exact | fps median [min,max] | tick gap p50 ms | tick gap p99 ms | gaps>28.6ms (sum) | late (sum) | makespan ms |
|---|---|---|---|---|---|---|---|---|
| serial_alone | 3 | True | 170.4 [170.3,170.6] | 6.23 | 10.20 | 0 | 19 | 1926 |
| serial_orig | 3 | True | 136.8 [136.2,137.5] | 7.27 | 18.37 | 0 | 45 | 2461 |
| serial_global | 3 | True | 136.2 [136.0,137.1] | 7.26 | 18.58 | 0 | 46 | 2465 |
| serial_static10 | 3 | True | 153.1 [153.1,153.3] | 7.11 | 10.49 | 0 | 27 | 2203 |
| serial_static25 | 3 | True | 153.1 [153.0,153.1] | 7.21 | 10.31 | 0 | 30 | 2217 |
| serial_dyn | 3 | True | 150.2 [150.2,150.3] | 6.91 | 11.10 | 0 | 29 | 2257 |
| serial_dyn_noresv | 3 | True | 150.0 [150.0,150.1] | 6.90 | 11.19 | 0 | 29 | 2260 |
| split8_alone | 3 | True | 206.3 [206.1,206.5] | 5.02 | 8.23 | 0 | 18 | 1620 |
| split8_orig | 3 | True | 149.5 [149.3,149.5] | 5.79 | 27.99 | 3 | 93 | 2294 |
| split8_global | 3 | True | 148.8 [148.2,148.9] | 5.84 | 28.15 | 7 | 79 | 2293 |
| split8_static10 | 3 | True | 178.1 [177.9,178.2] | 6.17 | 9.45 | 0 | 27 | 1937 |
| split8_static25 | 3 | True | 177.7 [177.6,177.7] | 6.12 | 11.17 | 0 | 24 | 1923 |
| split8_dyn | 3 | True | 174.1 [174.0,174.2] | 6.14 | 9.95 | 0 | 28 | 1980 |
| split8_dyn_noresv | 3 | True | 174.3 [173.8,174.5] | 6.09 | 10.00 | 0 | 29 | 1989 |
| split16_alone | 3 | True | 206.8 [206.7,206.8] | 5.02 | 8.24 | 0 | 20 | 1617 |
| split16_orig | 3 | True | 149.5 [149.5,149.8] | 5.83 | 27.87 | 3 | 93 | 2298 |
| split16_global | 3 | True | 148.8 [146.6,149.0] | 5.86 | 28.16 | 4 | 93 | 2293 |
| split16_static10 | 3 | True | 177.3 [176.8,177.4] | 6.17 | 9.46 | 0 | 27 | 1943 |
| split16_static25 | 3 | True | 177.9 [177.8,177.9] | 6.10 | 10.70 | 0 | 28 | 1946 |
| split16_dyn | 3 | True | 174.4 [174.4,174.6] | 6.13 | 10.05 | 0 | 27 | 1973 |
| split16_dyn_noresv | 3 | True | 174.3 [173.8,174.3] | 6.20 | 9.98 | 0 | 29 | 1980 |

## Heavy tenant, 1 repetition
| config | n | exact | fps median [min,max] | tick gap p50 ms | tick gap p99 ms | gaps>28.6ms (sum) | late (sum) | makespan ms |
|---|---|---|---|---|---|---|---|---|
| serial_alone | 1 | True | 170.3 [170.3,170.3] | 6.19 | 10.18 | 0 | 7 | 1927 |
| serial_orig | 1 | True | 125.4 [125.4,125.4] | 6.99 | 39.71 | 18 | 35 | 2658 |
| serial_global | 1 | True | 125.6 [125.6,125.6] | 6.93 | 35.22 | 18 | 36 | 2668 |
| serial_static10 | 1 | True | 152.0 [152.0,152.0] | 7.26 | 10.57 | 0 | 10 | 2227 |
| serial_static25 | 1 | True | 151.7 [151.7,151.7] | 7.17 | 12.38 | 0 | 9 | 2234 |
| serial_dyn | 1 | True | 149.6 [149.6,149.6] | 7.20 | 11.27 | 0 | 9 | 2249 |
| serial_dyn_noresv | 1 | True | 148.6 [148.6,148.6] | 7.26 | 11.20 | 0 | 9 | 2257 |
| split8_alone | 1 | True | 206.3 [206.3,206.3] | 5.00 | 8.25 | 0 | 6 | 1620 |
| split8_orig | 1 | True | 142.6 [142.6,142.6] | 5.41 | 77.13 | 9 | 36 | 2406 |
| split8_global | 1 | True | 143.6 [143.6,143.6] | 5.52 | 79.78 | 9 | 37 | 2416 |
| split8_static10 | 1 | True | 176.2 [176.2,176.2] | 6.18 | 9.54 | 0 | 9 | 1961 |
| split8_static25 | 1 | True | 175.5 [175.5,175.5] | 6.11 | 10.72 | 0 | 9 | 1967 |
| split8_dyn | 1 | True | 172.6 [172.6,172.6] | 6.30 | 10.14 | 0 | 9 | 1996 |
| split8_dyn_noresv | 1 | True | 172.1 [172.1,172.1] | 6.35 | 10.23 | 0 | 9 | 1996 |
| split16_alone | 1 | True | 206.9 [206.9,206.9] | 5.06 | 8.21 | 0 | 7 | 1634 |
| split16_orig | 1 | True | 141.7 [141.7,141.7] | 5.68 | 38.09 | 19 | 36 | 2401 |
| split16_global | 1 | True | 142.2 [142.2,142.2] | 5.49 | 76.01 | 9 | 37 | 2424 |
| split16_static10 | 1 | True | 175.1 [175.1,175.1] | 6.23 | 9.78 | 0 | 9 | 1971 |
| split16_static25 | 1 | True | 175.7 [175.7,175.7] | 6.04 | 10.94 | 0 | 8 | 1943 |
| split16_dyn | 1 | True | 172.7 [172.7,172.7] | 6.35 | 10.09 | 0 | 9 | 1980 |
| split16_dyn_noresv | 1 | True | 172.0 [172.0,172.0] | 6.32 | 9.88 | 0 | 9 | 2004 |

