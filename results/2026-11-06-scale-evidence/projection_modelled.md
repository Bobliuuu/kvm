## MODELLED projection, dense 1T (not a measurement)

Model: 0.998T parameters, 128 layers, d=25472, seq 2048; TP8 x PP16 x DP, ZeRO-3 over DP, 64 microbatches of 1 sequence per replica, GEMM efficiency 50% of peak (assumption), full recompute, 1F1B bubble.


### A100-40GB (NVLink 300 GB/s/dir, 200 Gb/s NIC per GPU); no comm/compute overlap

| GPUs | hop cost per handoff | step s | compute LB s | exposed comm s | MFU | GB/GPU | fits |
|---|---|---|---|---|---|---|---|
| 512 | assumed ideal (NVLink 2 us / net 15 us) | 68.9 | 65.5 | 3.43 | 28.9% | 37.0 | yes |
| 512 | local dependent hop (measured 5.6 us) | 69.0 | 65.5 | 3.58 | 28.8% | 37.0 | yes |
| 512 | cross-executor hop, median (measured 57-78 us; 57 used) | 71.3 | 65.5 | 5.80 | 27.9% | 37.0 | yes |
| 512 | cross-executor hop, p90 (measured 203-298 us; 203 used) | 77.6 | 65.5 | 12.12 | 25.7% | 37.0 | yes |
| 1024 | assumed ideal (NVLink 2 us / net 15 us) | 69.1 | 65.5 | 3.66 | 28.8% | 21.4 | yes |
| 1024 | local dependent hop (measured 5.6 us) | 69.3 | 65.5 | 3.82 | 28.7% | 21.4 | yes |
| 1024 | cross-executor hop, median (measured 57-78 us; 57 used) | 71.5 | 65.5 | 6.04 | 27.8% | 21.4 | yes |
| 1024 | cross-executor hop, p90 (measured 203-298 us; 203 used) | 77.8 | 65.5 | 12.37 | 25.6% | 21.4 | yes |
| 2048 | assumed ideal (NVLink 2 us / net 15 us) | 69.2 | 65.5 | 3.78 | 28.8% | 13.6 | yes |
| 2048 | local dependent hop (measured 5.6 us) | 69.4 | 65.5 | 3.93 | 28.7% | 13.6 | yes |
| 2048 | cross-executor hop, median (measured 57-78 us; 57 used) | 71.6 | 65.5 | 6.17 | 27.8% | 13.6 | yes |
| 2048 | cross-executor hop, p90 (measured 203-298 us; 203 used) | 78.0 | 65.5 | 12.53 | 25.5% | 13.6 | yes |
| 4096 | assumed ideal (NVLink 2 us / net 15 us) | 69.3 | 65.5 | 3.85 | 28.7% | 9.7 | yes |
| 4096 | local dependent hop (measured 5.6 us) | 69.5 | 65.5 | 3.99 | 28.7% | 9.7 | yes |
| 4096 | cross-executor hop, median (measured 57-78 us; 57 used) | 71.7 | 65.5 | 6.25 | 27.8% | 9.7 | yes |
| 4096 | cross-executor hop, p90 (measured 203-298 us; 203 used) | 78.1 | 65.5 | 12.66 | 25.5% | 9.7 | yes |
| 8192 | assumed ideal (NVLink 2 us / net 15 us) | 69.4 | 65.5 | 3.89 | 28.7% | 7.7 | yes |
| 8192 | local dependent hop (measured 5.6 us) | 69.5 | 65.5 | 4.03 | 28.7% | 7.7 | yes |
| 8192 | cross-executor hop, median (measured 57-78 us; 57 used) | 71.8 | 65.5 | 6.32 | 27.7% | 7.7 | yes |
| 8192 | cross-executor hop, p90 (measured 203-298 us; 203 used) | 78.3 | 65.5 | 12.85 | 25.4% | 7.7 | yes |

### A100-40GB (NVLink 300 GB/s/dir, 200 Gb/s NIC per GPU); 90% of communication overlapped

| GPUs | hop cost per handoff | step s | compute LB s | exposed comm s | MFU | GB/GPU | fits |
|---|---|---|---|---|---|---|---|
| 512 | assumed ideal (NVLink 2 us / net 15 us) | 65.8 | 65.5 | 0.34 | 30.3% | 37.0 | yes |
| 512 | local dependent hop (measured 5.6 us) | 65.8 | 65.5 | 0.36 | 30.3% | 37.0 | yes |
| 512 | cross-executor hop, median (measured 57-78 us; 57 used) | 66.0 | 65.5 | 0.58 | 30.2% | 37.0 | yes |
| 512 | cross-executor hop, p90 (measured 203-298 us; 203 used) | 66.7 | 65.5 | 1.21 | 29.9% | 37.0 | yes |
| 1024 | assumed ideal (NVLink 2 us / net 15 us) | 65.8 | 65.5 | 0.37 | 30.2% | 21.4 | yes |
| 1024 | local dependent hop (measured 5.6 us) | 65.8 | 65.5 | 0.38 | 30.2% | 21.4 | yes |
| 1024 | cross-executor hop, median (measured 57-78 us; 57 used) | 66.1 | 65.5 | 0.60 | 30.1% | 21.4 | yes |
| 1024 | cross-executor hop, p90 (measured 203-298 us; 203 used) | 66.7 | 65.5 | 1.24 | 29.9% | 21.4 | yes |
| 2048 | assumed ideal (NVLink 2 us / net 15 us) | 65.8 | 65.5 | 0.38 | 30.2% | 13.6 | yes |
| 2048 | local dependent hop (measured 5.6 us) | 65.9 | 65.5 | 0.39 | 30.2% | 13.6 | yes |
| 2048 | cross-executor hop, median (measured 57-78 us; 57 used) | 66.1 | 65.5 | 0.62 | 30.1% | 13.6 | yes |
| 2048 | cross-executor hop, p90 (measured 203-298 us; 203 used) | 66.7 | 65.5 | 1.25 | 29.8% | 13.6 | yes |
| 4096 | assumed ideal (NVLink 2 us / net 15 us) | 65.9 | 65.5 | 0.38 | 30.2% | 9.7 | yes |
| 4096 | local dependent hop (measured 5.6 us) | 65.9 | 65.5 | 0.40 | 30.2% | 9.7 | yes |
| 4096 | cross-executor hop, median (measured 57-78 us; 57 used) | 66.1 | 65.5 | 0.63 | 30.1% | 9.7 | yes |
| 4096 | cross-executor hop, p90 (measured 203-298 us; 203 used) | 66.7 | 65.5 | 1.27 | 29.8% | 9.7 | yes |
| 8192 | assumed ideal (NVLink 2 us / net 15 us) | 65.9 | 65.5 | 0.39 | 30.2% | 7.7 | yes |
| 8192 | local dependent hop (measured 5.6 us) | 65.9 | 65.5 | 0.40 | 30.2% | 7.7 | yes |
| 8192 | cross-executor hop, median (measured 57-78 us; 57 used) | 66.1 | 65.5 | 0.63 | 30.1% | 7.7 | yes |
| 8192 | cross-executor hop, p90 (measured 203-298 us; 203 used) | 66.8 | 65.5 | 1.28 | 29.8% | 7.7 | yes |

### H100-80GB (NVLink 450 GB/s/dir, 400 Gb/s NIC per GPU); no comm/compute overlap

| GPUs | hop cost per handoff | step s | compute LB s | exposed comm s | MFU | GB/GPU | fits |
|---|---|---|---|---|---|---|---|
| 512 | assumed ideal (NVLink 2 us / net 15 us) | 22.7 | 20.7 | 2.07 | 27.6% | 37.0 | yes |
| 512 | local dependent hop (measured 5.6 us) | 22.9 | 20.7 | 2.22 | 27.5% | 37.0 | yes |
| 512 | cross-executor hop, median (measured 57-78 us; 57 used) | 25.1 | 20.7 | 4.45 | 25.0% | 37.0 | yes |
| 512 | cross-executor hop, p90 (measured 203-298 us; 203 used) | 31.4 | 20.7 | 10.76 | 20.0% | 37.0 | yes |
| 1024 | assumed ideal (NVLink 2 us / net 15 us) | 22.8 | 20.7 | 2.19 | 27.5% | 21.4 | yes |
| 1024 | local dependent hop (measured 5.6 us) | 23.0 | 20.7 | 2.34 | 27.3% | 21.4 | yes |
| 1024 | cross-executor hop, median (measured 57-78 us; 57 used) | 25.2 | 20.7 | 4.57 | 24.9% | 21.4 | yes |
| 1024 | cross-executor hop, p90 (measured 203-298 us; 203 used) | 31.5 | 20.7 | 10.90 | 19.9% | 21.4 | yes |
| 2048 | assumed ideal (NVLink 2 us / net 15 us) | 22.9 | 20.7 | 2.25 | 27.4% | 13.6 | yes |
| 2048 | local dependent hop (measured 5.6 us) | 23.1 | 20.7 | 2.40 | 27.3% | 13.6 | yes |
| 2048 | cross-executor hop, median (measured 57-78 us; 57 used) | 25.3 | 20.7 | 4.64 | 24.8% | 13.6 | yes |
| 2048 | cross-executor hop, p90 (measured 203-298 us; 203 used) | 31.6 | 20.7 | 10.99 | 19.9% | 13.6 | yes |
| 4096 | assumed ideal (NVLink 2 us / net 15 us) | 22.9 | 20.7 | 2.29 | 27.4% | 9.7 | yes |
| 4096 | local dependent hop (measured 5.6 us) | 23.1 | 20.7 | 2.43 | 27.2% | 9.7 | yes |
| 4096 | cross-executor hop, median (measured 57-78 us; 57 used) | 25.3 | 20.7 | 4.69 | 24.8% | 9.7 | yes |
| 4096 | cross-executor hop, p90 (measured 203-298 us; 203 used) | 31.8 | 20.7 | 11.10 | 19.8% | 9.7 | yes |
| 8192 | assumed ideal (NVLink 2 us / net 15 us) | 23.0 | 20.7 | 2.31 | 27.4% | 7.7 | yes |
| 8192 | local dependent hop (measured 5.6 us) | 23.1 | 20.7 | 2.45 | 27.2% | 7.7 | yes |
| 8192 | cross-executor hop, median (measured 57-78 us; 57 used) | 25.4 | 20.7 | 4.75 | 24.7% | 7.7 | yes |
| 8192 | cross-executor hop, p90 (measured 203-298 us; 203 used) | 31.9 | 20.7 | 11.27 | 19.7% | 7.7 | yes |

### H100-80GB (NVLink 450 GB/s/dir, 400 Gb/s NIC per GPU); 90% of communication overlapped

| GPUs | hop cost per handoff | step s | compute LB s | exposed comm s | MFU | GB/GPU | fits |
|---|---|---|---|---|---|---|---|
| 512 | assumed ideal (NVLink 2 us / net 15 us) | 20.9 | 20.7 | 0.21 | 30.1% | 37.0 | yes |
| 512 | local dependent hop (measured 5.6 us) | 20.9 | 20.7 | 0.22 | 30.1% | 37.0 | yes |
| 512 | cross-executor hop, median (measured 57-78 us; 57 used) | 21.1 | 20.7 | 0.44 | 29.8% | 37.0 | yes |
| 512 | cross-executor hop, p90 (measured 203-298 us; 203 used) | 21.7 | 20.7 | 1.08 | 28.9% | 37.0 | yes |
| 1024 | assumed ideal (NVLink 2 us / net 15 us) | 20.9 | 20.7 | 0.22 | 30.1% | 21.4 | yes |
| 1024 | local dependent hop (measured 5.6 us) | 20.9 | 20.7 | 0.23 | 30.1% | 21.4 | yes |
| 1024 | cross-executor hop, median (measured 57-78 us; 57 used) | 21.1 | 20.7 | 0.46 | 29.8% | 21.4 | yes |
| 1024 | cross-executor hop, p90 (measured 203-298 us; 203 used) | 21.7 | 20.7 | 1.09 | 28.9% | 21.4 | yes |
| 2048 | assumed ideal (NVLink 2 us / net 15 us) | 20.9 | 20.7 | 0.23 | 30.1% | 13.6 | yes |
| 2048 | local dependent hop (measured 5.6 us) | 20.9 | 20.7 | 0.24 | 30.1% | 13.6 | yes |
| 2048 | cross-executor hop, median (measured 57-78 us; 57 used) | 21.1 | 20.7 | 0.46 | 29.8% | 13.6 | yes |
| 2048 | cross-executor hop, p90 (measured 203-298 us; 203 used) | 21.8 | 20.7 | 1.10 | 28.9% | 13.6 | yes |
| 4096 | assumed ideal (NVLink 2 us / net 15 us) | 20.9 | 20.7 | 0.23 | 30.1% | 9.7 | yes |
| 4096 | local dependent hop (measured 5.6 us) | 20.9 | 20.7 | 0.24 | 30.1% | 9.7 | yes |
| 4096 | cross-executor hop, median (measured 57-78 us; 57 used) | 21.1 | 20.7 | 0.47 | 29.7% | 9.7 | yes |
| 4096 | cross-executor hop, p90 (measured 203-298 us; 203 used) | 21.8 | 20.7 | 1.11 | 28.9% | 9.7 | yes |
| 8192 | assumed ideal (NVLink 2 us / net 15 us) | 20.9 | 20.7 | 0.23 | 30.1% | 7.7 | yes |
| 8192 | local dependent hop (measured 5.6 us) | 20.9 | 20.7 | 0.25 | 30.1% | 7.7 | yes |
| 8192 | cross-executor hop, median (measured 57-78 us; 57 used) | 21.1 | 20.7 | 0.47 | 29.7% | 7.7 | yes |
| 8192 | cross-executor hop, p90 (measured 203-298 us; 203 used) | 21.8 | 20.7 | 1.13 | 28.8% | 7.7 | yes |
