## 1. Headline: dense 1T, TP8 x PP16 x DP, pipeline stages across 4 datacenters (4 stages each), ZeRO-3 over DP
Per DP rank: 64 microbatches of 2 sequences x 4096 tokens (weak scaling), 8 extra microbatches in flight per stage. WAN 100 Gb/s per datacenter pair per direction, 20 ms one way.
| GPUs | step s | MFU | HFU | LB perfect-overlap s | LB +bubble s | actual / LB+bubble | exposed comm | other idle | comm hidden | GB/GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 (WAN 100 Gb/s) | 83.4 | 30.9% | 41.2% | 60.9 | 75.1 | 1.11 | 0.5% | 26.5% | 65% | 64.4 |
| 1024 (WAN 100 Gb/s) | 83.8 | 30.7% | 41.0% | 60.8 | 75.1 | 1.12 | 0.7% | 26.7% | 53% | 47.5 |
| 2048 (WAN 100 Gb/s) | 86.6 | 29.7% | 39.6% | 60.8 | 75.1 | 1.15 | 0.7% | 29.0% | 53% | 39.1 |
| 4096 (WAN 100 Gb/s) | 96.8 | 26.6% | 35.5% | 60.8 | 75.1 | 1.29 | 1.0% | 36.2% | 31% | 34.9 |
| 4096 (WAN 400 Gb/s) | 83.8 | 30.7% | 41.0% | 60.8 | 75.1 | 1.12 | 0.8% | 26.6% | 53% | 34.9 |
| 8192 (WAN 100 Gb/s) | 166.3 | 15.5% | 20.6% | 60.8 | 75.1 | 2.21 | 0.7% | 62.7% | 17% | 32.8 |
| 8192 (WAN 400 Gb/s) | 86.6 | 29.7% | 39.7% | 60.8 | 75.1 | 1.15 | 0.7% | 29.0% | 52% | 32.8 |
| 10240 (WAN 400 Gb/s) | 89.0 | 28.9% | 38.6% | 60.8 | 75.1 | 1.19 | 0.7% | 31.0% | 54% | 32.4 |

## 2. WAN sensitivity (4096 GPUs, dense 1T, M=64): step seconds
| one-way latency \ Gb/s per pair | 50 | 100 | 200 | 400 | 800 | 1600 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 ms | 166.6 | 124.6 | 107.0 | 96.0 | 89.2 | 87.4 |
| 10 ms | 166.7 | 124.3 | 108.8 | 96.6 | 89.3 | 88.8 |
| 20 ms | 166.7 | 124.2 | 108.4 | 95.2 | 89.4 | 88.9 |
| 50 ms | 167.0 | 126.1 | 110.8 | 97.5 | 91.4 | 89.1 |
| 100 ms | 167.3 | 126.9 | 111.4 | 100.2 | 96.3 | 89.4 |

Same, with 8 extra microbatches allowed in flight per stage (`window_extra=8`, costs activation memory):
| one-way latency \ Gb/s per pair | 50 | 100 | 200 | 400 | 800 | 1600 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 ms | 166.2 | 97.5 | 84.3 | 81.6 | 81.4 | 78.3 |
| 10 ms | 166.3 | 96.5 | 83.5 | 82.1 | 83.0 | 80.5 |
| 20 ms | 166.3 | 96.8 | 86.6 | 83.8 | 83.3 | 81.9 |
| 50 ms | 166.5 | 96.6 | 87.6 | 83.3 | 82.4 | 81.8 |
| 100 ms | 166.6 | 97.5 | 89.5 | 85.2 | 82.4 | 78.9 |

## 3. Microbatches, window and activation compression across the WAN (4096 GPUs, 100 Gb/s, 20 ms)
| variant | step s | MFU | bubble measured | bubble theory | GB/GPU |
| --- | --- | --- | --- | --- | --- |
| M=16 (8 seq/mb) | 182.1 | 14.1% | 66.6% | 48.4% | 56.0 |
| M=32 | 143.6 | 17.9% | 57.6% | 31.9% | 39.6 |
| M=64 | 124.2 | 20.7% | 51.0% | 19.0% | 31.4 |
| M=128 (1 seq/mb) | 116.1 | 22.2% | 47.6% | 10.5% | 27.3 |
| M=64, window +8 | 96.8 | 26.6% | 37.1% | 19.0% | 34.9 |
| M=64, window +8, fp8 activations on the wire (WHAT-IF) | 88.7 | 29.0% | 31.4% | 19.0% | 34.9 |
| M=64, window +16, fp8 activations (WHAT-IF) | 81.1 | 31.7% | 25.0% | 19.0% | 38.4 |

## 4. Overlap (4096 GPUs, WAN 400 Gb/s, 20 ms, window +8)
| variant | step s | MFU | exposed comm | comm hidden | actual / LB+bubble |
| --- | --- | --- | --- | --- | --- |
| as designed: comm on its own streams (tp_chunks=1) | 83.8 | 30.7% | 0.8% | 53% | 1.12 |
| tp_chunks=4 (needs row-tile ops: WHAT-IF) | 76.9 | 33.5% | 0.5% | 74% | 1.02 |
| comm SM tax 0% | 78.9 | 32.6% | 0.7% | 58% | 1.10 |
| comm SM tax 15% | 94.2 | 27.3% | 0.6% | 61% | 1.15 |
| ZeRO-3 re-gather every microbatch | 87.9 | 29.3% | 6.8% | 85% | 1.17 |
| ZeRO-1 (gather after optimizer) | 81.2 | 31.7% | 0.4% | 77% | 1.08 |
| no overlap at all (all comm serialized with compute) | 85.3 | 30.2% | 1.6% | 0% | 1.14 |

## 5. Placement: which axis crosses the WAN (4096 GPUs, 400 Gb/s, 20 ms)
| placement | step s | MFU |
| --- | --- | --- |
| PP across datacenters (TP in node, DP in rack) | 83.8 | 30.7% |
| DP across datacenters (PP inside) | 135.8 | 19.0% |

## 6. MoE 1T (384 experts, top-8, 64 layers, hidden 7168), expert parallel = DP, a2a inside the rack
| GPUs | TP x PP x DP | step s | MFU | a2a + TP comm exposed | comm hidden | GB/GPU |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 | 4 x 16 x 16 | 24.0 | 16.2% | 28.9% | 54% | 30.6 |
| 4096 | 4 x 16 x 64 | 29.3 | 13.3% | 33.7% | 41% | 16.9 |
| 4096 | 2 x 16 x 128 | 58.0 | 13.4% | 36.0% | 37% | 27.1 |
| 8192 | 4 x 16 x 128 | 43.7 | 8.9% | 20.6% | 0% | 14.6 |
| 4096 (a2a bytes x0.375, WHAT-IF) | 4 x 16 x 64 | 23.9 | 16.3% | 17.5% | 37% | 16.9 |
| 8192 (a2a bytes x0.375, WHAT-IF) | 4 x 16 x 128 | 42.6 | 9.1% | 8.7% | 0% | 14.6 |

## 7. Memory per GPU, headline configuration (GB; limit 80)
| component | GB |
| --- | --- |
| state | 4.01 |
| gathered_weights | 16.06 |
| stored_activations | 10.07 |
| recompute_workspace | 0.89 |
| logits | 0.00 |
| moe_dispatch | 0.00 |
| comm_buffers | 0.21 |
| cuda_context | 2.00 |
| fragmentation_5pct | 1.66 |
| total (worst stage) | 34.9 |

## 8. Cross-check with a published configuration
Megatron-LM, 1T parameters, 3072 A100-80GB, sequence 2048, NVLink/NVSwitch in the node and 8 x 200 Gb/s HDR per node: reported 163 TFLOP/s per GPU = 52% of peak (arXiv:2104.04473, abstract and text).
Our model with TP8 x PP64 x DP6 (TP/PP from the model's recollection of that paper, NOT verified), batch 3072 sequences, A100 constants. Kernel efficiency is a free parameter:
| kernel efficiency | step s | hardware TFLOP/s per GPU (recompute counted) | HFU |
| --- | --- | --- | --- |
| 0.6 | 113.3 | 148 | 47.4% |
| 0.7 | 96.2 | 174 | 55.8% |
| 0.8 | 84.3 | 199 | 63.7% |
This brackets 163 TFLOP/s; it shows the model is not off by a large factor in a regime with the same shape, not that it is calibrated.

## 9. Stragglers: work stealing across DP replicas (dynamic admission), 64 replicas x 64 bundles, bundle 100 us
| slow replicas | static step us | stealing step us | gain |
| --- | --- | --- | --- |
| 0 at 1.00x | 858 | 858 | 1.00x |
| 1 at 0.50x | 1671 | 951 | 1.76x |
| 1 at 0.25x | 3379 | 954 | 3.54x |
| 4 at 0.50x | 1692 | 1052 | 1.61x |
| 8 at 0.70x | 1228 | 959 | 1.28x |

## 10. Failure goodput model (first order; per-GPU MTBF from 419 unexpected interruptions / 54 days / 16,384 GPUs reported for Llama 3, via secondary summaries)
| GPUs | cluster MTBF h | checkpoint s | restart s | interval | goodput |
| --- | --- | --- | --- | --- | --- |
| 4096 | 12.4 | 30 | 300 | 18 steps (27 min) | 95.8% |
| 4096 | 12.4 | 5 | 120 | 7 steps (10 min) | 98.3% |
| 10240 | 4.9 | 30 | 300 | 11 steps (16 min) | 93.0% |
| 10240 | 4.9 | 5 | 120 | 5 steps (8 min) | 97.0% |
| 16384 | 3.1 | 30 | 300 | 9 steps (14 min) | 90.9% |
| 16384 | 3.1 | 5 | 120 | 4 steps (6 min) | 96.1% |
