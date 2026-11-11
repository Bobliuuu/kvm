# Where the step-down's small-token loss goes

One traced plain run (`--warm-ms 500 --trace-csv`) of each arm on the read's executor (6bf2bc1b), A100-SXM4-80GB, Nov 11 15:47Z and 15:56Z:
#41's bundle against #41 + this PR's, for the four shapes the read did not speed up, its fastest and its slowest gain. Each run's critical path is the longest path through the bundle's task
graph (successor lists and every publish/wait pair), each task weighted by its own measured body from the trace plus 1 µs a hop; along it,
the sums below are measured only: bodies, and each hop from a predecessor's publish to the next task's start. Each GEMM on the path is
classed by its flags in the bundle.

| shape | arm | run | path tasks | path GEMM bodies | path hops (median) | path span | GEMMs on the path, by tile | GEMM tasks in all, by tile |
|---|---|---|---|---|---|---|---|---|
| `ho-gpt-d256-h4-l2-t32-b2-v128` (64 tokens a step, static path) | #41 | 23.069 ms | 1,424 | 10,734 µs | 1,197 µs (0.67 µs) | 23,027 µs | 660 64x64 (K 128, 256, 1,024) | 8,580 64x64, 1,920 128x128 |
| | #41 + this PR | 23.317 ms | 1,426 | 10,767 µs | 1,418 µs (0.70 µs) | 23,286 µs | 660 64x64 (the same K) | 16,260 64x64 |
| `ho-gpt-d512-h8-l2-t128-b4-v512` (512 tokens a step, dynamic path) | #41 | 100.922 ms | 1,386 | 53,854 µs | 9,342 µs (3.58 µs) | 99,700 µs | 540 128x128 (K 512, 2,048), 120 256x128 | 14,880 128x128, 7,680 256x128 |
| | #41 + this PR | 79.323 ms | 1,385 | 29,937 µs | 11,172 µs (3.62 µs) | 77,366 µs | 540 64x64 (the same K), 120 256x128 | 59,520 64x64, 7,680 256x128 |
| `ho-gpt-d384-h6-l1-t64-b1-v64` (64 tokens, static) | #41 | 17.703 ms | 876 | 7,420 µs | 850 µs (0.64 µs) | 17,683 µs | 360 64x64 | 3,630 64x64, 3,240 128x128 |
| | #41 + this PR | 17.750 ms | 876 | 7,414 µs | 906 µs (0.67 µs) | 17,728 µs | 360 64x64 | 7,950 64x64, 2,160 128x128 |
| `ho-gpt-d512-h8-l4-t32-b1-v128` (32 tokens, static) | #41 | 62.216 ms | 2,544 | 37,280 µs | 3,805 µs (0.70 µs) | 62,203 µs | 1,260 64x64 | 18,060 64x64, 23,040 128x128 |
| | #41 + this PR | 62.775 ms | 2,545 | 37,050 µs | 4,369 µs (0.70 µs) | 62,759 µs | 1,260 64x64 | 48,780 64x64, 15,360 128x128 |
| `ho-gpt-d192-h3-l1-t32-b1-v64` (32 tokens, static) | #41 | 9.693 ms | 872 | 4,230 µs | 629 µs (0.61 µs) | 9,671 µs | 360 64x64 | 2,910 64x64, 720 128x128 |
| | #41 + this PR | 9.675 ms | 872 | 4,194 µs | 672 µs (0.64 µs) | 9,658 µs | 360 64x64 | 5,070 64x64 |
| `ho-gpt-d512-h8-l1-t64-b2-v256` (128 tokens, static) | #41 | 27.813 ms | 882 | 12,489 µs | 926 µs (0.61 µs) | 27,793 µs | 300 64x64, 60 128x128 | 5,520 64x64, 6,960 128x128 |
| | #41 + this PR | 25.507 ms | 881 | 9,890 µs | 977 µs (0.64 µs) | 25,485 µs | 360 64x64 | 18,000 64x64, 3,840 128x128 |

On the four shapes the read did not speed up, every GEMM on the critical path was 64x64 before this PR: the GEMMs it steps down are all
off the path, so the path's GEMM bodies stay flat while its hops grow (by 43-564 µs). On `ho-gpt-d512-h8-l2-t128-b4-v512` it steps down
540 of the path's 660 GEMMs, whose bodies fall from 53.9 to 29.9 ms; on `ho-gpt-d512-h8-l1-t64-b2-v256`, the read's smallest gain, 60, from
12.5 to 9.9 ms.
