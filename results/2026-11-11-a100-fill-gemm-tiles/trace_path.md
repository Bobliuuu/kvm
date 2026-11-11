# Where the step-down's small-token loss goes

One traced plain run (`--warm-ms 500 --trace-csv`) of each arm on the read's executor (6bf2bc1b), A100-SXM4-80GB, Nov 11 15:47Z: #41's bundle
against #41 + this PR's, for the read's slowest shape and its fastest. Each run's critical path is the longest path through the bundle's task
graph (successor lists and every publish/wait pair), each task weighted by its own measured body from the trace plus 1 µs a hop; along it,
the sums below are measured only: bodies, and each hop from a predecessor's publish to the next task's start. Each GEMM on the path is
classed by its flags in the bundle.

| shape | arm | run | path tasks | path GEMM bodies | path hops (median) | path span | GEMMs on the path, by tile | GEMM tasks in all, by tile |
|---|---|---|---|---|---|---|---|---|
| `ho-gpt-d256-h4-l2-t32-b2-v128` (64 tokens a step, static path) | #41 | 23.069 ms | 1,424 | 10,734 µs | 1,197 µs (0.67 µs) | 23,027 µs | 660 64x64 (K 128, 256, 1,024) | 8,580 64x64, 1,920 128x128 |
| | #41 + this PR | 23.317 ms | 1,426 | 10,767 µs | 1,418 µs (0.70 µs) | 23,286 µs | 660 64x64 (the same K) | 16,260 64x64 |
| `ho-gpt-d512-h8-l2-t128-b4-v512` (512 tokens a step, dynamic path) | #41 | 100.922 ms | 1,386 | 53,854 µs | 9,342 µs (3.58 µs) | 99,700 µs | 540 128x128 (K 512, 2,048), 120 256x128 | 14,880 128x128, 7,680 256x128 |
| | #41 + this PR | 79.323 ms | 1,385 | 29,937 µs | 11,172 µs (3.62 µs) | 77,366 µs | 540 64x64 (the same K), 120 256x128 | 59,520 64x64, 7,680 256x128 |

On the 64-token shape every GEMM on the critical path was 64x64 before this PR: the 1,920 it steps down are all off the path, so the path's
GEMM bodies stay flat while its hops grow by 221 µs (+18%) beside 6,002 more tasks. On the 512-token shape it steps down 540 of the path's
660 GEMMs, whose bodies fall from 53.9 to 29.9 ms.
