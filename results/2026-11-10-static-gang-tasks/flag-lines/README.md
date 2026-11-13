# The gang barriers on an RTX 5070 Laptop (sm_120, 36 SMs)

`kvm_gang::barrier_lines` (flag lines) beside `kvm_gang::barrier` (the counter), as this branch has them: both return through the block's OR
(`__syncthreads_or`), with no static shared memory. Built with `nvcc -O3 -std=c++17 -arch=sm_120`. Every figure here is (unverified) until
the team's verifier names it.

- **`test_gang_device.log`** (`tests/static/test_gang_device.cu`, both forms): PASS.
  - Gangs of the fused steps' widths run side by side, then one gang of all 36 blocks (two lines).
  - 2,000 rounds of two crossings, every slot read checked.
  - A second launch starts from the words the first left: neither form needs a reset.
  - An aborted gang ends without a hang.
- **`barrier_bench.log`** (`bench/gang/barrier_bench.cu`), ns a crossing (unverified), the median of 5 launches of 4,000 crossings, with the data checked between crossings:

  | members | counter | flag line | ratio | ordering errors |
  |---|---|---|---|---|
  | 2 | 1,140 | 571 | 2.00 | 0 |
  | 4 | 1,166 | 660 | 1.77 | 0 |
  | 8 | 1,170 | 684 | 1.71 | 0 |
  | 16 | 1,170 | 697 | 1.68 | 0 |
  | 32 | 1,173 | 721 | 1.63 | 0 |
  | 33 | 1,174 | 792 | 1.48 | 0 |

  At 33 members the flag line polls two lines. The A100's 108 members (four lines) are read in the next GPU window.
- `gpu.txt`: the card at the time (`nvidia-smi`).
