# The gang crossing by size: flag lines up to 64 members, the counter past them (A100)

`kvm_gang::cross` (include/kvm_gang.cuh) picks one of the two gang barriers by the gang's size: the flag-line form (`barrier_lines`) up to
`KVM_GANG_LINES_MAX` = 64 members, the counter form (`barrier`) past them. The split and grid MLP steps cross through it. Both forms are exact
barriers with the same ordering and abort contract, so the choice changes cost only, never a result: every check line below repeats digit
for digit across the arms.

Every figure here was re-derived from these raw files by the team's verifier (Nov 11). In `raw/`, the host's working folder reads
`/root/work/` and the GPU lock's own lines are removed; everything else is as the runs wrote it.

## Why: on 108 members the flag lines lost every bed

The beds (`bench/fused_mlp/grid_bed.cu`, `split_bed.cu`) on an A100-SXM4-80GB (108 SMs; `raw/grid1/gpu.txt`) run the op's own
step on one gang, check 3 steps against float64 and time 50. The two arms are the flag-line build (`bba414c`) and the counter-barrier build
(`7fd7d4a`), built on the host with nvcc 12.8.93 and the panel's numerics flags (`raw/builds/`). Each row was read in 5 rounds, the arms'
order alternating. A row would take 4 more rounds if either arm ran slow (above 1.10x its own fastest run); none did.

| bed | members | counter, µs a step | flag lines, µs a step | flag lines / counter, paired median |
|---|---|---|---|---|
| grid h1024-d2 | 108 | 133.97 | 159.93 | x1.194 |
| grid h496-d3 | 108 | 156.41 | 180.36 | x1.149 |
| grid h512-d4 | 108 | 202.41 | 230.56 | x1.139 |
| split h256-d4, 18 shares | 108 | 108.52 | 130.34 | x1.201 |
| split h256-d4, 16 shares | 96 | 101.70 | 116.29 | x1.142 |
| split h576-d1, 18 shares | 108 | 39.81 | 42.83 | x1.075 |
| grid h600-d3 (padded to 608) | 108 | (refuses: no padding) | 188.74 | |

Every run printed `check ok`, and round 1's check lines match across the arms. This includes depth 3-4 and the padded width, which no GPU
had run at 108 members before. Logs: `raw/grid1/`.

On an RTX 5070 Laptop (36 SMs) the same change had saved 2.7-3.8% on the split bed
(`results/2026-11-10-split-mlp-step/flag-lines/`). Both results hold, at their own sizes.

## The barrier alone, and inside a step

`bench/gang/barrier_bench.cu`, ns a crossing (the median of 5 launches of 4,000 crossings; data checked between crossings; `raw/barrier/`):

| members | 2 | 8 | 32 | 33 | 48 | 64 | 96 | 108 |
|---|---|---|---|---|---|---|---|---|
| counter | 2,085 | 2,621 | 2,202 | 2,212 | 2,221 | 2,216 | 2,300 | 2,355 |
| flag lines | 1,267 | 1,204 | 1,350 | 1,611 | 1,891 | 2,171 | 2,756 | 3,465 |

- **A flag-line crossing is one round trip to L2.** But every member's warps poll every line, 4 sectors a line a poll, all on the lines' L2
  slices, so its cost grows with members x lines.
- **The counter form polls one word with one thread a member,** and pays an atomic's round trip more. That costs 2.1-2.6 µs at any size.
- **Inside a step the gap is wider.** The polls of the members that already wait also slow the members still working. On the split bed
  (h256-d4, members = 6 x shares; `raw/sweep/`; the counter's time over the flag lines', paired median):

| members | 48 | 60 | 72 | 96 | 108 |
|---|---|---|---|---|---|
| split h256-d4 | x1.058 | x1.019 | x0.987 | x0.874 | x0.840 |
| grid h1024-d2 | | | | x0.919 | x0.837 |

The crossover sits between 60 and 72 members, so the bound is 64: two full lines. (Grids of h1024 below 96 members, and splits of h256-d4
below 8 shares, do not fit a member's shared memory; the sweep's verdict lists them as missing.)

## The fix, checked

- **Device test** (`tests/static/test_gang_device.cu`, `raw/test_gang_device_a100.log`): every form passes on the A100, 2,000 rounds of two
  crossings in two layouts, the control words as a clean run leaves them, and the abort path ends. The forms: the counter, the flag lines,
  `cross<8>` (so both paths run on any GPU) and `cross<64>`.
- **Registers** (`raw/builds/pod-cross/`): the split instantiations up to 60 members keep the flag-line build's 252-254. Those from 72 up
  get the counter build's 227-234. The grid, whose size is known only at run time, holds 226.
- **Beds, the cross build against each arm** (same host, 5 rounds a row, arms alternating, no slow runs, every check line identical to
  the other arm's). Each cell is the other arm's time over cross's, the paired median, so above x1 means cross is faster:

| bed | members | cross takes | against the counter build (`7fd7d4a`) | against the flag-line build (`bba414c`) |
|---|---|---|---|---|
| split h256-d4, 8 shares | 48 | flag lines | x1.058 | x1.001 |
| split h256-d4, 10 shares | 60 | flag lines | x1.021 | x0.999 |
| split h256-d4, 12 shares | 72 | counter | x0.992 | x1.013 |
| split h256-d4, 16 shares | 96 | counter | x0.994 | x1.140 |
| split h256-d4, 18 shares | 108 | counter | x0.993 | x1.196 |
| split h576-d1, 18 shares | 108 | counter | x0.992 | x1.053 |
| grid h1024-d2 | 96 | counter | x0.972 | x1.062 |
| grid h1024-d2 | 108 | counter | x0.981 | x1.172 |
| grid h496-d3 | 108 | counter | x0.976 | x1.125 |
| grid h512-d4 | 108 | counter | x0.977 | x1.103 |
| grid h600-d3 (padded) | 108 | counter | (refuses) | x1.122 |

- **Reading it:**
  - Below 64 members cross keeps the flag lines' gain over the counter, and it is the flag-line code there (x1.000 against it).
  - From 96 members up it recovers the flag lines' whole loss: the flag lines take x1.05-1.20 its time. At 72 the three forms sit within
    1.3% of each other.
  - Against the counter build `7fd7d4a`, the counter-path rows read x0.972-0.994: cross takes 1.006-1.029x its time. That tree differs in more than the barrier's form:
    `bba414c` adds the grid's padding, and both barriers answer the abort through `__syncthreads_or` instead of static shared memory. Which
    of these costs what is not measured here.
- **Builds:** each was made on the host from `bba414c` plus the change, its trees checked by git's hashes against the laptop's
  (`raw/builds/build_*.log`). This commit's `include/` differs from the shipped build's only in one comment of `kvm_gang.cuh`.
- **Two cross builds were read** (`raw/builds/build_cross1.log`, `build_cross.log`). The first read every member's flag-line word at a step's
  start, a volatile load, even on the counter path (`raw/cross1-vs-counter/`, `raw/cross1-vs-lines/`).
  - The shipped build reads it only on the flag-line path (`cross_epoch`). Against the counter build (`raw/cross-vs-counter/`) its
    counter-path split rows moved from x0.987-0.993 to x0.992-0.994; the grid's stayed within 0.5%.
  - Against the flag lines (`raw/cross-vs-lines/`) the shipped build reads x0.999-1.001 up to 60 members (the same code) and x1.013 at 72,
    where the first read x1.001 (`raw/cross1-vs-lines/`).
